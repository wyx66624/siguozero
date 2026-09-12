"""Transformer models for layout generation and move policy inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..board import HEADQUARTERS_COORDINATES
from ..pieces import (
    DEPLOYMENT_PIECE_SEQUENCE,
    PIECE_TYPE_INDICES,
    SETUP_COORDINATES,
    PieceType,
    PlayerSetup,
)
from .encoding import (
    ACTION_ENCODER_TYPE,
    ACTION_FEATURE_DIM,
    ACTION_PLAYER_PAD,
    BOARD_CODE_VOCAB_SIZE,
    BOARD_PAD_CODE,
    CASUALTY_SLOTS_PER_PLAYER,
    MAX_BOARD_POINTS,
    MAX_CASUALTY_BITS,
    PolicyState,
    StateTokenRecord,
    history_prefix_groups,
    policy_history_key,
)
from .history_arrays import HistoryArrayView, record_array
from .accelerator import is_accelerator, supports_pinned_memory
from .modes import MODE_SPECS, TrainingMode, mode_spec, normalize_mode
from .paged_kv import PagedKVCache, PagedKVState


DEPLOYMENT_POINT_ORDER: tuple[tuple[int, int], ...] = tuple(
    sorted(SETUP_COORDINATES)
)


def _validate_sampling_uniforms(
    uniforms: Tensor, *, state_count: int, sample_count: int
) -> None:
    """Check the sampling contract without synchronizing accelerator inputs.

    Accelerator values are checked when sampled actions return to the referee.
    Arena callers normally supply CPU uniforms from their per-game generators.
    """

    if not isinstance(uniforms, Tensor) or not uniforms.is_floating_point():
        raise ValueError("sampling_uniforms must be a floating-point tensor")
    if uniforms.shape != (state_count, sample_count, 2):
        raise ValueError(
            "sampling_uniforms must have shape [states, count, 2] "
            f"= [{state_count}, {sample_count}, 2]"
        )
    if uniforms.device.type == "cpu" and not bool(
        (torch.isfinite(uniforms) & (uniforms >= 0) & (uniforms < 1)).all()
    ):
        raise ValueError("sampling_uniforms must contain finite values in [0, 1)")


def _categorical_from_uniforms(probabilities: Tensor, uniforms: Tensor) -> Tensor:
    """Sample rows with supplied uniforms, without consuming a global RNG.

    Explicit positive weights skip masked points even at a rounded CDF plateau.
    Normalizing by the cumulative total makes the last positive entry exactly
    one; clamping handles float64 uniforms that round up when cast to float32.
    """

    probabilities = probabilities.float()
    cumulative = probabilities.cumsum(dim=-1)
    cumulative = cumulative / cumulative[..., -1:]
    # Cast on the originating device: Ascend does not support transferring a
    # CPU float64 tensor to the NPU and casting it afterwards.
    uniforms = uniforms.to(dtype=torch.float32).to(device=probabilities.device)
    uniforms = uniforms.clamp(max=1.0 - 2.0**-24)
    eligible = (cumulative.unsqueeze(-2) > uniforms.unsqueeze(-1)) & (
        probabilities.unsqueeze(-2) > 0
    )
    return eligible.to(torch.float32).argmax(dim=-1)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    board_dim: int = 256
    board_encoder_type: str = "whole_board_one_hot_linear"
    action_encoder_type: str = ACTION_ENCODER_TYPE
    temporal_dim: int = 512
    temporal_layers: int = 8
    temporal_heads: int = 8
    temporal_ffn_dim: int = 1024
    layout_dim: int = 256
    layout_layers: int = 8
    layout_heads: int = 8
    layout_ffn_dim: int = 1024
    max_transitions: int = 1000
    dropout: float = 0.0
    board_chunk_size: int = 256
    inference_board_cache_entries: int = 65536
    inference_temporal_cache_entries: int = 192
    incremental_inference: bool = True
    paged_kv_cache: bool = True
    paged_kv_length_bucket_tokens: int = 1001
    ppo_array_history: bool = True
    ppo_fixed_kv: bool = True
    ppo_cuda_graphs: bool = True
    temporal_causal_sdpa: bool = False
    ppo_tensor_learner: bool = False
    ppo_varlen_attention: bool = False
    ppo_low_precision_residual: bool = False
    ppo_sampling_graphs: bool = False
    ppo_compile_mode: str = "off"
    activation_checkpointing: bool = True
    dead_rules_enabled: bool = True

    def __post_init__(self) -> None:
        if self.action_encoder_type != ACTION_ENCODER_TYPE:
            raise ValueError("unsupported action encoder; expected coordinates_player_linear")
        if self.board_encoder_type != "whole_board_one_hot_linear":
            raise ValueError("unsupported board encoder; expected whole_board_one_hot_linear")
        if self.board_dim <= 0 or self.board_chunk_size <= 0:
            raise ValueError("board dimension and chunk size must be positive")
        if self.temporal_dim != 2 * self.board_dim:
            raise ValueError("temporal_dim must equal action_dim + board_dim")
        for dim, heads, label in (
            (self.temporal_dim, self.temporal_heads, "temporal"),
            (self.layout_dim, self.layout_heads, "layout"),
        ):
            if dim % heads:
                raise ValueError(f"{label} dimension must be divisible by heads")
        if self.max_transitions < 1:
            raise ValueError("max_transitions must be positive")
        if self.inference_board_cache_entries < 0:
            raise ValueError("inference board cache size cannot be negative")
        if self.inference_temporal_cache_entries < 0:
            raise ValueError("inference temporal cache size cannot be negative")
        if not isinstance(self.incremental_inference, bool):
            raise ValueError("incremental_inference must be a boolean")
        if not isinstance(self.paged_kv_cache, bool):
            raise ValueError("paged_kv_cache must be a boolean")
        if self.paged_kv_length_bucket_tokens <= 0:
            raise ValueError("paged KV length bucket must be positive")
        if not isinstance(self.dead_rules_enabled, bool):
            raise ValueError("dead_rules_enabled must be a boolean")
        for name in ("ppo_array_history", "ppo_fixed_kv", "ppo_cuda_graphs", "temporal_causal_sdpa"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ('ppo_tensor_learner', 'ppo_varlen_attention',
                     'ppo_low_precision_residual', 'ppo_sampling_graphs'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'{name} must be a boolean')
        if self.ppo_compile_mode not in ("off", "default", "reduce-overhead", "max-autotune-no-cudagraphs"):
            raise ValueError("invalid PPO compile mode")

    @property
    def max_sequence_tokens(self) -> int:
        return self.max_transitions + 1

    @classmethod
    def tiny(cls, *, dead_rules_enabled: bool = True) -> ModelConfig:
        """Small architecture used only by unit and accelerator smoke tests."""

        return cls(
            board_dim=32,
            temporal_dim=64,
            temporal_layers=1,
            temporal_heads=4,
            temporal_ffn_dim=128,
            layout_dim=32,
            layout_layers=1,
            layout_heads=4,
            layout_ffn_dim=64,
            max_transitions=16,
            board_chunk_size=32,
            activation_checkpointing=False,
            dead_rules_enabled=dead_rules_enabled,
        )

    @classmethod
    def main(cls, *, dead_rules_enabled: bool = True) -> ModelConfig:
        """Main temporal/layout capacity with a single whole-board projection."""

        return cls(
            board_dim=256,
            temporal_dim=512,
            temporal_layers=32,
            temporal_heads=8,
            temporal_ffn_dim=2048,
            layout_dim=256,
            layout_layers=16,
            layout_heads=8,
            layout_ffn_dim=1024,
            max_transitions=1000,
            board_chunk_size=128,
            activation_checkpointing=True,
            dead_rules_enabled=dead_rules_enabled,
        )

    @classmethod
    def extended(cls, *, dead_rules_enabled: bool = True) -> ModelConfig:
        """Extended temporal/layout capacity with the same whole-board encoder."""

        return cls(
            board_dim=256,
            temporal_dim=512,
            temporal_layers=48,
            temporal_heads=8,
            temporal_ffn_dim=2048,
            layout_dim=256,
            layout_layers=24,
            layout_heads=8,
            layout_ffn_dim=1024,
            max_transitions=1000,
            board_chunk_size=96,
            activation_checkpointing=True,
            dead_rules_enabled=dead_rules_enabled,
        )


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.up = nn.Linear(d_model, 2 * hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        gate, values = self.up(inputs).chunk(2, dim=-1)
        return self.down(self.dropout(F.silu(gate) * values))


class PreNormEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: Tensor,
        *,
        valid_mask: Tensor,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        normalized = self.attention_norm(inputs)
        if is_causal:
            # Temporal batches have valid prefixes and right padding. A valid
            # causal query cannot see a padded key, so no B*H*S*S mask is needed.
            if attention_mask is not None:
                raise ValueError('direct causal attention must not receive an explicit mask')
            qkv = F.linear(normalized, self.attention.in_proj_weight, self.attention.in_proj_bias)
            batch, tokens, width = inputs.shape
            heads = self.attention.num_heads
            q, k, v = [item.view(batch, tokens, heads, width // heads).transpose(1, 2)
                       for item in qkv.chunk(3, dim=-1)]
            attended = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attention.dropout if self.training else 0.)
            attended = attended.transpose(1, 2).contiguous().view(batch, tokens, width)
            attended = F.linear(attended, self.attention.out_proj.weight, self.attention.out_proj.bias)
            outputs = inputs + self.dropout(attended)
            outputs = outputs + self.dropout(self.ffn(self.ffn_norm(outputs)))
            return outputs.masked_fill(~valid_mask.unsqueeze(-1), 0.)
        key_padding_mask = torch.zeros(
            valid_mask.shape,
            dtype=normalized.dtype,
            device=normalized.device,
        ).masked_fill(~valid_mask, float("-inf"))
        attended = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        outputs = inputs + self.dropout(attended)
        outputs = outputs + self.dropout(self.ffn(self.ffn_norm(outputs)))
        return outputs.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

    def projected_keys_values(self, normalized_inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Project normalized tokens into per-head K/V tensors.

        This uses the exact parameters owned by ``nn.MultiheadAttention`` so
        the rollout cache remains checkpoint-compatible with the ordinary
        full-sequence learner path.
        """

        projection = F.linear(
            normalized_inputs,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        )
        _queries, keys, values = projection.chunk(3, dim=-1)
        batch, tokens, width = keys.shape
        heads = self.attention.num_heads
        head_dim = width // heads

        def split_heads(item: Tensor) -> Tensor:
            return item.view(batch, tokens, heads, head_dim).transpose(1, 2)

        return split_heads(keys), split_heads(values)

    def incremental(
        self,
        inputs: Tensor,
        *,
        past_keys: Tensor,
        past_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Evaluate one causal token and append it to an existing KV prefix."""

        if inputs.ndim != 3 or inputs.shape[1] != 1:
            raise ValueError("incremental attention requires [batch, 1, dim]")
        if past_keys.shape != past_values.shape or past_keys.ndim != 4:
            raise ValueError("past K/V tensors must share [batch, heads, time, dim]")
        queries, new_keys, new_values = self.incremental_projection(inputs)
        batch, heads, _one, head_dim = queries.shape
        width = heads * head_dim
        keys = torch.cat((past_keys, new_keys), dim=2)
        values = torch.cat((past_values, new_values), dim=2)
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch, 1, width)
        return self.incremental_output(inputs, attended), keys, values

    def incremental_projection(
        self, inputs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Project one token to per-head Q/K/V for either cache backend."""

        normalized = self.attention_norm(inputs)
        projection = F.linear(
            normalized,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        )
        queries, keys, values = projection.chunk(3, dim=-1)
        batch, tokens, width = queries.shape
        heads = self.attention.num_heads
        head_dim = width // heads

        def split_heads(item: Tensor) -> Tensor:
            return item.view(batch, tokens, heads, head_dim).transpose(1, 2)

        return split_heads(queries), split_heads(keys), split_heads(values)

    def incremental_output(self, inputs: Tensor, attended: Tensor) -> Tensor:
        """Apply the attention output projection, residual, and FFN."""

        attended = F.linear(
            attended,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        outputs = inputs + self.dropout(attended)
        outputs = outputs + self.dropout(self.ffn(self.ffn_norm(outputs)))
        return outputs


class WholeBoardEncoder(nn.Module):
    """Project one fixed, categorical whole-board vector into one history token.

    No learned point embeddings, spatial attention or point feature outputs.
    Position-major one-hot features preserve categorical IDs without treating
    their numbers as piece strength. Mode and optional casualty bits enter the
    same single linear layer.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.board_feature_dim = MAX_BOARD_POINTS * BOARD_CODE_VOCAB_SIZE
        self.mode_feature_dim = len(MODE_SPECS)
        self.casualty_feature_dim = MAX_CASUALTY_BITS if config.dead_rules_enabled else 0
        self.input_dim = (
            self.board_feature_dim + self.mode_feature_dim + self.casualty_feature_dim
        )
        self.projection = nn.Linear(self.input_dim, config.board_dim)
        # Sparse input: approximately one active feature per real board point.
        nn.init.normal_(self.projection.weight, std=MAX_BOARD_POINTS ** -0.5)
        nn.init.zeros_(self.projection.bias)

    def encode_vector(
        self,
        board_codes: Tensor,
        point_mask: Tensor,
        mode_ids: Tensor,
        casualty_bits: Tensor | None,
    ) -> Tensor:
        """Build [boards, features] directly; never allocate [boards, points, d]."""
        if board_codes.ndim != 2:
            raise ValueError("board_codes must have shape [batch, points]")
        batch, points = board_codes.shape
        if (
            batch == 0 or not 0 < points <= MAX_BOARD_POINTS
            or point_mask.shape != board_codes.shape
        ):
            raise ValueError("invalid board point tensor shape")
        if mode_ids.shape != (batch,):
            raise ValueError("mode_ids must have shape [batch]")
        if self.config.dead_rules_enabled:
            if casualty_bits is None or casualty_bits.shape != (batch, MAX_CASUALTY_BITS):
                raise ValueError(f"casualty_bits must have shape [batch, {MAX_CASUALTY_BITS}]")
        elif casualty_bits is not None:
            raise ValueError("casualty_bits must be absent when dead rules are disabled")
        dtype = (torch.get_autocast_dtype(board_codes.device.type)
                 if torch.is_autocast_enabled(board_codes.device.type) else self.projection.weight.dtype)
        vector = torch.zeros((batch, self.input_dim), dtype=dtype, device=board_codes.device)
        # Padded points have no active category; real empty points activate 0.
        codes = board_codes.masked_fill(~point_mask, BOARD_PAD_CODE)
        offsets = torch.arange(points, device=board_codes.device) * BOARD_CODE_VOCAB_SIZE
        # Invalid category IDs must fail, not alias a neighboring point/mode
        # after adding the flattened offsets. scatter checks the sentinel.
        indices = (codes + offsets).masked_fill(
            (codes < 0) | (codes >= BOARD_CODE_VOCAB_SIZE), self.input_dim
        )
        vector.scatter_(1, indices, point_mask.to(dtype))
        mode_indices = (mode_ids.unsqueeze(1) + self.board_feature_dim).masked_fill(
            (mode_ids.unsqueeze(1) < 0) | (mode_ids.unsqueeze(1) >= self.mode_feature_dim),
            self.input_dim,
        )
        vector.scatter_(1, mode_indices, 1.0)
        if casualty_bits is not None:
            vector[:, self.board_feature_dim + self.mode_feature_dim:] = casualty_bits.to(dtype)
        return vector

    def forward(
        self,
        board_codes: Tensor,
        point_mask: Tensor,
        mode_ids: Tensor,
        casualty_bits: Tensor | None,
    ) -> Tensor:
        return self.projection(self.encode_vector(board_codes, point_mask, mode_ids, casualty_bits))


class PublicActionEncoder(nn.Module):
    """Project five endpoint/actor numbers into one action history vector."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(ACTION_FEATURE_DIM, output_dim)

    def forward(self, fields: Tensor, present: Tensor) -> Tensor:
        if fields.ndim < 1 or fields.shape[-1] != ACTION_FEATURE_DIM:
            raise ValueError("action field tensor must end with 5 coordinate/player values")
        if present.shape != fields.shape[:-1] or present.dtype != torch.bool:
            raise ValueError("action presence mask must match the leading dimensions")
        values = fields.to(self.projection.weight.dtype).masked_fill(~present.unsqueeze(-1), 0)
        encoded = self.projection(values)
        # Bias and padding cannot introduce a learned initial/no-action token.
        return encoded.masked_fill(~present.unsqueeze(-1), 0)


@dataclass(slots=True)
class PolicyTensorBatch:
    states: Sequence[PolicyState]
    # Board rows are packed by valid history token, rather than materialized
    # as a mostly repeated/padded [batch, time, points] cube.
    board_codes: Tensor
    casualty_bits: Tensor | None
    token_owner: Tensor
    point_mask: Tensor
    token_mask: Tensor
    action_fields: Tensor
    action_present: Tensor
    no_interaction: Tensor
    active_mask: Tensor
    revealed_mask: Tensor
    current_player: Tensor
    mode_ids: Tensor
    valid_indices: Tensor


def collate_policy_states(
    states: Sequence[PolicyState],
    *,
    device: torch.device | str,
    dead_rules_enabled: bool = True,
) -> PolicyTensorBatch:
    if not states:
        raise ValueError("cannot collate an empty policy batch")
    if len({state.mode for state in states}) != 1:
        raise ValueError("a policy batch must use one information mode")
    spec = mode_spec(states[0].mode)
    lengths = np.asarray([len(state.records) for state in states], dtype=np.int64)
    if np.any(lengths <= 0):
        raise ValueError("a policy history cannot be empty")
    batch, time_steps, packed = len(states), int(lengths.max()), int(lengths.sum())
    width = spec.point_count + (MAX_CASUALTY_BITS if dead_rules_enabled else 0)
    requested_device = torch.device(device)
    pin_memory = supports_pinned_memory(requested_device)
    # One compact upload for observations, one for all integer indices. The
    # tensor's NumPy view lets whole history blocks fill pinned memory directly.
    raw_cpu = torch.empty((packed, width + 10), dtype=torch.int16, pin_memory=pin_memory)
    rows = raw_cpu.numpy()
    offset = 0
    for state, length in zip(states, lengths, strict=True):
        target = rows[offset:offset + length]
        if isinstance(state.records, HistoryArrayView):
            if state.records.mode is not state.mode or state.records.dead_rules != dead_rules_enabled:
                raise ValueError("history array mode/dead-rule features do not match the model")
            state.records.copy_rows(target)
        else:
            for row, record in zip(target, state.records, strict=True):
                record_array(record, state.mode, dead_rules_enabled, out=row)
        offset += length
    owners = np.repeat(np.arange(batch, dtype=np.int64), lengths)
    starts = np.cumsum(lengths) - lengths
    indices = np.arange(packed, dtype=np.int64) + np.repeat(np.arange(batch) * time_steps - starts, lengths)
    index_cpu = torch.empty(2 * packed + batch, dtype=torch.long, pin_memory=pin_memory)
    index_rows = index_cpu.numpy()
    index_rows[:packed], index_rows[packed:2 * packed], index_rows[2 * packed:] = owners, indices, lengths
    raw = raw_cpu.to(requested_device, non_blocking=pin_memory)
    index_data = index_cpu.to(requested_device, non_blocking=pin_memory)
    owner_tensor, valid_indices, length_tensor = index_data.split((packed, packed, batch))
    fields = torch.zeros((batch * time_steps, 10), dtype=torch.long, device=requested_device)
    # Padding's current-player embedding remains the established PAD index.
    fields[:, -1] = ACTION_PLAYER_PAD
    fields = fields.index_copy(0, valid_indices, raw[:, width:].long()).view(batch, time_steps, 10)
    return PolicyTensorBatch(
        states=states,
        board_codes=raw[:, :spec.point_count].long(),
        casualty_bits=raw[:, spec.point_count:width].float() if dead_rules_enabled else None,
        token_owner=owner_tensor,
        point_mask=torch.ones((batch, spec.point_count), dtype=torch.bool, device=requested_device),
        token_mask=torch.arange(time_steps, device=requested_device)[None, :] < length_tensor[:, None],
        action_fields=fields[..., :5].float(), action_present=fields[..., 5].bool(),
        no_interaction=fields[..., 6], active_mask=fields[..., 7],
        revealed_mask=fields[..., 8], current_player=fields[..., 9],
        mode_ids=torch.full((batch,), spec.mode_index, dtype=torch.long, device=requested_device),
        valid_indices=valid_indices,
    )


@dataclass(slots=True)
class PolicyFeatures:
    context: Tensor
    point_mask: Tensor


@dataclass(slots=True)
class _TemporalStateCache:
    """Immutable causal prefix used by rollout branches.

    Each branch initially references the same tensors.  Appending a token
    allocates a new K/V suffix while the parent cache remains untouched, which
    gives copy-on-write semantics without cloning a model per environment.
    """

    length: int
    layer_keys: tuple[Tensor, ...]
    layer_values: tuple[Tensor, ...]
    context: Tensor


class GamePolicyTransformer(nn.Module):
    """Board encoder plus causal transition Transformer and two-stage action head."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.board_encoder = WholeBoardEncoder(self.config)
        self.action_encoder = PublicActionEncoder(self.config.board_dim)
        temporal_dim = self.config.temporal_dim
        self.position_embedding = nn.Embedding(
            self.config.max_sequence_tokens, temporal_dim
        )
        self.mode_embedding = nn.Embedding(3, temporal_dim)
        self.no_interaction_embedding = nn.Embedding(61, temporal_dim)
        self.active_embedding = nn.Embedding(16, temporal_dim)
        self.revealed_embedding = nn.Embedding(16, temporal_dim)
        self.current_player_embedding = nn.Embedding(5, temporal_dim)
        self.temporal_layers = nn.ModuleList(
            PreNormEncoderBlock(
                temporal_dim,
                self.config.temporal_heads,
                self.config.temporal_ffn_dim,
                self.config.dropout,
            )
            for _ in range(self.config.temporal_layers)
        )
        self.temporal_norm = nn.LayerNorm(temporal_dim)
        self.source_query = nn.Sequential(
            nn.Linear(temporal_dim, self.config.board_dim),
            nn.SiLU(),
            nn.Linear(self.config.board_dim, MAX_BOARD_POINTS),
        )
        self.destination_query = nn.Sequential(
            nn.Linear(temporal_dim + MAX_BOARD_POINTS, self.config.board_dim),
            nn.SiLU(),
            nn.Linear(self.config.board_dim, MAX_BOARD_POINTS),
        )
        # The cache is explicitly enabled only for a frozen rollout actor and
        # cleared before learner mode.  It stores detached board-global tokens,
        # never tensors used for gradient computation.
        self._inference_board_cache: OrderedDict[object, Tensor] = OrderedDict()
        self._inference_board_cache_limit = 0
        self._inference_temporal_cache: OrderedDict[
            object, _TemporalStateCache | PagedKVState
        ] = OrderedDict()
        self._inference_temporal_cache_limit = 0
        self._paged_kv_requested = False
        self._paged_kv_store: PagedKVCache | None = None
        self._fixed_kv_store = None
        self._tensor_learner_callable = None
        self._packed_learner_callable = None
        self._ppo_sampling_graphs = OrderedDict()
        self.reset_board_encoding_stats()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def start_inference_board_cache(self, max_entries: int | None = None) -> None:
        """Start a fresh per-collection cache for immutable history boards."""

        self._ppo_sampling_graphs.clear()
        limit = (
            self.config.inference_board_cache_entries
            if max_entries is None
            else int(max_entries)
        )
        if limit < 0:
            raise ValueError("inference board cache size cannot be negative")
        self._inference_board_cache.clear()
        self._inference_board_cache_limit = limit
        self._inference_temporal_cache.clear()
        self._inference_temporal_cache_limit = (
            self.config.inference_temporal_cache_entries
            if self.config.incremental_inference
            else 0
        )
        self._paged_kv_requested = bool(
            self.config.paged_kv_cache
            and is_accelerator(self.device)
            and self._inference_temporal_cache_limit > 0
        )
        self._paged_kv_store = None
        self._fixed_kv_store = None
        self.reset_board_encoding_stats()

    def start_ppo_inference_cache(self, *, capacity: int, behavior_version: int) -> None:
        if not self.config.ppo_fixed_kv or not self.config.incremental_inference:
            return
        from .fixed_kv import FixedKVCache
        if (self._fixed_kv_store is not None and self._fixed_kv_store.capacity == capacity
                and self._fixed_kv_store.behavior_version == behavior_version):
            return
        self.clear_inference_board_cache()
        self._fixed_kv_store = FixedKVCache(self, capacity=capacity, behavior_version=behavior_version,
                                           cuda_graphs=self.config.ppo_cuda_graphs)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        self.clear_inference_board_cache()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def clear_inference_board_cache(self) -> None:
        self._ppo_sampling_graphs.clear()
        self._inference_board_cache.clear()
        self._inference_board_cache_limit = 0
        self._inference_temporal_cache.clear()
        self._inference_temporal_cache_limit = 0
        self._paged_kv_requested = False
        self._paged_kv_store = None
        self._fixed_kv_store = None

    def reset_inference_temporal_cache(self) -> None:
        """Drop rollout prefixes between bounded anchor waves."""

        self._inference_temporal_cache.clear()
        self._fixed_kv_store = None
        if self._paged_kv_store is not None:
            self._paged_kv_store.reset()

    def reset_board_encoding_stats(self) -> None:
        self._rollout_cache_snapshot = None
        self._history_input_states = 0
        self._history_unique_states = 0
        self._raw_board_tokens = 0
        self._board_input_tokens = 0
        self._board_unique_tokens = 0
        self._board_encoder_tokens = 0
        self._board_cache_hits = 0
        self._temporal_requested_pairs = 0
        self._temporal_computed_pairs = 0
        self._temporal_cache_hits = 0
        self._temporal_cold_states = 0
        self._temporal_incremental_tokens = 0
        self._temporal_incremental_batches = 0
        self._temporal_incremental_batch_rows = 0
        self._temporal_incremental_batch_max = 0
        self._paged_kv_valid_attention_tokens = 0
        self._paged_kv_total_attention_tokens = 0

    def board_encoding_metrics(self) -> dict[str, float]:
        raw_tokens = max(self._raw_board_tokens, 1)
        input_tokens = max(self._board_input_tokens, 1)
        unique_tokens = max(self._board_unique_tokens, 1)
        input_states = max(self._history_input_states, 1)
        return {
            "encoding/fixed_kv_enabled": float(self._fixed_kv_store is not None),
            "encoding/fixed_kv_slots": float(0 if self._fixed_kv_store is None else len(self._fixed_kv_store.entries)),
            "encoding/fixed_kv_allocated_gib": float(
                0 if self._fixed_kv_store is None or self._fixed_kv_store.storage is None
                else self._fixed_kv_store.storage.numel() * self._fixed_kv_store.storage.element_size() / 2**30),
            "encoding/fixed_kv_direct_attention": float(
                self._fixed_kv_store is not None and self._fixed_kv_store.kernels is not None),
            "encoding/cuda_graph_captures": float(0 if self._fixed_kv_store is None else self._fixed_kv_store.graph_captures),
            "encoding/cuda_graph_replays": float(0 if self._fixed_kv_store is None else self._fixed_kv_store.graph_replays),
            "encoding/history_input_states": float(self._history_input_states),
            "encoding/history_unique_states": float(self._history_unique_states),
            "encoding/raw_board_tokens": float(self._raw_board_tokens),
            "encoding/board_input_tokens": float(self._board_input_tokens),
            "encoding/board_unique_tokens": float(self._board_unique_tokens),
            "encoding/board_encoder_tokens": float(self._board_encoder_tokens),
            "encoding/within_batch_history_saved_fraction": float(
                1.0 - self._history_unique_states / input_states
            ),
            "encoding/within_batch_dedup_saved_fraction": float(
                1.0 - self._board_unique_tokens / input_tokens
            ),
            "encoding/cross_step_cache_hit_fraction": float(
                self._board_cache_hits / unique_tokens
            ),
            "encoding/board_encoder_saved_fraction": float(
                1.0 - self._board_encoder_tokens / raw_tokens
            ),
            "encoding/board_cache_entries": float(
                len(self._inference_board_cache)
            ),
            "encoding/temporal_cache_entries": float(
                len(self._inference_temporal_cache)
            ),
            "encoding/paged_kv_enabled": float(
                self._paged_kv_store is not None
            ),
            "encoding/paged_kv_pages": float(
                0
                if self._paged_kv_store is None
                else self._paged_kv_store.used_pages
            ),
            "encoding/paged_kv_peak_pages": float(
                0
                if self._paged_kv_store is None
                else self._paged_kv_store.peak_pages
            ),
            "encoding/paged_kv_capacity_pages": float(
                0
                if self._paged_kv_store is None
                else self._paged_kv_store.max_pages
            ),
            "encoding/paged_kv_allocated_gib": float(
                0.0
                if self._paged_kv_store is None
                else self._paged_kv_store.allocated_bytes / 2**30
            ),
            "encoding/temporal_cache_hits": float(self._temporal_cache_hits),
            "encoding/temporal_cold_states": float(self._temporal_cold_states),
            "encoding/temporal_incremental_tokens": float(
                self._temporal_incremental_tokens
            ),
            "encoding/temporal_incremental_batches": float(
                self._temporal_incremental_batches
            ),
            "encoding/temporal_incremental_batch_mean": float(
                0.0
                if self._temporal_incremental_batches == 0
                else self._temporal_incremental_batch_rows
                / self._temporal_incremental_batches
            ),
            "encoding/temporal_incremental_batch_max": float(
                self._temporal_incremental_batch_max
            ),
            "encoding/paged_kv_padding_fraction": float(
                0.0
                if self._paged_kv_total_attention_tokens == 0
                else 1.0
                - self._paged_kv_valid_attention_tokens
                / self._paged_kv_total_attention_tokens
            ),
            "encoding/temporal_attention_saved_fraction": float(
                0.0
                if self._temporal_requested_pairs == 0
                else 1.0
                - self._temporal_computed_pairs
                / self._temporal_requested_pairs
            ),
        }

    def train(self, mode: bool = True) -> "GamePolicyTransformer":
        if mode:
            self.clear_inference_board_cache()
        return super().train(mode)

    def _put_temporal_cache(
        self, key: object, value: _TemporalStateCache | PagedKVState
    ) -> None:
        if self._inference_temporal_cache_limit <= 0:
            return
        previous = self._inference_temporal_cache.get(key)
        if (
            previous is not None
            and previous is not value
            and isinstance(previous, PagedKVState)
            and self._paged_kv_store is not None
        ):
            self._paged_kv_store.release(previous)
        self._inference_temporal_cache[key] = value
        self._inference_temporal_cache.move_to_end(key)
        while (
            len(self._inference_temporal_cache)
            > self._inference_temporal_cache_limit
        ):
            _stale_key, stale = self._inference_temporal_cache.popitem(last=False)
            if isinstance(stale, PagedKVState):
                if self._paged_kv_store is None:
                    raise RuntimeError("paged KV state has no backing store")
                self._paged_kv_store.release(stale)

    def _get_temporal_cache(
        self, key: object
    ) -> _TemporalStateCache | PagedKVState | None:
        value = self._inference_temporal_cache.get(key)
        if value is not None:
            self._inference_temporal_cache.move_to_end(key)
        return value

    def _find_temporal_prefix(
        self, state: PolicyState
    ) -> tuple[_TemporalStateCache | PagedKVState | None, int]:
        # A player acts every two/four plies.  Eight covers the normal gap and
        # keeps lookup O(1) instead of hashing every possible prefix.
        maximum_gap = min(8, len(state.records) - 1)
        for missing in range(maximum_gap + 1):
            records = (
                state.records
                if missing == 0
                else state.records[:-missing]
            )
            cached = self._get_temporal_cache((state.mode, getattr(records, "cache_key", records)))
            if cached is not None:
                return cached, missing
        return None, len(state.records)

    def _ensure_paged_kv_store(self, dtype: torch.dtype) -> PagedKVCache | None:
        if not self._paged_kv_requested:
            return None
        if dtype not in (torch.float16, torch.bfloat16):
            if self._paged_kv_store is not None:
                raise RuntimeError("actor autocast dtype changed inside a paged KV phase")
            # Float32 evaluation is supported by the contiguous cache. Paged
            # decode kernels require half precision; keep the chosen arithmetic.
            return None
        if self._paged_kv_store is None:
            self._paged_kv_store = PagedKVCache(
                device=self.device,
                dtype=dtype,
                num_layers=len(self.temporal_layers),
                num_heads=self.config.temporal_heads,
                head_dim=(
                    self.config.temporal_dim // self.config.temporal_heads
                ),
                max_tokens=self.config.max_sequence_tokens,
                max_entries=self._inference_temporal_cache_limit,
            )
        elif self._paged_kv_store.dtype != dtype:
            raise RuntimeError("actor autocast dtype changed inside a paged KV phase")
        return self._paged_kv_store

    def _record_board_batch(
        self,
        records: Sequence[StateTokenRecord],
        mode: TrainingMode,
    ) -> tuple[Tensor, Tensor]:
        """Encode only the newly appended records of an incremental batch."""

        if not records:
            raise ValueError("record batch cannot be empty")
        spec = mode_spec(mode)
        if any(len(record.board_codes) != spec.point_count for record in records):
            raise ValueError("history board length does not match its mode")
        requested_device = self.device
        pin_memory = supports_pinned_memory(requested_device)
        codes = torch.tensor(
            [record.board_codes for record in records],
            dtype=torch.long,
            pin_memory=pin_memory,
        ).to(requested_device, non_blocking=pin_memory)
        point_mask = torch.ones(
            (len(records), spec.point_count),
            dtype=torch.bool,
            device=requested_device,
        )
        mode_ids = torch.full(
            (len(records),),
            spec.mode_index,
            dtype=torch.long,
            device=requested_device,
        )
        casualties: Tensor | None = None
        if self.config.dead_rules_enabled:
            casualty_cpu = torch.zeros(
                (len(records), MAX_CASUALTY_BITS),
                dtype=torch.float32,
                pin_memory=pin_memory,
            )
            for index, record in enumerate(records):
                if record.known_casualty_bits is None:
                    raise ValueError("dead-rule model requires casualty history")
                source = torch.tensor(
                    record.known_casualty_bits, dtype=torch.float32
                )
                if mode is TrainingMode.FOUR_DARK:
                    if source.numel() != MAX_CASUALTY_BITS:
                        raise ValueError("four-dark casualty history requires 75 bits")
                    casualty_cpu[index] = source
                elif mode is TrainingMode.DOUBLE_OPEN:
                    if source.numel() != 2 * CASUALTY_SLOTS_PER_PLAYER:
                        raise ValueError("double-open casualty history requires 50 bits")
                    casualty_cpu[index, :CASUALTY_SLOTS_PER_PLAYER] = source[
                        :CASUALTY_SLOTS_PER_PLAYER
                    ]
                    casualty_cpu[index, 2 * CASUALTY_SLOTS_PER_PLAYER :] = source[
                        CASUALTY_SLOTS_PER_PLAYER:
                    ]
                else:
                    if source.numel() != CASUALTY_SLOTS_PER_PLAYER:
                        raise ValueError("two-player casualty history requires 25 bits")
                    casualty_cpu[
                        index,
                        CASUALTY_SLOTS_PER_PLAYER : 2 * CASUALTY_SLOTS_PER_PLAYER,
                    ] = source
            casualties = casualty_cpu.to(
                requested_device, non_blocking=pin_memory
            )
        elif any(record.known_casualty_bits is not None for record in records):
            raise ValueError("non-dead-rule model must not receive casualty history")

        self._board_input_tokens += len(records)
        self._board_unique_tokens += len(records)
        self._board_encoder_tokens += len(records)
        board_globals = self.board_encoder(
            codes, point_mask, mode_ids, casualties
        )
        return board_globals, point_mask

    def _record_temporal_tokens(
        self,
        records: Sequence[StateTokenRecord],
        mode: TrainingMode,
        board_globals: Tensor,
        position: int | Sequence[int],
    ) -> Tensor:
        if isinstance(position, int):
            positions = [position] * len(records)
        else:
            positions = [int(value) for value in position]
            if len(positions) != len(records):
                raise ValueError("incremental positions must match record batch")
        if not positions or min(positions) < 0:
            raise ValueError("incremental positions must be non-negative")
        if max(positions) >= self.config.max_sequence_tokens:
            raise ValueError("incremental history exceeds model position capacity")
        pin_memory = supports_pinned_memory(self.device)
        fields = torch.zeros(
            (len(records), 1, ACTION_FEATURE_DIM), dtype=torch.float32, pin_memory=pin_memory
        )
        present = torch.zeros(
            (len(records), 1), dtype=torch.bool, pin_memory=pin_memory
        )
        for index, record in enumerate(records):
            if record.action is not None:
                fields[index, 0] = torch.tensor(
                    record.action.as_vector(mode), dtype=torch.float32
                )
                present[index, 0] = True
        fields = fields.to(self.device, non_blocking=pin_memory)
        present = present.to(self.device, non_blocking=pin_memory)
        actions = self.action_encoder(fields, present)
        mode_id = mode_spec(mode).mode_index
        scalar = lambda values: torch.tensor(
            values, dtype=torch.long, device=self.device
        ).unsqueeze(1)
        tokens = torch.cat((actions, board_globals.unsqueeze(1)), dim=-1)
        return (
            tokens
            + self.position_embedding(
                torch.tensor(
                    positions, dtype=torch.long, device=self.device
                ).unsqueeze(1)
            )
            + self.mode_embedding(
                torch.full(
                    (len(records), 1),
                    mode_id,
                    dtype=torch.long,
                    device=self.device,
                )
            )
            + self.no_interaction_embedding(
                scalar([record.no_interaction_plies for record in records])
            )
            + self.active_embedding(
                scalar([record.active_mask for record in records])
            )
            + self.revealed_embedding(
                scalar([record.revealed_mask for record in records])
            )
            + self.current_player_embedding(
                scalar([record.current_player for record in records])
            )
        )

    def _advance_temporal_group(
        self,
        states: Sequence[PolicyState],
        prefixes: Sequence[_TemporalStateCache | PagedKVState],
        missing: int,
    ) -> PolicyFeatures:
        if not states or len(states) != len(prefixes):
            raise ValueError("incremental state/cache batch mismatch")
        mode = states[0].mode
        if any(state.mode is not mode for state in states):
            raise ValueError("incremental batch must use one mode")
        if all(isinstance(prefix, PagedKVState) for prefix in prefixes):
            return self._advance_paged_temporal_group(
                states,
                [prefix for prefix in prefixes if isinstance(prefix, PagedKVState)],
                missing,
            )
        if any(isinstance(prefix, PagedKVState) for prefix in prefixes):
            raise RuntimeError("cannot mix contiguous and paged temporal caches")
        caches = list(prefixes)
        point_mask: Tensor | None = None
        if missing == 0:
            point_mask = torch.ones(
                (len(states), mode_spec(mode).point_count), dtype=torch.bool, device=self.device
            )
            return PolicyFeatures(
                torch.stack([cache.context for cache in caches]),
                point_mask,
            )

        self._temporal_incremental_batches += missing
        self._temporal_incremental_batch_rows += len(states) * missing
        self._temporal_incremental_batch_max = max(
            self._temporal_incremental_batch_max, len(states)
        )
        for offset in range(missing):
            position = caches[0].length
            if any(cache.length != position for cache in caches):
                raise ValueError("incremental cache lengths must be bucketed")
            records = [
                state.records[len(state.records) - missing + offset]
                for state in states
            ]
            board_globals, point_mask = self._record_board_batch(
                records, mode
            )
            hidden = self._record_temporal_tokens(
                records, mode, board_globals, position
            )
            next_keys: list[Tensor] = []
            next_values: list[Tensor] = []
            for layer_index, layer in enumerate(self.temporal_layers):
                past_keys = torch.stack(
                    [cache.layer_keys[layer_index] for cache in caches]
                )
                past_values = torch.stack(
                    [cache.layer_values[layer_index] for cache in caches]
                )
                hidden, keys, values = layer.incremental(
                    hidden, past_keys=past_keys, past_values=past_values
                )
                next_keys.append(keys)
                next_values.append(values)
            contexts = self.temporal_norm(hidden).squeeze(1)
            new_caches: list[_TemporalStateCache] = []
            for index, state in enumerate(states):
                new_cache = _TemporalStateCache(
                    length=position + 1,
                    layer_keys=tuple(
                        values[index].detach() for values in next_keys
                    ),
                    layer_values=tuple(
                        values[index].detach() for values in next_values
                    ),
                    context=contexts[index].detach(),
                )
                # Only retain the current tip.  Intermediate append states are
                # never queried by turn-based self-play; keeping them would
                # multiply contiguous KV memory without improving hit rate.
                if offset == missing - 1:
                    self._put_temporal_cache(
                        policy_history_key(state), new_cache
                    )
                new_caches.append(new_cache)
            caches = new_caches
            self._temporal_incremental_tokens += len(states)
            self._temporal_computed_pairs += len(states) * (position + 1)
        assert point_mask is not None
        return PolicyFeatures(contexts, point_mask)

    def _advance_paged_temporal_group(
        self,
        states: Sequence[PolicyState],
        prefixes: Sequence[PagedKVState],
        missing: int,
    ) -> PolicyFeatures:
        """Append rollout tokens with page-level COW and batched SDPA."""

        store = self._paged_kv_store
        if store is None:
            raise RuntimeError("paged temporal cache has no backing store")
        mode = states[0].mode
        caches = list(prefixes)
        point_mask: Tensor | None = None
        contexts = torch.stack([cache.context for cache in caches])
        if missing == 0:
            point_mask = torch.ones(
                (len(states), mode_spec(mode).point_count), dtype=torch.bool, device=self.device
            )
            return PolicyFeatures(contexts, point_mask)

        self._temporal_incremental_batches += missing
        self._temporal_incremental_batch_rows += len(states) * missing
        self._temporal_incremental_batch_max = max(
            self._temporal_incremental_batch_max, len(states)
        )
        # Initial prefixes belong to the LRU. Every fork belongs to this call
        # until installed in that LRU; errors at a later append offset must
        # release intermediate generations as well as partially computed tips.
        owned_caches: dict[int, PagedKVState] = {}
        try:
            for offset in range(missing):
                positions = [cache.length for cache in caches]
                records = [
                    state.records[len(state.records) - missing + offset]
                    for state in states
                ]
                board_globals, point_mask = self._record_board_batch(
                    records, mode
                )
                hidden = self._record_temporal_tokens(
                    records, mode, board_globals, positions
                )
                children, append_pages, append_offsets = store.fork_for_append(caches)
                owned_caches.update((id(child), child) for child in children)
                page_table, valid_tokens, max_tokens = store.make_page_table(
                    children
                )
                for layer_index, layer in enumerate(self.temporal_layers):
                    queries, keys, values = layer.incremental_projection(hidden)
                    store.append(
                        layer_index,
                        append_pages,
                        append_offsets,
                        keys.squeeze(2),
                        values.squeeze(2),
                    )
                    attended = store.decode(
                        layer_index,
                        queries.squeeze(2),
                        page_table,
                        valid_tokens,
                        max_tokens,
                    )
                    attended = attended.reshape(
                        len(states), 1, self.config.temporal_dim
                    )
                    hidden = layer.incremental_output(hidden, attended)
                contexts = self.temporal_norm(hidden).squeeze(1)
                for index, child in enumerate(children):
                    child.context = contexts[index].detach()

                # Borrowed prefixes remain owned by the LRU. Only release
                # parents created by an earlier offset of this same call.
                for cache in caches:
                    if id(cache) in owned_caches:
                        store.release(cache)
                        del owned_caches[id(cache)]
                caches = children
                self._temporal_incremental_tokens += len(states)
                valid_attention_tokens = sum(cache.length for cache in caches)
                self._temporal_computed_pairs += valid_attention_tokens
                self._paged_kv_valid_attention_tokens += valid_attention_tokens
                self._paged_kv_total_attention_tokens += len(caches) * max_tokens

            for state, cache in zip(states, caches, strict=True):
                key = policy_history_key(state)
                self._put_temporal_cache(key, cache)
                if self._inference_temporal_cache.get(key) is cache:
                    del owned_caches[id(cache)]
            assert point_mask is not None
            return PolicyFeatures(contexts, point_mask)
        finally:
            if owned_caches:
                retained = {id(cache) for cache in self._inference_temporal_cache.values()}
                for identity, cache in owned_caches.items():
                    # Also tolerate an exception after a cache insertion committed.
                    if identity not in retained:
                        store.release(cache)

    def _encode_incremental(
        self, states: Sequence[PolicyState]
    ) -> PolicyFeatures:
        if not states:
            raise ValueError("cannot encode an empty policy batch")
        modes = {state.mode for state in states}
        if len(modes) != 1:
            raise ValueError("a policy batch must use one information mode")

        unique_states: list[PolicyState] = []
        inverse: list[int] = []
        state_to_unique: dict[object, int] = {}
        for state in states:
            key = policy_history_key(state)
            index = state_to_unique.get(key)
            if index is None:
                index = len(unique_states)
                state_to_unique[key] = index
                unique_states.append(state)
            inverse.append(index)

        self._history_input_states += len(states)
        self._history_unique_states += len(unique_states)
        self._raw_board_tokens += sum(len(state.records) for state in states)
        self._temporal_requested_pairs += sum(
            len(state.records) * (len(state.records) + 1) // 2
            for state in unique_states
        )

        prefixes: list[_TemporalStateCache | PagedKVState | None] = []
        missing_counts: list[int] = []
        for state in unique_states:
            prefix, missing = self._find_temporal_prefix(state)
            prefixes.append(prefix)
            missing_counts.append(missing)
            if prefix is not None:
                self._temporal_cache_hits += 1

        contexts: list[Tensor | None] = [None] * len(unique_states)
        masks: list[Tensor | None] = [None] * len(unique_states)
        cold_indices = [
            index for index, prefix in enumerate(prefixes) if prefix is None
        ]
        if cold_indices:
            cold_states = [unique_states[index] for index in cold_indices]
            if (
                self._paged_kv_store is not None
                and len(cold_indices) == len(unique_states)
            ):
                required_pages = sum(
                    math.ceil(
                        len(state.records) / self._paged_kv_store.page_size
                    )
                    for state in cold_states
                )
                if self._paged_kv_store.free_pages < required_pages:
                    # After the pinned-initial-token sliding window advances,
                    # none of the old absolute-position prefixes are reusable.
                    self.reset_inference_temporal_cache()
            cold_features = self._encode_full(
                cold_states, count_history_stats=False
            )
            for row, index in enumerate(cold_indices):
                contexts[index] = cold_features.context[row]
                masks[index] = cold_features.point_mask[row]

        buckets: dict[tuple[str, int, int], list[int]] = {}
        for index, prefix in enumerate(prefixes):
            if prefix is None:
                continue
            if isinstance(prefix, PagedKVState):
                # Paged attention can mask padded tail tokens, so nearby
                # prefix lengths should share one accelerator invocation.
                # Exact-length bucketing eventually shrinks a nominal actor
                # batch to the eight replicas of a single diverse anchor.
                length_bucket = (
                    (prefix.length - 1)
                    // self.config.paged_kv_length_bucket_tokens
                )
                key = ("paged", length_bucket, missing_counts[index])
            else:
                # Contiguous tensors still require exactly matching lengths
                # before they can be stacked.
                key = ("contiguous", prefix.length, missing_counts[index])
            buckets.setdefault(key, []).append(index)
        for (_cache_kind, _length_bucket, missing), indices in buckets.items():
            bucket_states = [unique_states[index] for index in indices]
            bucket_prefixes = [prefixes[index] for index in indices]
            features = self._advance_temporal_group(
                bucket_states,
                [item for item in bucket_prefixes if item is not None],
                missing,
            )
            for row, index in enumerate(indices):
                contexts[index] = features.context[row]
                masks[index] = features.point_mask[row]

        if any(item is None for item in contexts + masks):
            raise RuntimeError("incremental cache did not produce every feature row")
        unique_features = PolicyFeatures(
            torch.stack([item for item in contexts if item is not None]),
            torch.stack([item for item in masks if item is not None]),
        )
        if len(unique_states) == len(states):
            return unique_features
        expansion = torch.tensor(inverse, dtype=torch.long, device=self.device)
        return PolicyFeatures(
            unique_features.context.index_select(0, expansion),
            unique_features.point_mask.index_select(0, expansion),
        )

    def encode(self, states: Sequence[PolicyState]) -> PolicyFeatures:
        if (self._fixed_kv_store is not None and states and not self.training and not torch.is_grad_enabled()
                and all(isinstance(state.records, HistoryArrayView) for state in states)):
            if len({state.mode for state in states}) != 1:
                raise ValueError("a policy batch must use one information mode")
            return self._fixed_kv_store.encode(states)
        cache_active = (
            self._inference_temporal_cache_limit > 0
            and not self.training
            and not torch.is_grad_enabled()
        )
        if cache_active:
            return self._encode_incremental(states)
        return self._encode_full(states)

    def _encode_full(
        self,
        states: Sequence[PolicyState],
        *,
        count_history_stats: bool = True,
        pack_prefixes: bool = False,
        fixed_slots: Sequence[int] | None = None,
    ) -> PolicyFeatures:
        if not states:
            raise ValueError("cannot encode an empty policy batch")
        if (self.config.ppo_tensor_learner and self.config.dropout == 0.
                and self.config.temporal_causal_sdpa
                and not self.config.activation_checkpointing
                and (pack_prefixes or fixed_slots is not None)
                and all(isinstance(s.records, HistoryArrayView) for s in states)):
            return self._encode_array_batch(states, pack_prefixes=pack_prefixes,
                                           fixed_slots=fixed_slots, count_history_stats=count_history_stats)
        deduplicate = not self.training or self.config.dropout == 0.0
        original_states = states
        packed = pack_prefixes and self.config.dropout == 0.0
        query_rows: list[int] = []
        query_positions: list[int] = []
        state_inverse: list[int] = []
        if packed:
            groups = history_prefix_groups(states)
            query_rows = [0] * len(original_states)
            query_positions = [len(state.records) - 1 for state in original_states]
            for row, members in enumerate(groups):
                for index in members:
                    query_rows[index] = row
            states = [original_states[members[0]] for members in groups]
        elif deduplicate:
            unique_states: list[PolicyState] = []
            state_to_unique: dict[object, int] = {}
            for state in original_states:
                key = policy_history_key(state)
                unique_offset = state_to_unique.get(key)
                if unique_offset is None:
                    unique_offset = len(unique_states)
                    state_to_unique[key] = unique_offset
                    unique_states.append(state)
                state_inverse.append(unique_offset)
            states = unique_states
        else:
            state_inverse = list(range(len(states)))
        if not packed:
            query_rows = list(range(len(states)))
            query_positions = [len(state.records) - 1 for state in states]
        if count_history_stats:
            self._history_input_states += len(original_states)
            self._history_unique_states += len(states)
            self._raw_board_tokens += sum(
                len(state.records) for state in original_states
            )
        batch = collate_policy_states(
            states,
            device=self.device,
            dead_rules_enabled=self.config.dead_rules_enabled,
        )
        batch_size, time_steps = batch.token_mask.shape
        valid_indices = batch.valid_indices
        owner_indices = batch.token_owner
        flat_codes = batch.board_codes
        flat_casualties = batch.casualty_bits
        flat_point_mask = batch.point_mask[owner_indices]
        flat_modes = batch.mode_ids[owner_indices]

        # Rollout branches and successive plies share most immutable history
        # records.  Encode each distinct board/casualty pair once per call, and
        # reuse detached global board tokens across frozen-policy calls.  With
        # dropout enabled during training, per-token dropout semantics are kept
        # by disabling the within-batch merge.
        cache_active = (
            self._inference_board_cache_limit > 0
            and not self.training
            and not torch.is_grad_enabled()
        )
        unique_keys: list[object] = []
        unique_source_offsets: list[int] = []
        inverse_offsets: list[int] = []
        key_to_unique: dict[object, int] = {}
        compact_offset = 0
        for state in states:
            if isinstance(state.records, HistoryArrayView):
                view = state.records
                record_keys = [(state.mode, view.identity, 0), *(
                    (state.mode, view.identity, position)
                    for position in range(view.window_start, view.window_start + len(view) - 1)
                )]
            else:
                record_keys = [(state.mode, record.board_codes, record.known_casualty_bits)
                               for record in state.records]
            for record_key in record_keys:
                key: object
                if deduplicate:
                    key = record_key
                else:
                    key = compact_offset
                unique_offset = key_to_unique.get(key)
                if unique_offset is None:
                    unique_offset = len(unique_keys)
                    key_to_unique[key] = unique_offset
                    unique_keys.append(key)
                    unique_source_offsets.append(compact_offset)
                inverse_offsets.append(unique_offset)
                compact_offset += 1
        self._board_input_tokens += len(inverse_offsets)
        self._board_unique_tokens += len(unique_keys)
        encode_unique: list[int] = []
        global_rows: list[Tensor | None] = [None] * len(unique_keys)
        if cache_active:
            for unique_offset, key in enumerate(unique_keys):
                cached = self._inference_board_cache.get(key)
                if cached is not None:
                    self._inference_board_cache.move_to_end(key)
                    global_rows[unique_offset] = cached
                    self._board_cache_hits += 1
                else:
                    encode_unique.append(unique_offset)
        else:
            encode_unique = list(range(len(unique_keys)))
        self._board_encoder_tokens += len(encode_unique)

        encoded_chunks: list[Tensor] = []
        # Upload once, then slice on device. Hundreds of small synchronous
        # tensor constructions otherwise serialize a long-history backward.
        encode_sources = torch.tensor(
            [unique_source_offsets[index] for index in encode_unique],
            dtype=torch.long, device=self.device,
        )
        for start in range(0, len(encode_unique), self.config.board_chunk_size):
            chunk_unique = encode_unique[
                start : start + self.config.board_chunk_size
            ]
            chunk_sources = encode_sources[start : start + self.config.board_chunk_size]
            chunk_globals = self.board_encoder(
                flat_codes.index_select(0, chunk_sources),
                flat_point_mask.index_select(0, chunk_sources),
                flat_modes.index_select(0, chunk_sources),
                (
                    None
                    if flat_casualties is None
                    else flat_casualties.index_select(0, chunk_sources)
                ),
            )
            if not cache_active:
                # Keep [boards, width] throughout the learner. Unbinding each
                # row and stacking it again creates thousands of backward nodes.
                encoded_chunks.append(chunk_globals)
                continue
            for local_offset, unique_offset in enumerate(chunk_unique):
                global_rows[unique_offset] = chunk_globals[local_offset]
                if cache_active:
                    key = unique_keys[unique_offset]
                    self._inference_board_cache[key] = (
                        chunk_globals[local_offset].detach().clone()
                    )
                    self._inference_board_cache.move_to_end(key)
                    # A continuous arena must retain recent games after the
                    # cache fills; global_rows still owns any evicted tensors
                    # needed by the current batch.
                    while len(self._inference_board_cache) > self._inference_board_cache_limit:
                        self._inference_board_cache.popitem(last=False)

        if cache_active:
            if any(row is None for row in global_rows):
                raise RuntimeError("board cache failed to provide a required token")
            unique_globals = torch.stack([row for row in global_rows if row is not None])
        else:
            unique_globals = torch.cat(encoded_chunks, dim=0)
        inverse_tensor = torch.tensor(
            inverse_offsets, dtype=torch.long, device=self.device
        )
        valid_globals = unique_globals.index_select(0, inverse_tensor)
        globals_by_token = torch.zeros(
            (batch_size * time_steps, self.config.board_dim),
            dtype=valid_globals.dtype,
            device=self.device,
        ).index_copy(0, valid_indices, valid_globals)

        board_globals = globals_by_token.view(batch_size, time_steps, -1)
        action_embeddings = self.action_encoder(
            batch.action_fields, batch.action_present
        )
        tokens = torch.cat((action_embeddings, board_globals), dim=-1)
        positions = torch.arange(time_steps, device=self.device).unsqueeze(0)
        tokens = (
            tokens
            + self.position_embedding(positions)
            + self.mode_embedding(batch.mode_ids).unsqueeze(1)
            + self.no_interaction_embedding(batch.no_interaction)
            + self.active_embedding(batch.active_mask)
            + self.revealed_embedding(batch.revealed_mask)
            + self.current_player_embedding(batch.current_player)
        )
        if self.config.ppo_low_precision_residual and torch.is_autocast_enabled(self.device.type):
            tokens = tokens.to(torch.get_autocast_dtype(self.device.type))
        causal_mask = None if self.config.temporal_causal_sdpa else torch.zeros(
            (time_steps, time_steps), dtype=tokens.dtype, device=self.device
        ).masked_fill(
            torch.triu(
                torch.ones(
                    (time_steps, time_steps), dtype=torch.bool, device=self.device
                ),
                diagonal=1,
            ),
            float("-inf"),
        )
        capture_temporal = (
            self._inference_temporal_cache_limit > 0
            and not self.training
            and not torch.is_grad_enabled()
            and not packed
            and fixed_slots is None
        )
        fixed_metadata = None
        if fixed_slots is not None:
            fixed_metadata = torch.tensor(
                [(slot, 0, len(state.records)) for slot, state in zip(fixed_slots, states, strict=True)],
                dtype=torch.long, device=self.device,
            )
        captured_keys: list[Tensor] = []
        captured_values: list[Tensor] = []
        for layer_index, layer in enumerate(self.temporal_layers):
            if fixed_metadata is not None:
                q, k, v = layer.incremental_projection(tokens)
                self._fixed_kv_store.write(layer_index, k, v, fixed_metadata)
                # Right padding cannot affect a valid causal query. This also
                # avoids projecting Q/K/V twice during cold-cache population.
                attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
                tokens = layer.incremental_output(tokens, attended.transpose(1, 2).reshape(batch_size, time_steps, -1))
                tokens = tokens.masked_fill(~batch.token_mask.unsqueeze(-1), 0.)
                continue
            if capture_temporal:
                keys, values = layer.projected_keys_values(
                    layer.attention_norm(tokens)
                )
                captured_keys.append(keys)
                captured_values.append(values)
            if self.training and self.config.activation_checkpointing:
                tokens = checkpoint(
                    lambda values, block=layer: block(
                        values,
                        valid_mask=batch.token_mask,
                        attention_mask=causal_mask,
                        is_causal=self.config.temporal_causal_sdpa,
                    ),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(
                    tokens,
                    valid_mask=batch.token_mask,
                    attention_mask=causal_mask,
                    is_causal=self.config.temporal_causal_sdpa,
                )
        tokens = self.temporal_norm(tokens)
        query_row_tensor = torch.tensor(query_rows, dtype=torch.long, device=self.device)
        query_position_tensor = torch.tensor(query_positions, dtype=torch.long, device=self.device)
        contexts = tokens[query_row_tensor, query_position_tensor]
        features = PolicyFeatures(
            contexts, batch.point_mask.index_select(0, query_row_tensor)
        )
        if capture_temporal:
            paged_store = self._ensure_paged_kv_store(captured_keys[0].dtype)
            for index, state in enumerate(states):
                length = len(state.records)
                temporal_cache: _TemporalStateCache | PagedKVState
                if paged_store is not None:
                    temporal_cache = paged_store.from_contiguous(
                        [values[index, :, :length] for values in captured_keys],
                        [values[index, :, :length] for values in captured_values],
                        length=length,
                        context=contexts[index].detach(),
                    )
                else:
                    temporal_cache = _TemporalStateCache(
                        length=length,
                        layer_keys=tuple(
                            values[index, :, :length].detach()
                            for values in captured_keys
                        ),
                        layer_values=tuple(
                            values[index, :, :length].detach()
                            for values in captured_values
                        ),
                        context=contexts[index].detach(),
                    )
                self._put_temporal_cache(
                    policy_history_key(state),
                    temporal_cache,
                )
                pairs = length * (length + 1) // 2
                if count_history_stats:
                    self._temporal_requested_pairs += pairs
                self._temporal_computed_pairs += pairs
                self._temporal_cold_states += 1
        if not packed and len(states) != len(original_states):
            expansion = torch.tensor(
                state_inverse, dtype=torch.long, device=self.device
            )
            features = PolicyFeatures(
                features.context.index_select(0, expansion),
                features.point_mask.index_select(0, expansion),
            )
        return features

    def _encode_array_batch(self, states, *, pack_prefixes, fixed_slots, count_history_stats):
        """One compact upload; CPU history objects stay outside compiled regions."""
        self._tensor_learner_calls = getattr(self, '_tensor_learner_calls', 0) + 1
        from .packed_attention import AVAILABLE, prepare
        if (self.config.ppo_varlen_attention and AVAILABLE and fixed_slots is None and self.device.type == 'cuda'
                and torch.is_autocast_enabled('cuda')
                and torch.get_autocast_dtype('cuda') in (torch.bfloat16, torch.float16)):
            arguments, lengths = prepare(self, states)
            function = self._packed_temporal_forward
            if self.training and self.config.ppo_compile_mode != 'off':
                if self._packed_learner_callable is None:
                    self._packed_learner_callable = torch.compile(
                        self._packed_temporal_forward, dynamic=True, mode=self.config.ppo_compile_mode)
                function = self._packed_learner_callable
            contexts = function(*arguments)
            self._varlen_calls = getattr(self, '_varlen_calls', 0) + 1
            if count_history_stats:
                self._history_input_states += len(states)
                self._history_unique_states += len(lengths)
                self._raw_board_tokens += sum(len(s.records) for s in states)
            self._board_input_tokens += int(lengths.sum())
            self._board_unique_tokens += int(lengths.sum())
            self._board_encoder_tokens += int(lengths.sum())
            return PolicyFeatures(contexts, torch.ones((len(states), mode_spec(states[0].mode).point_count),
                                                       dtype=torch.bool, device=self.device))
        if len({s.mode for s in states}) != 1:
            raise ValueError("a policy batch must use one information mode")
        groups = history_prefix_groups(states) if pack_prefixes else [[i] for i in range(len(states))]
        longest = [states[group[0]] for group in groups]
        lengths = [len(s.records) for s in longest]
        times = max(lengths)
        # Small alignment padding stabilizes kernels without truncating history.
        times = min(self.config.max_sequence_tokens, ((times + 31) // 32) * 32)
        spec = mode_spec(states[0].mode)
        width = spec.point_count + (MAX_CASUALTY_BITS if self.config.dead_rules_enabled else 0) + 10
        pin = supports_pinned_memory(self.device)
        raw_cpu = torch.zeros((len(groups), times, width), dtype=torch.int16, pin_memory=pin)
        raw_numpy = raw_cpu.numpy()
        for row, state in enumerate(longest):
            if state.records.dead_rules != self.config.dead_rules_enabled or state.records.mode is not state.mode:
                raise ValueError("history array mode/dead-rule features do not match the model")
            state.records.copy_rows(raw_numpy[row, :lengths[row]])
        query_rows = np.empty(len(states), dtype=np.int64)
        for row, indices in enumerate(groups):
            query_rows[indices] = row
        metadata_cpu = torch.empty(len(groups) + 2 * len(states), dtype=torch.long, pin_memory=pin)
        metadata_cpu.numpy()[:] = np.concatenate((lengths, query_rows, [len(s.records) - 1 for s in states]))
        raw = raw_cpu.to(self.device, non_blocking=pin)
        lengths_tensor, rows, positions = metadata_cpu.to(self.device, non_blocking=pin).split(
            (len(groups), len(states), len(states)))
        fixed_metadata = None if fixed_slots is None else torch.tensor(
            [(slot, 0, length) for slot, length in zip(fixed_slots, lengths, strict=True)],
            dtype=torch.long, device=self.device)
        function = self._array_temporal_forward
        if self.training and self.device.type == "cuda" and self.config.ppo_compile_mode != "off":
            if self._tensor_learner_callable is None:
                self._tensor_learner_callable = torch.compile(
                    self._array_temporal_forward, dynamic=True, mode=self.config.ppo_compile_mode,
                )
            function = self._tensor_learner_callable
            if self.config.ppo_compile_mode == 'reduce-overhead':
                torch.compiler.cudagraph_mark_step_begin()
        tokens = function(raw, lengths_tensor, states[0].mode, fixed_metadata)
        contexts = tokens[rows, positions]
        if count_history_stats:
            self._history_input_states += len(states)
            self._history_unique_states += len(groups)
            self._raw_board_tokens += sum(len(s.records) for s in states)
        actual = sum(lengths)
        self._board_input_tokens += actual
        self._board_unique_tokens += actual
        self._board_encoder_tokens += len(groups) * times
        return PolicyFeatures(contexts, torch.ones((len(states), spec.point_count),
                                                   dtype=torch.bool, device=self.device))

    def _packed_temporal_forward(self, *arguments):
        from .packed_attention import forward
        return forward(self, *arguments)

    def _array_temporal_forward(self, raw, lengths, mode, fixed_metadata=None):
        times = raw.shape[1]
        positions = torch.arange(times, device=raw.device)[None, :]
        valid = positions < lengths[:, None]
        tokens = self._embed_observation_rows(raw, mode, positions)
        for index, layer in enumerate(self.temporal_layers):
            if fixed_metadata is None:
                tokens = layer(tokens, valid_mask=valid, is_causal=True)
            else:
                q, k, v = layer.incremental_projection(tokens)
                self._fixed_kv_store.write(index, k, v, fixed_metadata)
                attended = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.)
                tokens = layer.incremental_output(tokens, attended.transpose(1, 2).reshape_as(tokens))
                tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0.)
        return self.temporal_norm(tokens)

    def _embed_observation_rows(self, rows: Tensor, mode: TrainingMode, positions: Tensor) -> Tensor:
        batch, times, _ = rows.shape
        spec = mode_spec(mode)
        flat = rows.reshape(batch * times, -1)
        width = spec.point_count + (MAX_CASUALTY_BITS if self.config.dead_rules_enabled else 0)
        board = self.board_encoder(
            flat[:, :spec.point_count].long(),
            torch.ones((batch * times, spec.point_count), dtype=torch.bool, device=rows.device),
            torch.full((batch * times,), spec.mode_index, dtype=torch.long, device=rows.device),
            flat[:, spec.point_count:width].float() if self.config.dead_rules_enabled else None,
        ).view(batch, times, -1)
        fields = rows[:, :, width:]
        actions = self.action_encoder(fields[..., :5].float(), fields[..., 5].bool())
        hidden = (torch.cat((actions, board), dim=-1) + self.position_embedding(positions)
                + self.mode_embedding(torch.full((batch, 1), spec.mode_index, dtype=torch.long, device=rows.device))
                + self.no_interaction_embedding(fields[..., 6].long())
                + self.active_embedding(fields[..., 7].long())
                + self.revealed_embedding(fields[..., 8].long())
                + self.current_player_embedding(fields[..., 9].long()))
        if self.config.ppo_low_precision_residual and torch.is_autocast_enabled(rows.device.type):
            hidden = hidden.to(torch.get_autocast_dtype(rows.device.type))
        return hidden

    def forward(
        self,
        states: Sequence[PolicyState],
        actions_by_state: Sequence[Sequence[tuple[int, int]]],
        temperature: float = 1.0,
        *,
        pack_sequences: bool = False,
        ppo_actions: Sequence[tuple[int, int]] | None = None,
    ) -> list[Tensor] | tuple[Tensor, Tensor]:
        """DDP-compatible entry point for policy likelihood training."""

        if ppo_actions is not None:
            return self.ppo_statistics(states, ppo_actions, pack_sequences=pack_sequences)
        return self.log_probs_for_action_groups(
            states, actions_by_state, temperature=temperature, pack_sequences=pack_sequences
        )

    def _source_log_probs(
        self,
        features: PolicyFeatures,
        states: Sequence[PolicyState],
        temperature: float,
        *,
        legal_masks: Tensor | None = None,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        logits = self.source_query(features.context)[:, :features.point_mask.shape[1]]
        logits = logits / temperature
        if legal_masks is None:
            legal_masks = torch.zeros(
                logits.shape, dtype=torch.bool, device="cpu"
            )
            for index, state in enumerate(states):
                legal_sources = {source for source, _target in state.legal_actions}
                legal_masks[index, list(legal_sources)] = True
            legal_masks = legal_masks.to(self.device, non_blocking=True)
        logits = logits.masked_fill(~legal_masks, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def _sampling_legal_masks(
        self,
        states: Sequence[PolicyState],
        point_count: int,
    ) -> tuple[Tensor, Tensor]:
        """Build both factorized legal masks before accelerator sampling.

        The old sampling path copied sampled source indices back to Python in
        order to discover their legal destinations, then copied a second mask
        to the device.  That introduced a synchronization in the middle of every
        actor step.  A dense board-sized mask is small (at most 129² booleans per
        state) and lets the complete two-stage sample stay on the accelerator
        until the final action pair is returned to the CPU referee.
        """

        pin_memory = supports_pinned_memory(self.device)
        destination_masks = torch.zeros(
            (len(states), point_count, point_count),
            dtype=torch.bool,
            pin_memory=pin_memory,
        )
        lengths = [len(state.legal_actions) for state in states]
        if not all(lengths):
            raise ValueError("a non-terminal policy state needs a legal action")
        pairs = np.concatenate([state.legal_array for state in states])
        owners = np.repeat(np.arange(len(states)), lengths)
        destination_masks.numpy()[owners, pairs[:, 0], pairs[:, 1]] = True
        source_masks = destination_masks.any(dim=-1)
        return (
            source_masks.to(self.device, non_blocking=pin_memory),
            destination_masks.to(self.device, non_blocking=pin_memory),
        )

    def _destination_logits(self, contexts: Tensor, sources: Tensor, point_count: int) -> Tensor:
        source_codes = F.one_hot(sources, num_classes=MAX_BOARD_POINTS).to(contexts.dtype)
        return self.destination_query(torch.cat((contexts, source_codes), dim=-1))[:, :point_count]

    def _destination_log_probs(
        self,
        features: PolicyFeatures,
        states: Sequence[PolicyState],
        batch_indices: Tensor,
        sources: Tensor,
        temperature: float,
    ) -> Tensor:
        contexts = features.context[batch_indices]
        logits = self._destination_logits(contexts, sources, features.point_mask.shape[1]) / temperature
        masks = torch.zeros(logits.shape, dtype=torch.bool, device="cpu")
        for row, (batch_index, source) in enumerate(
            zip(batch_indices.tolist(), sources.tolist(), strict=True)
        ):
            targets = [
                target
                for action_source, target in states[batch_index].legal_actions
                if action_source == source
            ]
            if not targets:
                raise ValueError(f"sampled source {source} has no legal target")
            masks[row, targets] = True
        masks = masks.to(self.device, non_blocking=True)
        logits = logits.masked_fill(~masks, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def _all_legal_log_probs(self, features: PolicyFeatures, states: Sequence[PolicyState],
                             temperature: float = 1.0) -> tuple[Tensor, Tensor, list[int]]:
        """Evaluate all legal moves as one flat tensor with no device-to-host queries."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        lengths = [len(state.legal_actions) for state in states]
        if not all(lengths):
            raise ValueError("a non-terminal policy state needs a legal action")
        points = features.point_mask.shape[1]
        pairs = np.concatenate([state.legal_array for state in states])
        owners = np.repeat(np.arange(len(states)), lengths)
        unique, inverse = np.unique(owners * points + pairs[:, 0], return_inverse=True)
        count = len(pairs)
        pin = supports_pinned_memory(self.device)
        packed = torch.empty(count * 4 + len(unique), dtype=torch.long, pin_memory=pin)
        packed.numpy()[:] = np.concatenate((owners, pairs[:, 0], pairs[:, 1], inverse, unique))
        owner, source, target, destination_row, unique_key = packed.to(self.device, non_blocking=pin).split(
            (count, count, count, count, len(unique)))
        source_masks = torch.zeros(len(states) * points, dtype=torch.bool, device=self.device)
        source_masks.scatter_(0, unique_key, True)
        source_logs = self._source_log_probs(features, states, temperature,
                                              legal_masks=source_masks.view(len(states), points))
        destination_masks = torch.zeros(len(unique) * points, dtype=torch.bool, device=self.device)
        destination_masks.scatter_(0, destination_row * points + target, True)
        logits = self._destination_logits(features.context.index_select(0, unique_key // points),
                                          unique_key % points, points) / temperature
        destination_logs = F.log_softmax(
            logits.masked_fill(~destination_masks.view(len(unique), points), float("-inf")), dim=-1,
        )
        return source_logs[owner, source] + destination_logs[destination_row, target], owner, lengths

    def ppo_statistics(self, states: Sequence[PolicyState], actions: Sequence[tuple[int, int]],
                       *, pack_sequences: bool = False) -> tuple[Tensor, Tensor]:
        if len(states) != len(actions):
            raise ValueError("PPO actions must match the state batch")
        features = (self._encode_full(states, pack_prefixes=True)
                    if pack_sequences else self.encode(states))
        joint, owners, lengths = self._all_legal_log_probs(features, states)
        offsets = np.cumsum([0, *lengths[:-1]])
        selected = [int(offset) + state.legal_actions.index(action)
                    for offset, state, action in zip(offsets, states, actions, strict=True)]
        current = joint.index_select(0, torch.tensor(selected, dtype=torch.long, device=self.device)).float()
        # FP32 reductions keep exact joint entropy, including every legal move.
        logs = joint.float()
        entropy = torch.zeros(len(states), device=self.device).scatter_add_(0, owners, -logs.exp() * logs)
        return current, entropy

    def log_probs_for_action_groups(
        self,
        states: Sequence[PolicyState],
        actions_by_state: Sequence[Sequence[tuple[int, int]]],
        *,
        temperature: float = 1.0,
        pack_sequences: bool = False,
    ) -> list[Tensor]:
        if len(states) != len(actions_by_state):
            raise ValueError("states and action groups must have the same length")
        features = (self._encode_full(states, pack_prefixes=True)
                    if pack_sequences else self.encode(states))
        source_log_probs = self._source_log_probs(features, states, temperature)
        flat_batch: list[int] = []
        flat_sources: list[int] = []
        flat_targets: list[int] = []
        source_groups: dict[tuple[int, int], int] = {}
        unique_batch: list[int] = []
        unique_sources: list[int] = []
        destination_rows: list[int] = []
        lengths: list[int] = []
        for batch_index, actions in enumerate(actions_by_state):
            lengths.append(len(actions))
            for source, target in actions:
                if (source, target) not in states[batch_index].legal_actions:
                    raise ValueError(f"action {(source, target)} is not legal")
                flat_batch.append(batch_index)
                flat_sources.append(source)
                flat_targets.append(target)
                key = (batch_index, source)
                group = source_groups.get(key)
                if group is None:
                    group = len(unique_batch)
                    source_groups[key] = group
                    unique_batch.append(batch_index)
                    unique_sources.append(source)
                destination_rows.append(group)
        if not flat_batch:
            return [torch.empty(0, device=self.device) for _ in states]
        batch_indices = torch.tensor(flat_batch, dtype=torch.long, device=self.device)
        sources = torch.tensor(flat_sources, dtype=torch.long, device=self.device)
        targets = torch.tensor(flat_targets, dtype=torch.long, device=self.device)
        # All destinations of one source share its conditional distribution.
        # Evaluate it once, avoiding one duplicated destination evaluation per legal move.
        destination_log_probs = self._destination_log_probs(
            features, states,
            torch.tensor(unique_batch, dtype=torch.long, device=self.device),
            torch.tensor(unique_sources, dtype=torch.long, device=self.device), temperature
        )
        joint = source_log_probs[batch_indices, sources] + destination_log_probs[
            torch.tensor(destination_rows, dtype=torch.long, device=self.device), targets
        ]
        return list(joint.split(lengths))

    @torch.no_grad()
    def sample_action_groups(
        self,
        states: Sequence[PolicyState],
        *,
        count: int = 1,
        temperature: float = 1.0,
        return_log_probs: bool = True,
        sampling_uniforms: Tensor | None = None,
    ) -> tuple[list[list[tuple[int, int]]], list[Tensor]]:
        """Joint-policy samples and logs, optionally using per-state RNG draws.

        ``sampling_uniforms[batch, sample]`` supplies source and destination
        uniforms, making game randomness independent of actor batching order.
        Without supplied uniforms, the optional PPO sampling graph uses inverse
        CDF sampling; other paths use multinomial. Both sample the joint policy.
        """

        if count <= 0:
            raise ValueError("sample count must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        uniform_values_valid = None
        if sampling_uniforms is not None:
            _validate_sampling_uniforms(
                sampling_uniforms, state_count=len(states), sample_count=count
            )
            if sampling_uniforms.device.type != "cpu":
                uniform_values_valid = (
                    torch.isfinite(sampling_uniforms)
                    & (sampling_uniforms >= 0)
                    & (sampling_uniforms < 1)
                ).all().to(self.device)
            sampling_uniforms = sampling_uniforms.to(dtype=torch.float32).to(self.device)
        features = self.encode(states)
        source_masks, destination_masks = self._sampling_legal_masks(
            states, features.point_mask.shape[1]
        )
        if (self.config.ppo_sampling_graphs and self.config.ppo_cuda_graphs and count == 1
                and sampling_uniforms is None and self._fixed_kv_store is not None
                and self.device.type == 'cuda' and not self.training):
            from .ppo_sampling import sample_graph
            uniforms = torch.rand((len(states), 2), device=self.device)
            sampled = sample_graph(self, features.context, source_masks, destination_masks,
                                    uniforms, temperature).cpu()
            pairs = sampled[:, :2].to(torch.long).tolist()
            return ([[tuple(pair)] for pair in pairs],
                    [sampled[i, 2:3] for i in range(len(states))] if return_log_probs else [])
        source_log_probs = self._source_log_probs(
            features,
            states,
            temperature,
            legal_masks=source_masks,
        )
        source_probabilities = source_log_probs.exp()
        sampled_sources = (
            torch.multinomial(source_probabilities, count, replacement=True)
            if sampling_uniforms is None
            else _categorical_from_uniforms(
                source_probabilities, sampling_uniforms[..., 0]
            )
        )
        batch_indices = torch.arange(len(states), device=self.device).repeat_interleave(
            count
        )
        flat_sources = sampled_sources.reshape(-1)
        contexts = features.context[batch_indices]
        destination_logits = self._destination_logits(
            contexts, flat_sources, features.point_mask.shape[1]
        ) / temperature
        selected_destination_masks = destination_masks[
            batch_indices, flat_sources
        ]
        destination_log_probs = F.log_softmax(
            destination_logits.masked_fill(
                ~selected_destination_masks, float("-inf")
            ),
            dim=-1,
        )
        sampled_targets = (
            torch.multinomial(destination_log_probs.exp(), 1, replacement=True)
            if sampling_uniforms is None
            else _categorical_from_uniforms(
                destination_log_probs.exp(), sampling_uniforms[..., 1].reshape(-1, 1)
            )
        ).squeeze(-1)
        joint = None
        if return_log_probs:
            joint = source_log_probs[batch_indices, flat_sources] + destination_log_probs[
                torch.arange(len(flat_sources), device=self.device), sampled_targets
            ]
        # This is the only mandatory device-to-host synchronization in a
        # sampling step: the Python rules engine needs the chosen action.
        sampled_pairs = torch.stack((flat_sources, sampled_targets), dim=-1)
        if uniform_values_valid is not None:
            # Include validation with the required action transfer: invalid
            # device inputs raise normally without an extra synchronization or
            # an asynchronous assertion that could poison the CUDA context.
            sampled_pairs = torch.cat((
                sampled_pairs,
                uniform_values_valid.to(torch.long).reshape(1, 1).expand(1, 2),
            ))
        sampled_pairs = sampled_pairs.cpu()
        if uniform_values_valid is not None:
            if not bool(sampled_pairs[-1, 0]):
                raise ValueError(
                    "sampling_uniforms must contain finite values in [0, 1)"
                )
            sampled_pairs = sampled_pairs[:-1]
        joint_cpu = None if joint is None else joint.detach().cpu()
        actions: list[list[tuple[int, int]]] = []
        log_probs: list[Tensor] = []
        pair_rows = sampled_pairs.tolist()
        for index in range(len(states)):
            start = index * count
            stop = start + count
            actions.append(
                [tuple(pair) for pair in pair_rows[start:stop]]
            )
            if joint_cpu is not None:
                log_probs.append(joint_cpu[start:stop])
        return actions, log_probs

    def distributions(
        self,
        states: Sequence[PolicyState],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Exact joint log-probabilities for all legal actions in each state."""

        groups = [state.legal_actions for state in states]
        logs = self.log_probs_for_action_groups(states, groups)
        return logs, [item.exp() for item in logs]


class GameValueTransformer(GamePolicyTransformer):
    """Independent policy-sized critic with one scalar per player-view history.

    The board and causal history encoders are identical to the policy's.  Only
    the two action queries are replaced; no policy parameters or caches are
    shared.  Values describe the observing player's team's expected return.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__(config)
        del self.source_query
        del self.destination_query
        self.value_head = nn.Linear(self.config.temporal_dim, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)

    def initialize_from_policy(self, policy: GamePolicyTransformer) -> None:
        if policy.config != self.config:
            raise ValueError("critic and policy configurations must match")
        backbone = {
            key: value for key, value in policy.state_dict().items()
            if not key.startswith(("source_query.", "destination_query."))
        }
        missing, unexpected = self.load_state_dict(backbone, strict=False)
        if set(missing) != {"value_head.weight", "value_head.bias"} or unexpected:
            raise RuntimeError("policy and critic backbones do not match")

    def forward(self, states: Sequence[PolicyState], *, pack_sequences: bool = False) -> Tensor:
        features = (self._encode_full(states, pack_prefixes=True)
                    if pack_sequences else self.encode(states))
        return self.value_head(features.context).squeeze(-1).float()


def _layout_rule_mask() -> Tensor:
    mask = torch.zeros((len(PieceType), len(DEPLOYMENT_POINT_ORDER)), dtype=torch.bool)
    for kind in PieceType:
        for point_index, coordinate in enumerate(DEPLOYMENT_POINT_ORDER):
            row, _column = coordinate
            legal = True
            if kind is PieceType.FLAG:
                legal = coordinate in HEADQUARTERS_COORDINATES
            elif kind is PieceType.MINE:
                legal = row in (5, 6)
            elif kind is PieceType.BOMB:
                legal = row != 1
            mask[PIECE_TYPE_INDICES[kind], point_index] = legal
    return mask


@dataclass(frozen=True, slots=True)
class LayoutSample:
    setup: PlayerSetup
    position_indices: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    mode: TrainingMode


def layout_sample_from_trace(
    mode: TrainingMode | str,
    position_indices: Sequence[int],
    old_log_probs: Sequence[float],
) -> LayoutSample:
    if len(position_indices) != 25 or len(old_log_probs) != 25:
        raise ValueError("a layout trace must contain exactly 25 decisions")
    normalized = normalize_mode(mode)
    placements = {
        DEPLOYMENT_POINT_ORDER[int(position)]: piece
        for piece, position in zip(
            DEPLOYMENT_PIECE_SEQUENCE, position_indices, strict=True
        )
    }
    return LayoutSample(
        setup=PlayerSetup(placements),
        position_indices=tuple(int(value) for value in position_indices),
        old_log_probs=tuple(float(value) for value in old_log_probs),
        mode=normalized,
    )


class PieceConditionedLayoutPointerDecoder(nn.Module):
    """Autoregressively place a fixed inventory by pointing to empty positions."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        dim = self.config.layout_dim
        self.point_embedding = nn.Embedding(25, dim)
        self.occupant_embedding = nn.Embedding(len(PieceType) + 1, dim)
        self.row_embedding = nn.Embedding(7, dim)
        self.point_type_embedding = nn.Embedding(2, dim)
        self.mode_embedding = nn.Embedding(3, dim)
        self.piece_embedding = nn.Embedding(len(PieceType), dim)
        self.step_embedding = nn.Embedding(25, dim)
        self.layout_token = nn.Parameter(torch.empty(dim))
        self.layers = nn.ModuleList(
            PreNormEncoderBlock(
                dim,
                self.config.layout_heads,
                self.config.layout_ffn_dim,
                self.config.dropout,
            )
            for _ in range(self.config.layout_layers)
        )
        self.final_norm = nn.LayerNorm(dim)
        self.query = nn.Sequential(
            nn.Linear(4 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.query_projection = nn.Linear(dim, dim, bias=False)
        self.key_projection = nn.Linear(dim, dim, bias=False)
        rows = torch.tensor([row for row, _ in DEPLOYMENT_POINT_ORDER])
        point_types = torch.tensor(
            [
                int(coordinate in HEADQUARTERS_COORDINATES)
                for coordinate in DEPLOYMENT_POINT_ORDER
            ]
        )
        sequence = torch.tensor(
            [PIECE_TYPE_INDICES[kind] for kind in DEPLOYMENT_PIECE_SEQUENCE]
        )
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("point_types", point_types, persistent=False)
        self.register_buffer("piece_sequence", sequence, persistent=False)
        self.register_buffer("piece_rule_mask", _layout_rule_mask(), persistent=False)
        nn.init.normal_(self.layout_token, std=0.02)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def logits(self, occupants: Tensor, steps: Tensor, mode_ids: Tensor) -> Tensor:
        batch = occupants.shape[0]
        if occupants.shape != (batch, 25):
            raise ValueError("occupants must have shape [batch, 25]")
        if steps.shape != (batch,) or mode_ids.shape != (batch,):
            raise ValueError("steps and mode_ids must have shape [batch]")
        positions = torch.arange(25, device=self.device)
        tokens = (
            self.point_embedding(positions).unsqueeze(0)
            + self.occupant_embedding(occupants)
            + self.row_embedding(self.rows).unsqueeze(0)
            + self.point_type_embedding(self.point_types).unsqueeze(0)
            + self.mode_embedding(mode_ids).unsqueeze(1)
        )
        pool = (
            self.layout_token.view(1, 1, -1)
            + self.mode_embedding(mode_ids).unsqueeze(1)
        )
        tokens = torch.cat((pool, tokens), dim=1)
        valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=self.device)
        for layer in self.layers:
            if self.training and self.config.activation_checkpointing:
                tokens = checkpoint(
                    lambda values, block=layer: block(
                        values, valid_mask=valid
                    ),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(tokens, valid_mask=valid)
        tokens = self.final_norm(tokens)
        global_state, point_states = tokens[:, 0], tokens[:, 1:]
        piece_ids = self.piece_sequence[steps]
        query = self.query(
            torch.cat(
                (
                    self.piece_embedding(piece_ids),
                    self.step_embedding(steps),
                    self.mode_embedding(mode_ids),
                    global_state,
                ),
                dim=-1,
            )
        )
        scores = torch.einsum(
            "bd,bnd->bn",
            self.query_projection(query),
            self.key_projection(point_states),
        ) / math.sqrt(self.config.layout_dim)
        legal = (occupants == 0) & self.piece_rule_mask[piece_ids]
        if not torch.all(legal.any(dim=-1)):
            raise RuntimeError("layout hard mask has no legal position")
        return scores.masked_fill(~legal, float("-inf"))

    @torch.no_grad()
    def sample_layouts(
        self,
        count: int,
        mode: TrainingMode | str,
        *,
        temperature: float = 0.7,
    ) -> list[LayoutSample]:
        if count <= 0 or temperature <= 0:
            raise ValueError("count and temperature must be positive")
        normalized = normalize_mode(mode)
        mode_id = mode_spec(normalized).mode_index
        occupants = torch.zeros((count, 25), dtype=torch.long, device=self.device)
        choices: list[Tensor] = []
        old_logs: list[Tensor] = []
        mode_ids = torch.full((count,), mode_id, dtype=torch.long, device=self.device)
        rows = torch.arange(count, device=self.device)
        for step in range(25):
            steps = torch.full((count,), step, dtype=torch.long, device=self.device)
            log_probs = F.log_softmax(
                self.logits(occupants, steps, mode_ids) / temperature, dim=-1
            )
            selected = torch.multinomial(log_probs.exp(), 1).squeeze(-1)
            piece_id = PIECE_TYPE_INDICES[DEPLOYMENT_PIECE_SEQUENCE[step]]
            occupants[rows, selected] = piece_id + 1
            choices.append(selected)
            old_logs.append(log_probs[rows, selected])

        choice_tensor = torch.stack(choices, dim=1).cpu()
        log_tensor = torch.stack(old_logs, dim=1).cpu()
        samples: list[LayoutSample] = []
        for sample_index in range(count):
            samples.append(
                layout_sample_from_trace(
                    normalized,
                    choice_tensor[sample_index].tolist(),
                    log_tensor[sample_index].tolist(),
                )
            )
        return samples

    def evaluate_layouts(
        self,
        position_indices: Tensor,
        mode_ids: Tensor,
        *,
        temperature: float = 0.7,
    ) -> tuple[Tensor, Tensor]:
        """Replay position choices and return per-step log-probability and entropy."""

        if position_indices.ndim != 2 or position_indices.shape[1] != 25:
            raise ValueError("position_indices must have shape [batch, 25]")
        batch = position_indices.shape[0]
        occupants = torch.zeros((batch, 25), dtype=torch.long, device=self.device)
        rows = torch.arange(batch, device=self.device)
        selected_logs: list[Tensor] = []
        entropies: list[Tensor] = []
        for step in range(25):
            steps = torch.full((batch,), step, dtype=torch.long, device=self.device)
            log_probs = F.log_softmax(
                self.logits(occupants, steps, mode_ids) / temperature, dim=-1
            )
            probabilities = log_probs.exp()
            selected = position_indices[:, step]
            if not torch.isfinite(log_probs[rows, selected]).all():
                raise ValueError("layout replay contains an illegal position")
            selected_logs.append(log_probs[rows, selected])
            entropies.append(-(probabilities * log_probs.nan_to_num()).sum(dim=-1))
            piece_id = PIECE_TYPE_INDICES[DEPLOYMENT_PIECE_SEQUENCE[step]]
            occupants = occupants.clone()
            occupants[rows, selected] = piece_id + 1
        return torch.stack(selected_logs, dim=1), torch.stack(entropies, dim=1)


class ModelBundle(nn.Module):
    """Registered pair used for checkpointing and device movement."""

    def __init__(
        self,
        policy: GamePolicyTransformer,
        layout: PieceConditionedLayoutPointerDecoder,
    ) -> None:
        super().__init__()
        self.policy = policy
        self.layout = layout


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())
