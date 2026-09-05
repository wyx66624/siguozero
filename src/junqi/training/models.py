"""Transformer models for layout generation and move policy inference."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from ..board import (
    ArmPoint,
    FourPlayerBoard,
    HEADQUARTERS_COORDINATES,
    PathKind,
    PointKind,
    TwoPlayerBoard,
)
from ..pieces import (
    DEPLOYMENT_PIECE_SEQUENCE,
    PIECE_TYPE_INDICES,
    SETUP_COORDINATES,
    PieceType,
    PlayerSetup,
)
from .encoding import (
    ACTION_COMBAT_PAD,
    ACTION_PLAYER_PAD,
    ACTION_POINT_PAD,
    BOARD_CODE_VOCAB_SIZE,
    BOARD_PAD_CODE,
    CASUALTY_SLOTS_PER_PLAYER,
    MAX_CASUALTY_BITS,
    PolicyState,
    StateTokenRecord,
)
from .modes import MODE_SPECS, TrainingMode, mode_spec, normalize_mode


DEPLOYMENT_POINT_ORDER: tuple[tuple[int, int], ...] = tuple(
    sorted(SETUP_COORDINATES)
)
@dataclass(frozen=True, slots=True)
class ModelConfig:
    board_dim: int = 256
    board_layers: int = 4
    board_heads: int = 8
    board_ffn_dim: int = 1024
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
    activation_checkpointing: bool = True
    dead_rules_enabled: bool = True

    def __post_init__(self) -> None:
        if self.temporal_dim != 2 * self.board_dim:
            raise ValueError("temporal_dim must equal action_dim + board_dim")
        for dim, heads, label in (
            (self.board_dim, self.board_heads, "board"),
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
        if not isinstance(self.dead_rules_enabled, bool):
            raise ValueError("dead_rules_enabled must be a boolean")

    @property
    def max_sequence_tokens(self) -> int:
        return self.max_transitions + 1

    @classmethod
    def tiny(cls, *, dead_rules_enabled: bool = True) -> ModelConfig:
        """Small architecture used only by unit and CUDA smoke tests."""

        return cls(
            board_dim=32,
            board_layers=1,
            board_heads=4,
            board_ffn_dim=64,
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
        """Approximately 162M-parameter main-training pair from the design."""

        return cls(
            board_dim=256,
            board_layers=8,
            board_heads=8,
            board_ffn_dim=1024,
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
        """Approximately 243M-parameter capacity-ablation pair."""

        return cls(
            board_dim=256,
            board_layers=12,
            board_heads=8,
            board_ffn_dim=1024,
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
    ) -> Tensor:
        normalized = self.attention_norm(inputs)
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
        normalized = self.attention_norm(inputs)
        projection = F.linear(
            normalized,
            self.attention.in_proj_weight,
            self.attention.in_proj_bias,
        )
        queries, new_keys, new_values = projection.chunk(3, dim=-1)
        batch, _one, width = queries.shape
        heads = self.attention.num_heads
        head_dim = width // heads

        def split_heads(item: Tensor) -> Tensor:
            return item.view(batch, 1, heads, head_dim).transpose(1, 2)

        queries = split_heads(queries)
        new_keys = split_heads(new_keys)
        new_values = split_heads(new_values)
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
        attended = F.linear(
            attended,
            self.attention.out_proj.weight,
            self.attention.out_proj.bias,
        )
        outputs = inputs + self.dropout(attended)
        outputs = outputs + self.dropout(self.ffn(self.ffn_norm(outputs)))
        return outputs, keys, values


def _static_board_features() -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    point_kind = torch.zeros((3, 129), dtype=torch.long)
    row_index = torch.zeros((3, 129), dtype=torch.long)
    road_degree = torch.zeros((3, 129), dtype=torch.long)
    rail_degree = torch.zeros((3, 129), dtype=torch.long)
    valid = torch.zeros((3, 129), dtype=torch.bool)
    kind_indices = {
        PointKind.STATION: 0,
        PointKind.CAMP: 1,
        PointKind.HEADQUARTERS: 2,
        PointKind.CENTRAL_STATION: 3,
    }

    boards = {
        TrainingMode.FOUR_DARK: FourPlayerBoard(),
        TrainingMode.DOUBLE_OPEN: FourPlayerBoard(),
        TrainingMode.TWO_PLAYER: TwoPlayerBoard(),
    }
    for mode, board in boards.items():
        mode_index = mode_spec(mode).mode_index
        valid[mode_index, : board.point_count] = True
        for code in range(board.point_count):
            record = board.point(code)
            point_kind[mode_index, code] = kind_indices[record.kind]
            physical = record.physical
            row_index[mode_index, code] = (
                physical.row if isinstance(physical, ArmPoint) else 0
            )
            road_degree[mode_index, code] = min(
                len(board.neighbors(code, PathKind.ROAD)), 8
            )
            rail_degree[mode_index, code] = min(
                len(board.neighbors(code, PathKind.RAILWAY)), 8
            )
    return point_kind, row_index, road_degree, rail_degree, valid


def _graph_relation_tables() -> Tensor:
    """Return graph-distance/edge-type categories including a pooling token."""

    tables = torch.full((3, 130, 130), 7, dtype=torch.long)
    boards = {
        TrainingMode.FOUR_DARK: FourPlayerBoard(),
        TrainingMode.DOUBLE_OPEN: FourPlayerBoard(),
        TrainingMode.TWO_PLAYER: TwoPlayerBoard(),
    }
    for mode, board in boards.items():
        mode_index = mode_spec(mode).mode_index
        size = board.point_count
        tables[mode_index, 0, : size + 1] = 0
        tables[mode_index, : size + 1, 0] = 0
        for source in range(size):
            distances = [-1] * size
            distances[source] = 0
            queue = [source]
            for current in queue:
                if distances[current] >= 3:
                    continue
                for target in board.neighbors(current):
                    if distances[target] == -1:
                        distances[target] = distances[current] + 1
                        queue.append(target)
            for target, distance in enumerate(distances):
                if source == target:
                    relation = 1
                elif board.path_kind(source, target) is PathKind.ROAD:
                    relation = 2
                elif board.path_kind(source, target) is PathKind.RAILWAY:
                    relation = 3
                elif distance == 2:
                    relation = 4
                elif distance == 3:
                    relation = 5
                else:
                    relation = 6
                tables[mode_index, source + 1, target + 1] = relation
    return tables


class GraphBoardEncoder(nn.Module):
    """Encode categorical board chains with explicit road/rail graph bias."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        dim = config.board_dim
        self.config = config
        self.code_embedding = nn.Embedding(
            BOARD_CODE_VOCAB_SIZE, dim, padding_idx=BOARD_PAD_CODE
        )
        self.position_embedding = nn.Embedding(129, dim)
        self.point_kind_embedding = nn.Embedding(4, dim)
        self.row_embedding = nn.Embedding(7, dim)
        self.road_degree_embedding = nn.Embedding(9, dim)
        self.rail_degree_embedding = nn.Embedding(9, dim)
        self.mode_embedding = nn.Embedding(3, dim)
        self.board_token = nn.Parameter(torch.empty(dim))
        if config.dead_rules_enabled:
            self.casualty_projection: nn.Linear | None = nn.Linear(
                MAX_CASUALTY_BITS, dim, bias=False
            )
            self.board_casualty_fusion: nn.Sequential | None = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.SiLU(),
                nn.LayerNorm(dim),
            )
        else:
            # The off variant contains no dormant 75-D input or unused
            # parameters, making it a genuine architecture-level ablation.
            self.casualty_projection = None
            self.board_casualty_fusion = None
        self.relation_bias = nn.Embedding(8, 1)
        self.layers = nn.ModuleList(
            PreNormEncoderBlock(
                dim,
                config.board_heads,
                config.board_ffn_dim,
                config.dropout,
            )
            for _ in range(config.board_layers)
        )
        self.final_norm = nn.LayerNorm(dim)

        features = _static_board_features()
        self.register_buffer("point_kind", features[0], persistent=False)
        self.register_buffer("row_index", features[1], persistent=False)
        self.register_buffer("road_degree", features[2], persistent=False)
        self.register_buffer("rail_degree", features[3], persistent=False)
        self.register_buffer("static_valid", features[4], persistent=False)
        self.register_buffer(
            "relation_tables", _graph_relation_tables(), persistent=False
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.board_token, std=0.02)
        nn.init.zeros_(self.relation_bias.weight)

    def forward(
        self,
        board_codes: Tensor,
        point_mask: Tensor,
        mode_ids: Tensor,
        casualty_bits: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if board_codes.ndim != 2:
            raise ValueError("board_codes must have shape [batch, points]")
        batch, points = board_codes.shape
        if points > 129 or point_mask.shape != board_codes.shape:
            raise ValueError("invalid board point tensor shape")
        if mode_ids.shape != (batch,):
            raise ValueError("mode_ids must have shape [batch]")
        if self.config.dead_rules_enabled:
            if casualty_bits is None or casualty_bits.shape != (
                batch,
                MAX_CASUALTY_BITS,
            ):
                raise ValueError(
                    f"casualty_bits must have shape [batch, {MAX_CASUALTY_BITS}]"
                )
        elif casualty_bits is not None:
            raise ValueError("casualty_bits must be absent when dead rules are disabled")
        if batch == 0:
            raise ValueError("cannot encode an empty board batch")
        positions = torch.arange(points, device=board_codes.device)
        static_kind = self.point_kind[mode_ids, :points]
        rows = self.row_index[mode_ids, :points]
        road = self.road_degree[mode_ids, :points]
        rail = self.rail_degree[mode_ids, :points]
        tokens = (
            self.code_embedding(board_codes)
            + self.position_embedding(positions).unsqueeze(0)
            + self.point_kind_embedding(static_kind)
            + self.row_embedding(rows)
            + self.road_degree_embedding(road)
            + self.rail_degree_embedding(rail)
            + self.mode_embedding(mode_ids).unsqueeze(1)
        )
        pool = self.board_token.view(1, 1, -1).expand(batch, 1, -1)
        tokens = torch.cat((pool, tokens), dim=1)
        valid = torch.cat(
            (
                torch.ones((batch, 1), dtype=torch.bool, device=tokens.device),
                point_mask,
            ),
            dim=1,
        )
        relation_ids = self.relation_tables[mode_ids[0], : points + 1, : points + 1]
        attention_bias = self.relation_bias(relation_ids).squeeze(-1)
        attention_bias = attention_bias.to(tokens.dtype)
        for layer in self.layers:
            if self.training and self.config.activation_checkpointing:
                tokens = checkpoint(
                    lambda values, block=layer: block(
                        values,
                        valid_mask=valid,
                        attention_mask=attention_bias,
                    ),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(
                    tokens,
                    valid_mask=valid,
                    attention_mask=attention_bias,
                )
        tokens = self.final_norm(tokens)
        board_global = tokens[:, 0]
        if self.config.dead_rules_enabled:
            if self.casualty_projection is None or self.board_casualty_fusion is None:
                raise RuntimeError("dead-rule feature modules were not initialized")
            assert casualty_bits is not None
            casualty_embedding = self.casualty_projection(
                casualty_bits.to(tokens.dtype)
            )
            board_global = self.board_casualty_fusion(
                torch.cat((board_global, casualty_embedding), dim=-1)
            )
        return board_global, tokens[:, 1:]


class PublicActionEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        field_dim = max(8, output_dim // 4)
        self.source = nn.Embedding(130, field_dim, padding_idx=ACTION_POINT_PAD)
        self.destination = nn.Embedding(130, field_dim, padding_idx=ACTION_POINT_PAD)
        self.actor = nn.Embedding(5, field_dim, padding_idx=ACTION_PLAYER_PAD)
        self.combat = nn.Embedding(5, field_dim, padding_idx=ACTION_COMBAT_PAD)
        self.was_attack = nn.Embedding(3, field_dim, padding_idx=2)
        self.flag_owner = nn.Embedding(5, field_dim, padding_idx=ACTION_PLAYER_PAD)
        self.reveal_count = nn.Embedding(5, field_dim)
        self.eliminated_count = nn.Embedding(5, field_dim)
        self.projection = nn.Linear(8 * field_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, fields: Tensor, present: Tensor) -> Tensor:
        if fields.shape[-1] != 8:
            raise ValueError("action field tensor must end with 8 values")
        embeddings = torch.cat(
            (
                self.source(fields[..., 0]),
                self.destination(fields[..., 1]),
                self.actor(fields[..., 2]),
                self.combat(fields[..., 3]),
                self.was_attack(fields[..., 4]),
                self.flag_owner(fields[..., 5]),
                self.reveal_count(fields[..., 6]),
                self.eliminated_count(fields[..., 7]),
            ),
            dim=-1,
        )
        encoded = self.norm(self.projection(embeddings))
        return encoded * present.unsqueeze(-1).to(encoded.dtype)


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


def collate_policy_states(
    states: Sequence[PolicyState],
    *,
    device: torch.device | str,
    dead_rules_enabled: bool = True,
) -> PolicyTensorBatch:
    if not states:
        raise ValueError("cannot collate an empty policy batch")
    modes = {state.mode for state in states}
    if len(modes) != 1:
        raise ValueError("a policy batch must use one information mode")
    max_tokens = max(len(state.records) for state in states)
    max_points = max(len(record.board_codes) for state in states for record in state.records)
    batch = len(states)
    packed_tokens = sum(len(state.records) for state in states)
    requested_device = torch.device(device)
    pin_memory = requested_device.type == "cuda"
    board_codes = torch.full(
        (packed_tokens, max_points),
        BOARD_PAD_CODE,
        dtype=torch.long,
        pin_memory=pin_memory,
    )
    casualty_bits = (
        torch.zeros(
            (packed_tokens, MAX_CASUALTY_BITS),
            dtype=torch.float32,
            pin_memory=pin_memory,
        )
        if dead_rules_enabled
        else None
    )
    point_mask = torch.zeros(
        (batch, max_points), dtype=torch.bool, pin_memory=pin_memory
    )
    token_owner = torch.empty(
        (packed_tokens,), dtype=torch.long, pin_memory=pin_memory
    )
    token_mask = torch.zeros(
        (batch, max_tokens), dtype=torch.bool, pin_memory=pin_memory
    )
    action_fields = torch.zeros(
        (batch, max_tokens, 8), dtype=torch.long, pin_memory=pin_memory
    )
    action_present = torch.zeros(
        (batch, max_tokens), dtype=torch.bool, pin_memory=pin_memory
    )
    no_interaction = torch.zeros(
        (batch, max_tokens), dtype=torch.long, pin_memory=pin_memory
    )
    active_mask = torch.zeros(
        (batch, max_tokens), dtype=torch.long, pin_memory=pin_memory
    )
    revealed_mask = torch.zeros(
        (batch, max_tokens), dtype=torch.long, pin_memory=pin_memory
    )
    current_player = torch.full(
        (batch, max_tokens),
        ACTION_PLAYER_PAD,
        dtype=torch.long,
        pin_memory=pin_memory,
    )
    mode_ids = torch.empty((batch,), dtype=torch.long, pin_memory=pin_memory)

    packed_index = 0
    for batch_index, state in enumerate(states):
        spec = mode_spec(state.mode)
        mode_ids[batch_index] = spec.mode_index
        point_mask[batch_index, : spec.point_count] = True
        for time_index, record in enumerate(state.records):
            if len(record.board_codes) != spec.point_count:
                raise ValueError("history board length does not match its mode")
            token_mask[batch_index, time_index] = True
            token_owner[packed_index] = batch_index
            board_codes[packed_index, : spec.point_count] = torch.as_tensor(
                record.board_codes, dtype=torch.long
            )
            if dead_rules_enabled:
                if record.known_casualty_bits is None:
                    raise ValueError("dead-rule model requires casualty history")
                expected_casualty_bits = {
                    TrainingMode.FOUR_DARK: 3 * CASUALTY_SLOTS_PER_PLAYER,
                    TrainingMode.DOUBLE_OPEN: 2 * CASUALTY_SLOTS_PER_PLAYER,
                    TrainingMode.TWO_PLAYER: CASUALTY_SLOTS_PER_PLAYER,
                }[state.mode]
                if len(record.known_casualty_bits) != expected_casualty_bits:
                    raise ValueError("history casualty length does not match its mode")
                if any(value not in (0, 1) for value in record.known_casualty_bits):
                    raise ValueError("history casualty values must be binary")
                source_bits = torch.as_tensor(
                    record.known_casualty_bits, dtype=torch.float32
                )
                assert casualty_bits is not None
                if state.mode is TrainingMode.FOUR_DARK:
                    casualty_bits[packed_index] = source_bits
                elif state.mode is TrainingMode.DOUBLE_OPEN:
                    casualty_bits[
                        packed_index,
                        :CASUALTY_SLOTS_PER_PLAYER,
                    ] = source_bits[:CASUALTY_SLOTS_PER_PLAYER]
                    casualty_bits[
                        packed_index,
                        2 * CASUALTY_SLOTS_PER_PLAYER :,
                    ] = source_bits[CASUALTY_SLOTS_PER_PLAYER:]
                else:
                    casualty_bits[
                        packed_index,
                        CASUALTY_SLOTS_PER_PLAYER : 2 * CASUALTY_SLOTS_PER_PLAYER,
                    ] = source_bits
            elif record.known_casualty_bits is not None:
                raise ValueError("non-dead-rule model must not receive casualty history")
            no_interaction[batch_index, time_index] = record.no_interaction_plies
            active_mask[batch_index, time_index] = record.active_mask
            revealed_mask[batch_index, time_index] = record.revealed_mask
            current_player[batch_index, time_index] = record.current_player
            if record.action is not None:
                action_present[batch_index, time_index] = True
                action_fields[batch_index, time_index] = torch.as_tensor(
                    record.action.as_tuple(), dtype=torch.long
                )
            packed_index += 1

    def transfer(tensor: Tensor) -> Tensor:
        return tensor.to(requested_device, non_blocking=pin_memory)

    return PolicyTensorBatch(
        states=states,
        board_codes=transfer(board_codes),
        casualty_bits=(None if casualty_bits is None else transfer(casualty_bits)),
        token_owner=transfer(token_owner),
        point_mask=transfer(point_mask),
        token_mask=transfer(token_mask),
        action_fields=transfer(action_fields),
        action_present=transfer(action_present),
        no_interaction=transfer(no_interaction),
        active_mask=transfer(active_mask),
        revealed_mask=transfer(revealed_mask),
        current_player=transfer(current_player),
        mode_ids=transfer(mode_ids),
    )


@dataclass(slots=True)
class PolicyFeatures:
    context: Tensor
    current_points: Tensor
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
        self.board_encoder = GraphBoardEncoder(self.config)
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
            nn.Linear(self.config.board_dim, self.config.board_dim),
        )
        self.destination_query = nn.Sequential(
            nn.Linear(temporal_dim + self.config.board_dim, self.config.board_dim),
            nn.SiLU(),
            nn.Linear(self.config.board_dim, self.config.board_dim),
        )
        # The cache is explicitly enabled only for a frozen rollout actor and
        # cleared before learner mode.  It stores detached board-global tokens,
        # never tensors used for gradient computation.
        self._inference_board_cache: dict[object, Tensor] = {}
        self._inference_board_cache_limit = 0
        self._inference_temporal_cache: OrderedDict[
            object, _TemporalStateCache
        ] = OrderedDict()
        self._inference_temporal_cache_limit = 0
        self.reset_board_encoding_stats()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def start_inference_board_cache(self, max_entries: int | None = None) -> None:
        """Start a fresh per-collection cache for immutable history boards."""

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
        self.reset_board_encoding_stats()

    def clear_inference_board_cache(self) -> None:
        self._inference_board_cache.clear()
        self._inference_board_cache_limit = 0
        self._inference_temporal_cache.clear()
        self._inference_temporal_cache_limit = 0

    def reset_inference_temporal_cache(self) -> None:
        """Drop rollout prefixes between bounded anchor waves."""

        self._inference_temporal_cache.clear()

    def reset_board_encoding_stats(self) -> None:
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

    def board_encoding_metrics(self) -> dict[str, float]:
        raw_tokens = max(self._raw_board_tokens, 1)
        input_tokens = max(self._board_input_tokens, 1)
        unique_tokens = max(self._board_unique_tokens, 1)
        input_states = max(self._history_input_states, 1)
        return {
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
            "encoding/temporal_cache_hits": float(self._temporal_cache_hits),
            "encoding/temporal_cold_states": float(self._temporal_cold_states),
            "encoding/temporal_incremental_tokens": float(
                self._temporal_incremental_tokens
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
        self, key: object, value: _TemporalStateCache
    ) -> None:
        if self._inference_temporal_cache_limit <= 0:
            return
        self._inference_temporal_cache[key] = value
        self._inference_temporal_cache.move_to_end(key)
        while (
            len(self._inference_temporal_cache)
            > self._inference_temporal_cache_limit
        ):
            self._inference_temporal_cache.popitem(last=False)

    def _get_temporal_cache(
        self, key: object
    ) -> _TemporalStateCache | None:
        value = self._inference_temporal_cache.get(key)
        if value is not None:
            self._inference_temporal_cache.move_to_end(key)
        return value

    def _find_temporal_prefix(
        self, state: PolicyState
    ) -> tuple[_TemporalStateCache | None, int]:
        # A player acts every two/four plies.  Eight covers the normal gap and
        # keeps lookup O(1) instead of hashing every possible prefix.
        maximum_gap = min(8, len(state.records) - 1)
        for missing in range(maximum_gap + 1):
            records = (
                state.records
                if missing == 0
                else state.records[:-missing]
            )
            cached = self._get_temporal_cache((state.mode, records))
            if cached is not None:
                return cached, missing
        return None, len(state.records)

    def _record_board_batch(
        self,
        records: Sequence[StateTokenRecord],
        mode: TrainingMode,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode only the newly appended records of an incremental batch."""

        if not records:
            raise ValueError("record batch cannot be empty")
        spec = mode_spec(mode)
        if any(len(record.board_codes) != spec.point_count for record in records):
            raise ValueError("history board length does not match its mode")
        requested_device = self.device
        pin_memory = requested_device.type == "cuda"
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
        board_globals, point_tokens = self.board_encoder(
            codes, point_mask, mode_ids, casualties
        )
        return board_globals, point_tokens, point_mask

    def _record_temporal_tokens(
        self,
        records: Sequence[StateTokenRecord],
        mode: TrainingMode,
        board_globals: Tensor,
        position: int,
    ) -> Tensor:
        if position >= self.config.max_sequence_tokens:
            raise ValueError("incremental history exceeds model position capacity")
        pin_memory = self.device.type == "cuda"
        fields = torch.zeros(
            (len(records), 1, 8), dtype=torch.long, pin_memory=pin_memory
        )
        present = torch.zeros(
            (len(records), 1), dtype=torch.bool, pin_memory=pin_memory
        )
        for index, record in enumerate(records):
            if record.action is not None:
                fields[index, 0] = torch.tensor(
                    record.action.as_tuple(), dtype=torch.long
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
                torch.full(
                    (len(records), 1),
                    position,
                    dtype=torch.long,
                    device=self.device,
                )
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
        prefixes: Sequence[_TemporalStateCache],
        missing: int,
    ) -> PolicyFeatures:
        if not states or len(states) != len(prefixes):
            raise ValueError("incremental state/cache batch mismatch")
        mode = states[0].mode
        if any(state.mode is not mode for state in states):
            raise ValueError("incremental batch must use one mode")
        caches = list(prefixes)
        current_points: Tensor | None = None
        point_mask: Tensor | None = None
        if missing == 0:
            _globals, current_points, point_mask = self._record_board_batch(
                [state.records[-1] for state in states], mode
            )
            return PolicyFeatures(
                torch.stack([cache.context for cache in caches]),
                current_points,
                point_mask,
            )

        for offset in range(missing):
            position = caches[0].length
            if any(cache.length != position for cache in caches):
                raise ValueError("incremental cache lengths must be bucketed")
            records = [
                state.records[len(state.records) - missing + offset]
                for state in states
            ]
            board_globals, current_points, point_mask = self._record_board_batch(
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
                        (mode, state.records), new_cache
                    )
                new_caches.append(new_cache)
            caches = new_caches
            self._temporal_incremental_tokens += len(states)
            self._temporal_computed_pairs += len(states) * (position + 1)
        assert current_points is not None and point_mask is not None
        return PolicyFeatures(contexts, current_points, point_mask)

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
            key = (state.mode, state.records)
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

        prefixes: list[_TemporalStateCache | None] = []
        missing_counts: list[int] = []
        for state in unique_states:
            prefix, missing = self._find_temporal_prefix(state)
            prefixes.append(prefix)
            missing_counts.append(missing)
            if prefix is not None:
                self._temporal_cache_hits += 1

        contexts: list[Tensor | None] = [None] * len(unique_states)
        points: list[Tensor | None] = [None] * len(unique_states)
        masks: list[Tensor | None] = [None] * len(unique_states)
        cold_indices = [
            index for index, prefix in enumerate(prefixes) if prefix is None
        ]
        if cold_indices:
            cold_states = [unique_states[index] for index in cold_indices]
            cold_features = self._encode_full(
                cold_states, count_history_stats=False
            )
            for row, index in enumerate(cold_indices):
                contexts[index] = cold_features.context[row]
                points[index] = cold_features.current_points[row]
                masks[index] = cold_features.point_mask[row]

        buckets: dict[tuple[int, int], list[int]] = {}
        for index, prefix in enumerate(prefixes):
            if prefix is None:
                continue
            key = (prefix.length, missing_counts[index])
            buckets.setdefault(key, []).append(index)
        for (_prefix_length, missing), indices in buckets.items():
            bucket_states = [unique_states[index] for index in indices]
            bucket_prefixes = [prefixes[index] for index in indices]
            features = self._advance_temporal_group(
                bucket_states,
                [item for item in bucket_prefixes if item is not None],
                missing,
            )
            for row, index in enumerate(indices):
                contexts[index] = features.context[row]
                points[index] = features.current_points[row]
                masks[index] = features.point_mask[row]

        if any(item is None for item in contexts + points + masks):
            raise RuntimeError("incremental cache did not produce every feature row")
        unique_features = PolicyFeatures(
            torch.stack([item for item in contexts if item is not None]),
            torch.stack([item for item in points if item is not None]),
            torch.stack([item for item in masks if item is not None]),
        )
        if len(unique_states) == len(states):
            return unique_features
        expansion = torch.tensor(inverse, dtype=torch.long, device=self.device)
        return PolicyFeatures(
            unique_features.context.index_select(0, expansion),
            unique_features.current_points.index_select(0, expansion),
            unique_features.point_mask.index_select(0, expansion),
        )

    def encode(self, states: Sequence[PolicyState]) -> PolicyFeatures:
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
    ) -> PolicyFeatures:
        if not states:
            raise ValueError("cannot encode an empty policy batch")
        deduplicate = not self.training or self.config.dropout == 0.0
        original_states = states
        state_inverse: list[int] = []
        if deduplicate:
            unique_states: list[PolicyState] = []
            state_to_unique: dict[object, int] = {}
            for state in original_states:
                key = (state.mode, state.records)
                unique_offset = state_to_unique.get(key)
                if unique_offset is None:
                    unique_offset = len(unique_states)
                    state_to_unique[key] = unique_offset
                    unique_states.append(state)
                state_inverse.append(unique_offset)
            states = unique_states
        else:
            state_inverse = list(range(len(states)))
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
        points = batch.board_codes.shape[1]
        flat_token_mask = batch.token_mask.reshape(-1)
        valid_indices = flat_token_mask.nonzero(as_tuple=False).squeeze(-1)
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
        last_unique_offsets: list[int] = []
        key_to_unique: dict[object, int] = {}
        compact_offset = 0
        for state in states:
            for record in state.records:
                key: object
                if deduplicate:
                    key = (
                        state.mode,
                        record.board_codes,
                        record.known_casualty_bits,
                    )
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
            last_unique_offsets.append(inverse_offsets[-1])

        self._board_input_tokens += len(inverse_offsets)
        self._board_unique_tokens += len(unique_keys)
        current_unique = set(last_unique_offsets)
        encode_unique: list[int] = []
        global_rows: list[Tensor | None] = [None] * len(unique_keys)
        current_point_rows: list[Tensor | None] = [None] * batch_size
        if cache_active:
            for unique_offset, key in enumerate(unique_keys):
                cached = self._inference_board_cache.get(key)
                if cached is not None and unique_offset not in current_unique:
                    global_rows[unique_offset] = cached
                    self._board_cache_hits += 1
                else:
                    encode_unique.append(unique_offset)
        else:
            encode_unique = list(range(len(unique_keys)))
        self._board_encoder_tokens += len(encode_unique)

        last_unique_tensor = torch.tensor(
            last_unique_offsets, dtype=torch.long, device=self.device
        )
        for start in range(0, len(encode_unique), self.config.board_chunk_size):
            chunk_unique = encode_unique[
                start : start + self.config.board_chunk_size
            ]
            chunk_sources = torch.tensor(
                [unique_source_offsets[index] for index in chunk_unique],
                dtype=torch.long,
                device=self.device,
            )
            chunk_globals, chunk_points = self.board_encoder(
                flat_codes.index_select(0, chunk_sources),
                flat_point_mask.index_select(0, chunk_sources),
                flat_modes.index_select(0, chunk_sources),
                (
                    None
                    if flat_casualties is None
                    else flat_casualties.index_select(0, chunk_sources)
                ),
            )
            chunk_unique_tensor = torch.tensor(
                chunk_unique, dtype=torch.long, device=self.device
            )
            for local_offset, unique_offset in enumerate(chunk_unique):
                global_rows[unique_offset] = chunk_globals[local_offset]
                if (
                    cache_active
                    and unique_keys[unique_offset]
                    not in self._inference_board_cache
                    and len(self._inference_board_cache)
                    < self._inference_board_cache_limit
                ):
                    self._inference_board_cache[unique_keys[unique_offset]] = (
                        chunk_globals[local_offset].detach().clone()
                    )
            matches = (
                chunk_unique_tensor.unsqueeze(1)
                == last_unique_tensor.unsqueeze(0)
            )
            local_indices, batch_indices = matches.nonzero(as_tuple=True)
            for local_index, batch_index in zip(
                local_indices.tolist(), batch_indices.tolist(), strict=True
            ):
                current_point_rows[batch_index] = chunk_points[local_index]

        if any(row is None for row in global_rows):
            raise RuntimeError("board cache failed to provide a required token")
        if any(row is None for row in current_point_rows):
            raise RuntimeError("current board point embeddings were not encoded")
        unique_globals = torch.stack(
            [row for row in global_rows if row is not None]
        )
        inverse_tensor = torch.tensor(
            inverse_offsets, dtype=torch.long, device=self.device
        )
        valid_globals = unique_globals.index_select(0, inverse_tensor)
        globals_by_token = torch.zeros(
            (batch_size * time_steps, self.config.board_dim),
            dtype=valid_globals.dtype,
            device=self.device,
        ).index_copy(0, valid_indices, valid_globals)
        current_points = torch.stack(
            [row for row in current_point_rows if row is not None]
        )

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
        causal_mask = torch.zeros(
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
        )
        captured_keys: list[Tensor] = []
        captured_values: list[Tensor] = []
        for layer in self.temporal_layers:
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
                    ),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(
                    tokens,
                    valid_mask=batch.token_mask,
                    attention_mask=causal_mask,
                )
        tokens = self.temporal_norm(tokens)
        last_indices = batch.token_mask.sum(dim=1) - 1
        contexts = tokens[torch.arange(batch_size, device=self.device), last_indices]
        features = PolicyFeatures(contexts, current_points, batch.point_mask)
        if capture_temporal:
            for index, state in enumerate(states):
                length = len(state.records)
                self._put_temporal_cache(
                    (state.mode, state.records),
                    _TemporalStateCache(
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
                    ),
                )
                pairs = length * (length + 1) // 2
                if count_history_stats:
                    self._temporal_requested_pairs += pairs
                self._temporal_computed_pairs += pairs
                self._temporal_cold_states += 1
        if len(states) != len(original_states):
            expansion = torch.tensor(
                state_inverse, dtype=torch.long, device=self.device
            )
            features = PolicyFeatures(
                features.context.index_select(0, expansion),
                features.current_points.index_select(0, expansion),
                features.point_mask.index_select(0, expansion),
            )
        return features

    def forward(
        self,
        states: Sequence[PolicyState],
        actions_by_state: Sequence[Sequence[tuple[int, int]]],
        temperature: float = 1.0,
    ) -> list[Tensor]:
        """DDP-compatible entry point for policy likelihood training."""

        return self.log_probs_for_action_groups(
            states, actions_by_state, temperature=temperature
        )

    def _source_log_probs(
        self,
        features: PolicyFeatures,
        states: Sequence[PolicyState],
        temperature: float,
    ) -> Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        query = self.source_query(features.context)
        logits = torch.einsum("bd,bnd->bn", query, features.current_points)
        logits = logits / math.sqrt(self.config.board_dim) / temperature
        legal_masks = torch.zeros(
            logits.shape, dtype=torch.bool, device="cpu"
        )
        for index, state in enumerate(states):
            legal_sources = {source for source, _target in state.legal_actions}
            legal_masks[index, list(legal_sources)] = True
        legal_masks = legal_masks.to(self.device, non_blocking=True)
        logits = logits.masked_fill(~legal_masks, float("-inf"))
        return F.log_softmax(logits, dim=-1)

    def _destination_log_probs(
        self,
        features: PolicyFeatures,
        states: Sequence[PolicyState],
        batch_indices: Tensor,
        sources: Tensor,
        temperature: float,
    ) -> Tensor:
        contexts = features.context[batch_indices]
        source_points = features.current_points[batch_indices, sources]
        query = self.destination_query(torch.cat((contexts, source_points), dim=-1))
        all_points = features.current_points[batch_indices]
        logits = torch.einsum("md,mnd->mn", query, all_points)
        logits = logits / math.sqrt(self.config.board_dim) / temperature
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

    def log_probs_for_action_groups(
        self,
        states: Sequence[PolicyState],
        actions_by_state: Sequence[Sequence[tuple[int, int]]],
        *,
        temperature: float = 1.0,
    ) -> list[Tensor]:
        if len(states) != len(actions_by_state):
            raise ValueError("states and action groups must have the same length")
        features = self.encode(states)
        source_log_probs = self._source_log_probs(features, states, temperature)
        flat_batch: list[int] = []
        flat_sources: list[int] = []
        flat_targets: list[int] = []
        lengths: list[int] = []
        for batch_index, actions in enumerate(actions_by_state):
            lengths.append(len(actions))
            for source, target in actions:
                if (source, target) not in states[batch_index].legal_actions:
                    raise ValueError(f"action {(source, target)} is not legal")
                flat_batch.append(batch_index)
                flat_sources.append(source)
                flat_targets.append(target)
        if not flat_batch:
            return [torch.empty(0, device=self.device) for _ in states]
        batch_indices = torch.tensor(flat_batch, dtype=torch.long, device=self.device)
        sources = torch.tensor(flat_sources, dtype=torch.long, device=self.device)
        targets = torch.tensor(flat_targets, dtype=torch.long, device=self.device)
        destination_log_probs = self._destination_log_probs(
            features, states, batch_indices, sources, temperature
        )
        joint = source_log_probs[batch_indices, sources] + destination_log_probs[
            torch.arange(len(flat_batch), device=self.device), targets
        ]
        return list(joint.split(lengths))

    @torch.no_grad()
    def sample_action_groups(
        self,
        states: Sequence[PolicyState],
        *,
        count: int = 1,
        temperature: float = 1.0,
    ) -> tuple[list[list[tuple[int, int]]], list[Tensor]]:
        """IID joint-policy samples with replacement and their behavior log-probs."""

        if count <= 0:
            raise ValueError("sample count must be positive")
        features = self.encode(states)
        source_log_probs = self._source_log_probs(features, states, temperature)
        source_probabilities = source_log_probs.exp()
        sampled_sources = torch.multinomial(
            source_probabilities, count, replacement=True
        )
        batch_indices = torch.arange(len(states), device=self.device).repeat_interleave(
            count
        )
        flat_sources = sampled_sources.reshape(-1)
        destination_log_probs = self._destination_log_probs(
            features, states, batch_indices, flat_sources, temperature
        )
        sampled_targets = torch.multinomial(
            destination_log_probs.exp(), 1, replacement=True
        ).squeeze(-1)
        joint = source_log_probs[batch_indices, flat_sources] + destination_log_probs[
            torch.arange(len(flat_sources), device=self.device), sampled_targets
        ]
        actions: list[list[tuple[int, int]]] = []
        log_probs: list[Tensor] = []
        for index in range(len(states)):
            start = index * count
            stop = start + count
            actions.append(
                list(
                    zip(
                        flat_sources[start:stop].tolist(),
                        sampled_targets[start:stop].tolist(),
                        strict=True,
                    )
                )
            )
            log_probs.append(joint[start:stop].detach().cpu())
        return actions, log_probs

    def distributions(
        self,
        states: Sequence[PolicyState],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Exact joint log-probabilities for all legal actions in each state."""

        groups = [state.legal_actions for state in states]
        logs = self.log_probs_for_action_groups(states, groups)
        return logs, [item.exp() for item in logs]


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
            piece_id = int(self.piece_sequence[step].item())
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
            piece_id = int(self.piece_sequence[step].item())
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
