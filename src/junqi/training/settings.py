"""Validated YAML settings for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any

import yaml

from .models import ModelConfig
from .modes import TrainingMode, normalize_mode
from .rollout import REPLICAS_PER_CANDIDATE, ROOT_CANDIDATE_COUNT


PLAYER_PARAMETER_SHARING = "single_policy_and_layout_instance_all_seats"
DEAD_RULES_ON_DIRECTORY = "with_dead_rules"
DEAD_RULES_OFF_DIRECTORY = "without_dead_rules"


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    mode: TrainingMode
    model: ModelConfig
    dead_rules_enabled: bool
    source_path: Path
    raw_config: dict[str, Any]
    output_root: Path
    total_updates: int
    target_continuation_plies: int | None
    seed: int
    device: str
    amp: str
    enable_tf32: bool
    enable_torch_compile: bool
    base_game_pool_size: int
    max_game_plies: int | None
    anchor_batch: int
    policy_microbatch: int
    actor_inference_batch: int
    rollout_anchor_wave_size: int
    rollout_environment_workers: int
    learner_length_bucketing: bool
    policy_epochs: int
    policy_learning_rate: float
    minimum_learning_rate: float
    layout_learning_rate: float
    warmup_updates: int
    clip_epsilon: float
    gradient_norm_clip: float
    entropy_coefficient: float
    kl_coefficient: float
    target_kl: float
    early_stop_kl_multiple: float
    early_stop_clip_fraction: float
    advantage_epsilon: float
    layout_update_interval: int
    layout_outcomes_per_update: int
    reference_refresh_updates: int
    checkpoint_every_updates: int
    archive_every_updates: int
    keep_checkpoint_archives: int
    metrics_every_updates: int
    resource_monitor_interval_seconds: float
    auto_reduce_microbatch_on_oom: bool
    minimum_policy_microbatch: int
    # One game.step on any training trajectory, including simulated branches.
    # target_continuation_plies remains a legacy branch-only GRPO option.
    target_environment_plies: int | None = None
    algorithm: str = "grpo"
    discount: float = 1.0
    gae_lambda: float = 0.95
    value_coefficient: float = 0.5
    value_clip_epsilon: float = 0.2
    critic_learning_rate: float = 1.0e-4
    critic_epochs: int = 3
    ppo_sequence_training: bool = True
    ppo_max_samples_per_sequence: int = 64
    ppo_minibatch_samples: int = 512
    grpo_equivalent_plies: int | None = None
    grpo_mean_remaining_plies: float = 334.5
    arena_enabled: bool = True
    arena_start_percent: int = 30
    arena_interval_percent: int = 5
    arena_games: int = 1000
    arena_max_plies: int = 2000
    arena_seed: int = 20260910
    arena_temporal_cache_entries: int = 8
    arena_parallel_games: int = 32
    arena_inference_batch_size: int = 32
    arena_environment_workers: int = 4

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        mode: TrainingMode | str,
        *,
        tiny: bool = False,
        model_scale: str = "bootstrap",
        dead_rules_enabled: bool | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> TrainingSettings:
        source = Path(path).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        normalized = normalize_mode(mode)
        sharing = data.get("models", {}).get("common", {}).get(
            "player_parameter_sharing"
        )
        if sharing != PLAYER_PARAMETER_SHARING:
            raise ValueError(
                "all seats must share one Policy and one Layout model instance"
            )
        rollout = data["rollout"]
        grpo = data["grpo"]
        learner = data["learner"]
        runtime = data.get("runtime", {})
        ppo = data.get("ppo", {})
        model_selection = data.get("model_selection", {})
        if not isinstance(model_selection, dict):
            raise ValueError("model_selection must be a mapping")
        algorithm = "grpo" if normalized is TrainingMode.TWO_PLAYER else "ppo"
        configured_dead_rules = runtime.get("dead_rules_enabled", True)
        if not isinstance(configured_dead_rules, bool):
            raise ValueError("runtime.dead_rules_enabled must be a boolean")
        selected_dead_rules = (
            configured_dead_rules
            if dead_rules_enabled is None
            else dead_rules_enabled
        )
        if not isinstance(selected_dead_rules, bool):
            raise ValueError("dead_rules_enabled override must be a boolean")
        if overrides and "dead_rules_enabled" in overrides:
            raise ValueError(
                "pass dead_rules_enabled as its named argument, not in overrides"
            )
        if algorithm == "grpo" and rollout["root_candidate_count"] != ROOT_CANDIDATE_COUNT:
            raise ValueError("runtime requires exactly four root candidates")
        if (
            algorithm == "grpo" and rollout["environment_replicas_per_root_candidate"]
            != REPLICAS_PER_CANDIDATE
        ):
            raise ValueError("runtime requires exactly two replicas per candidate")

        if tiny:
            model = ModelConfig.tiny(dead_rules_enabled=selected_dead_rules)
        elif model_scale == "bootstrap":
            model = _model_config(
                data, dead_rules_enabled=selected_dead_rules
            )
        elif model_scale == "main":
            model = ModelConfig.main(dead_rules_enabled=selected_dead_rules)
        elif model_scale == "extended":
            model = ModelConfig.extended(dead_rules_enabled=selected_dead_rules)
        else:
            raise ValueError(
                "model_scale must be bootstrap, main, or extended"
            )
        model = replace(
            model,
            inference_board_cache_entries=int(
                runtime.get("inference_board_cache_entries", 65536)
            ),
            inference_temporal_cache_entries=int(
                ppo.get("inference_temporal_cache_entries", 96)
                if algorithm == "ppo"
                else runtime.get("inference_temporal_cache_entries", 192)
            ),
            incremental_inference=bool(
                runtime.get("incremental_inference", True)
            ),
            paged_kv_cache=bool(runtime.get("paged_kv_cache", True)),
            paged_kv_length_bucket_tokens=int(
                runtime.get("paged_kv_length_bucket_tokens", 1001)
            ),
        )
        profile_pool = runtime.get("base_game_pool_size", {})
        microbatch_profile = runtime.get("policy_microbatch", 1)
        policy_microbatch = (
            int(microbatch_profile.get(model_scale, 1))
            if isinstance(microbatch_profile, dict)
            else int(microbatch_profile)
        )
        values: dict[str, Any] = {
            "algorithm": algorithm,
            "discount": float(ppo.get("discount", 1.0)),
            "gae_lambda": float(ppo.get("gae_lambda", 0.95)),
            "value_coefficient": float(ppo.get("value_coefficient", 0.5)),
            "value_clip_epsilon": float(ppo.get("value_clip_epsilon", 0.2)),
            "critic_learning_rate": float(ppo.get("critic_learning_rate", 1.0e-4)),
            "critic_epochs": int(ppo.get("critic_epochs", 3)),
            "ppo_sequence_training": bool(ppo.get("sequence_training", True)),
            "ppo_max_samples_per_sequence": int(ppo.get("max_samples_per_sequence", 64)),
            "ppo_minibatch_samples": int(ppo.get("optimizer_minibatch_samples", 512)),
            "grpo_equivalent_plies": None,
            "grpo_mean_remaining_plies": float(ppo.get("grpo_mean_remaining_plies", 334.5)),
            # Preserve the YAML types so malformed booleans and fractional
            # schedules fail validation instead of changing meaning silently.
            "arena_enabled": model_selection.get("enabled", True),
            "arena_start_percent": model_selection.get("start_percent", 30),
            "arena_interval_percent": model_selection.get("interval_percent", 5),
            "arena_games": model_selection.get("games", 1000),
            "arena_max_plies": model_selection.get("max_plies", 2000),
            "arena_seed": model_selection.get("seed", 20260910),
            "arena_temporal_cache_entries": model_selection.get(
                "temporal_cache_entries", 8
            ),
            "arena_parallel_games": model_selection.get("parallel_games", 32),
            "arena_inference_batch_size": model_selection.get("inference_batch_size", 32),
            "arena_environment_workers": model_selection.get("environment_workers", 4),
            "output_root": Path(runtime.get("output_root", "runs")),
            "total_updates": int(runtime.get("total_updates", 200000)),
            "target_continuation_plies": runtime.get(
                "target_continuation_plies"
            ),
            "target_environment_plies": runtime.get("target_environment_plies"),
            "seed": int(runtime.get("seed", 20260902)),
            "device": str(runtime.get("device", "auto")),
            "amp": str(runtime.get("amp", "bfloat16")),
            "enable_tf32": bool(runtime.get("enable_tf32", True)),
            "enable_torch_compile": bool(
                runtime.get("enable_torch_compile", False)
            ),
            "base_game_pool_size": int(profile_pool.get(normalized.value, 32)),
            "max_game_plies": runtime.get("max_game_plies", 2000),
            "anchor_batch": int(learner["policy_batch"]["global_target"]),
            "policy_microbatch": policy_microbatch,
            "actor_inference_batch": int(
                runtime.get("actor_inference_batch", 64)
            ),
            "rollout_anchor_wave_size": int(
                runtime.get("rollout_anchor_wave_size", 8)
            ),
            "rollout_environment_workers": int(
                runtime.get("rollout_environment_workers", 2)
            ),
            "learner_length_bucketing": bool(
                runtime.get("learner_length_bucketing", True)
            ),
            "policy_epochs": int(
                runtime.get("policy_epochs", learner["max_epochs_per_batch"])
            ),
            "policy_learning_rate": float(learner["learning_rate"]["initial"]),
            "minimum_learning_rate": float(learner["learning_rate"]["minimum"]),
            "layout_learning_rate": float(
                runtime.get("layout_learning_rate", 5.0e-5)
            ),
            "warmup_updates": int(runtime.get("warmup_updates", 2000)),
            "clip_epsilon": float(learner["clip_epsilon"]),
            "gradient_norm_clip": float(learner["gradient_norm_clip"]),
            "entropy_coefficient": float(
                learner["entropy_coefficient"]["initial"]
            ),
            "kl_coefficient": float(learner["kl"]["initial_coefficient"]),
            "target_kl": float(learner["kl"]["target_per_action"]),
            "early_stop_kl_multiple": float(
                learner["early_stop"]["observed_kl_multiple"]
            ),
            "early_stop_clip_fraction": float(
                learner["early_stop"]["clipped_sample_fraction"]
            ),
            "advantage_epsilon": float(grpo["advantage_epsilon"]),
            "layout_update_interval": int(
                runtime.get("layout_update_interval", 8)
            ),
            "layout_outcomes_per_update": int(
                runtime.get("layout_outcomes_per_update", 64)
            ),
            "reference_refresh_updates": int(
                runtime.get("reference_refresh_updates", 50000)
            ),
            "checkpoint_every_updates": int(
                runtime.get("checkpoint_every_updates", 10)
            ),
            "archive_every_updates": int(
                runtime.get("archive_every_updates", 500)
            ),
            "keep_checkpoint_archives": int(
                runtime.get("keep_checkpoint_archives", 10)
            ),
            "metrics_every_updates": int(runtime.get("metrics_every_updates", 1)),
            "resource_monitor_interval_seconds": float(
                runtime.get("resource_monitor_interval_seconds", 30.0)
            ),
            "auto_reduce_microbatch_on_oom": bool(
                runtime.get("auto_reduce_microbatch_on_oom", True)
            ),
            "minimum_policy_microbatch": int(
                runtime.get("minimum_policy_microbatch", 1)
            ),
        }
        if algorithm == "ppo":
            values.update(
                base_game_pool_size=int(ppo.get("base_game_pool_size", 8)),
                actor_inference_batch=int(ppo.get("actor_inference_batch", 8)),
                anchor_batch=int(ppo.get("transition_batch", 512)),
                target_environment_plies=ppo.get(
                    "target_environment_plies",
                    values["target_environment_plies"] if values["target_environment_plies"] is not None
                    else values["target_continuation_plies"],
                ),
                target_continuation_plies=None,
            )
        if tiny:
            values.update(
                {
                    "total_updates": 1,
                    "target_continuation_plies": None,
                    "target_environment_plies": None,
                    "base_game_pool_size": 2,
                    "max_game_plies": 4,
                    "anchor_batch": 1,
                    "policy_microbatch": 1,
                    "actor_inference_batch": 16,
                    "rollout_anchor_wave_size": 1,
                    "rollout_environment_workers": 1,
                    "policy_epochs": 1,
                    "critic_epochs": 1,
                    "warmup_updates": 1,
                    "layout_update_interval": 1,
                    "layout_outcomes_per_update": 2,
                    "checkpoint_every_updates": 1,
                    "archive_every_updates": 1,
                    "arena_enabled": False,
                }
            )
        if overrides:
            selected_overrides = dict(overrides)
            if algorithm == "ppo" and "target_continuation_plies" in selected_overrides:
                if "target_environment_plies" in selected_overrides:
                    raise ValueError("choose one environment budget option, not both aliases")
                # Old PPO callers used the GRPO field name for the same counter.
                selected_overrides["target_environment_plies"] = selected_overrides.pop(
                    "target_continuation_plies"
                )
            elif "target_environment_plies" in selected_overrides and "target_continuation_plies" not in selected_overrides:
                values["target_continuation_plies"] = None
            values.update(selected_overrides)
        if values["grpo_equivalent_plies"] is not None:
            if algorithm != "ppo":
                raise ValueError("GRPO-equivalent coverage is only a PPO planning option")
            if overrides and any(overrides.get(key) is not None for key in (
                "target_environment_plies", "target_continuation_plies"
            )):
                raise ValueError("choose an environment step target or a GRPO-equivalent budget, not both")
            values["target_environment_plies"] = equivalent_ppo_decisions(
                values["grpo_equivalent_plies"], values["grpo_mean_remaining_plies"]
            )
            if not tiny and (not overrides or "warmup_updates" not in overrides):
                # Former GRPO warmup: 2000 updates x 128 root decision states.
                values["warmup_updates"] = max(1, math.ceil(256_000 / values["anchor_batch"]))
        if values["max_game_plies"] is not None:
            values["max_game_plies"] = int(values["max_game_plies"])
        if values["target_continuation_plies"] is not None:
            values["target_continuation_plies"] = int(
                values["target_continuation_plies"]
            )
        if values["target_environment_plies"] is not None:
            target = values["target_environment_plies"]
            if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
                raise ValueError("target environment plies must be a positive integer or None")
            if algorithm == "ppo" and not tiny and (not overrides or "total_updates" not in overrides):
                if values["anchor_batch"] <= 0:
                    raise ValueError("anchor_batch must be positive")
                # This horizon assumes the current branch-free PPO collector.
                # Actual environment interactions, including any extra branches,
                # are the stopping counter; optimizer epochs never add steps.
                values["total_updates"] = (
                    target + values["anchor_batch"] - 1
                ) // values["anchor_batch"]
        result = cls(
            mode=normalized,
            model=model,
            dead_rules_enabled=selected_dead_rules,
            source_path=source,
            raw_config=data,
            **values,
        )
        result.validate()
        return result

    def validate(self) -> None:
        expected_algorithm = "grpo" if self.mode is TrainingMode.TWO_PLAYER else "ppo"
        if self.algorithm != expected_algorithm:
            raise ValueError(f"{self.mode.value} requires {expected_algorithm}")
        if not 0 < self.discount <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("discount must be in (0, 1] and gae_lambda in [0, 1]")
        if self.value_coefficient <= 0 or self.value_clip_epsilon <= 0:
            raise ValueError("value loss coefficient and clipping must be positive")
        if self.critic_learning_rate <= 0:
            raise ValueError("critic learning rate must be positive")
        if self.model.dead_rules_enabled != self.dead_rules_enabled:
            raise ValueError("model and training dead-rule modes must match")
        positive = {
            "total_updates": self.total_updates,
            "base_game_pool_size": self.base_game_pool_size,
            "anchor_batch": self.anchor_batch,
            "policy_microbatch": self.policy_microbatch,
            "actor_inference_batch": self.actor_inference_batch,
            "rollout_anchor_wave_size": self.rollout_anchor_wave_size,
            "rollout_environment_workers": self.rollout_environment_workers,
            "policy_epochs": self.policy_epochs,
            "critic_epochs": self.critic_epochs,
            "ppo_max_samples_per_sequence": self.ppo_max_samples_per_sequence,
            "ppo_minibatch_samples": self.ppo_minibatch_samples,
            "layout_update_interval": self.layout_update_interval,
            "layout_outcomes_per_update": self.layout_outcomes_per_update,
            "checkpoint_every_updates": self.checkpoint_every_updates,
            "archive_every_updates": self.archive_every_updates,
            "minimum_policy_microbatch": self.minimum_policy_microbatch,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.policy_microbatch > self.anchor_batch:
            raise ValueError("policy microbatch cannot exceed anchor batch")
        if self.minimum_policy_microbatch > self.policy_microbatch:
            raise ValueError(
                "minimum policy microbatch cannot exceed the initial microbatch"
            )
        if self.resource_monitor_interval_seconds <= 0:
            raise ValueError("resource monitor interval must be positive")
        if (
            self.target_continuation_plies is not None
            and self.target_continuation_plies <= 0
        ):
            raise ValueError("target continuation plies must be positive or None")
        if self.target_environment_plies is not None:
            if (isinstance(self.target_environment_plies, bool)
                    or not isinstance(self.target_environment_plies, int)
                    or self.target_environment_plies <= 0):
                raise ValueError("target environment plies must be a positive integer or None")
            if self.target_continuation_plies is not None:
                raise ValueError("choose a total environment budget or a legacy continuation budget, not both")
        if self.amp not in ("bfloat16", "float16", "float32"):
            raise ValueError("amp must be bfloat16, float16, or float32")
        if self.keep_checkpoint_archives < 0:
            raise ValueError("keep checkpoint archives cannot be negative")
        if not isinstance(self.learner_length_bucketing, bool):
            raise ValueError("learner_length_bucketing must be a boolean")
        if not isinstance(self.ppo_sequence_training, bool):
            raise ValueError("ppo_sequence_training must be a boolean")
        if type(self.arena_enabled) is not bool:
            raise ValueError("arena_enabled must be a boolean")
        for name in ("arena_start_percent", "arena_interval_percent"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError(f"{name} must be an integer in [1, 100]")
        for name in (
            "arena_games", "arena_max_plies", "arena_temporal_cache_entries",
            "arena_parallel_games", "arena_inference_batch_size", "arena_environment_workers",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        games_per_group = 2 if self.mode is TrainingMode.TWO_PLAYER else 4
        if self.arena_games % games_per_group:
            raise ValueError(
                f"arena_games must be divisible by {games_per_group} for {self.mode.value}"
            )
        match_seed_span = (2 + games_per_group) * (self.arena_games // games_per_group)
        final_percent = self.arena_start_percent + (
            (100 - self.arena_start_percent) // self.arena_interval_percent
        ) * self.arena_interval_percent
        if (
            type(self.arena_seed) is not int
            or self.arena_seed < 0
            or self.arena_seed + (final_percent + 1) * match_seed_span >= 2**63
        ):
            raise ValueError(
                "arena_seed and all scheduled matches must fit a nonnegative signed 64-bit integer"
            )

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["mode"] = self.mode.value
        values["source_path"] = str(self.source_path)
        values["output_root"] = str(self.output_root)
        values["step_budget_counter"] = self.step_budget_counter
        values["step_budget_unit"] = (
            "training_environment_transitions" if self.step_budget_counter == "environment_plies"
            else "legacy_grpo_branch_transitions"
        )
        return values

    @property
    def step_budget_target(self) -> int | None:
        return (self.target_environment_plies if self.target_environment_plies is not None
                else self.target_continuation_plies)

    @property
    def step_budget_counter(self) -> str:
        return ("environment_plies" if self.target_environment_plies is not None or self.algorithm == "ppo"
                else "continuation_plies")

    @property
    def run_variant_directory(self) -> str:
        return (
            DEAD_RULES_ON_DIRECTORY
            if self.dead_rules_enabled
            else DEAD_RULES_OFF_DIRECTORY
        )

    def resolve_run_directory(
        self, base: str | Path | None = None
    ) -> Path:
        """Return a path that cannot mix the two incompatible model variants.

        ``base`` is intentionally a parent directory.  The feature tag is
        always appended unless the caller already supplied that exact tag.
        Supplying the opposite tag is rejected rather than silently writing a
        checkpoint into the wrong experiment.
        """

        root = self.output_root if base is None else Path(base)
        known_tags = {DEAD_RULES_ON_DIRECTORY, DEAD_RULES_OFF_DIRECTORY}
        mode_names = {mode.value for mode in TrainingMode}
        if root.name in known_tags:
            if root.name != self.run_variant_directory:
                raise ValueError(
                    f"run directory tag {root.name!r} contradicts "
                    f"dead_rules_enabled={self.dead_rules_enabled}"
                )
            if root.parent.name != self.mode.value:
                raise ValueError(
                    "an explicit variant directory must be nested under its "
                    f"mode directory {self.mode.value!r}"
                )
            return root
        if root.name in mode_names:
            if root.name != self.mode.value:
                raise ValueError(
                    f"run directory mode {root.name!r} contradicts "
                    f"training mode {self.mode.value!r}"
                )
            mode_root = root
        else:
            mode_root = root / self.mode.value
        return mode_root / self.run_variant_directory


def equivalent_ppo_decisions(continuation_plies: int, mean_remaining_plies: float) -> int:
    """Approximate equal root-state coverage, never equal strength or real steps."""
    if continuation_plies <= 0 or not math.isfinite(mean_remaining_plies) or mean_remaining_plies <= 0:
        raise ValueError("equivalent budget and mean remaining length must be positive and finite")
    return math.ceil(continuation_plies / (ROOT_CANDIDATE_COUNT * REPLICAS_PER_CANDIDATE * mean_remaining_plies))


def _model_config(
    data: dict[str, Any], *, dead_rules_enabled: bool
) -> ModelConfig:
    models = data["models"]
    common = models["common"]
    policy = models["policy"]
    board = policy["board_encoder"]
    temporal = policy["temporal_transformer"]
    layout = models["layout"]
    history = data["history"]
    return ModelConfig(
        board_dim=int(board["output_dim"]),
        board_layers=int(board["transformer_layers"]),
        board_heads=int(board["attention_heads"]),
        board_ffn_dim=int(board["ffn_dim"]),
        temporal_dim=int(temporal["d_model"]),
        temporal_layers=int(temporal["transformer_layers"]),
        temporal_heads=int(temporal["attention_heads"]),
        temporal_ffn_dim=int(temporal["ffn_dim"]),
        layout_dim=int(layout["d_model"]),
        layout_layers=int(layout["transformer_layers"]),
        layout_heads=int(layout["attention_heads"]),
        layout_ffn_dim=int(layout["ffn_dim"]),
        max_transitions=int(history["max_previous_decisions"]),
        dropout=float(common["dropout"]),
        dead_rules_enabled=dead_rules_enabled,
    )
