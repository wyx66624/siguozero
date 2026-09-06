"""Validated YAML settings for reproducible training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
        if rollout["root_candidate_count"] != ROOT_CANDIDATE_COUNT:
            raise ValueError("runtime requires exactly four root candidates")
        if (
            rollout["environment_replicas_per_root_candidate"]
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
                runtime.get("inference_temporal_cache_entries", 192)
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
            "output_root": Path(runtime.get("output_root", "runs")),
            "total_updates": int(runtime.get("total_updates", 200000)),
            "target_continuation_plies": runtime.get(
                "target_continuation_plies"
            ),
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
        if tiny:
            values.update(
                {
                    "total_updates": 1,
                    "base_game_pool_size": 2,
                    "max_game_plies": 4,
                    "anchor_batch": 1,
                    "policy_microbatch": 1,
                    "actor_inference_batch": 16,
                    "rollout_anchor_wave_size": 1,
                    "rollout_environment_workers": 1,
                    "policy_epochs": 1,
                    "warmup_updates": 1,
                    "layout_update_interval": 1,
                    "layout_outcomes_per_update": 2,
                    "checkpoint_every_updates": 1,
                    "archive_every_updates": 1,
                }
            )
        if overrides:
            values.update(overrides)
        if values["max_game_plies"] is not None:
            values["max_game_plies"] = int(values["max_game_plies"])
        if values["target_continuation_plies"] is not None:
            values["target_continuation_plies"] = int(
                values["target_continuation_plies"]
            )
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
        if self.amp not in ("bfloat16", "float16", "float32"):
            raise ValueError("amp must be bfloat16, float16, or float32")
        if self.keep_checkpoint_archives < 0:
            raise ValueError("keep checkpoint archives cannot be negative")
        if not isinstance(self.learner_length_bucketing, bool):
            raise ValueError("learner_length_bucketing must be a boolean")

    def serializable(self) -> dict[str, Any]:
        values = asdict(self)
        values["mode"] = self.mode.value
        values["source_path"] = str(self.source_path)
        values["output_root"] = str(self.output_root)
        return values

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
