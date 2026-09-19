"""Adopt changed training rewards without discarding training state."""
from __future__ import annotations


def reconcile_draw_objective(payload, settings, *, adopt: bool = False):
    config = payload.get("config", {})
    old = float(config.get("draw_reward", config.get("raw_config", {}).get(
        "rules", {}).get("terminal_reward", {}).get("draw", 0.0)))
    target = float(settings.draw_reward)
    if old == target:
        return
    if not adopt:
        raise RuntimeError("draw reward changed across resume; use --adopt-draw-penalty to retain state and adopt it explicitly")
    state = payload["trainer_state"]
    changed = 0
    for item in state.get("layout_buffer", []):
        if float(item["reward"]) == old:
            item["reward"] = target
            changed += 1
    migration = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from_draw_reward": old, "to_draw_reward": target,
        "action_feature_dim": 6, "layout_outcomes_remapped": changed,
        "gae": ("fixed_team_scalar_global_value" if settings.algorithm == "mappo"
                else "signed_team_outcome_plus_unsigned_shared_draw_penalty"),
    }
    state["draw_objective_migration"] = migration
    config["draw_reward"] = target


def reconcile_flag_capture_objective(payload, settings, *, adopt: bool = False):
    # Old checkpoints have no event shaping, irrespective of their raw YAML.
    config = payload["config"]
    old = float(config.get("flag_capture_reward", 0.0))
    target = float(settings.flag_capture_reward)
    if old == target:
        return
    if not adopt:
        raise RuntimeError("flag capture reward changed across resume; use --adopt-flag-capture-reward to retain state and adopt it explicitly")
    state = payload["trainer_state"]
    state["flag_capture_objective_migration"] = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from_flag_capture_reward": old, "to_flag_capture_reward": target,
        "scope": "future_ppo_transitions_only",
        "perspective": "value_owner_team",
        "terminal_reward": "additive",
    }
    config["flag_capture_reward"] = target


def reconcile_futile_attack_objective(payload, settings, *, adopt: bool = False):
    from .futile_attack import FutileAttackPenalty
    config = payload["config"]
    old = float(config.get("futile_attack_penalty", 0.0))
    target = float(settings.futile_attack_penalty)
    unchanged = (old == target and (target == 0 or (
        config.get("futile_attack_ema_decay", 0.95) == settings.futile_attack_ema_decay
        and config.get("target_environment_plies") == settings.target_environment_plies)))
    if unchanged:
        return
    if not adopt:
        raise RuntimeError("futile attack penalty changed across resume; use --adopt-futile-attack-penalty to preserve state and adopt it explicitly")
    state = payload["trainer_state"]
    progress = int(state.get("cumulative", {}).get("environment_plies", 0))
    controller = FutileAttackPenalty(target, settings.target_environment_plies, settings.futile_attack_ema_decay)
    if old > 0:
        controller.load_state_dict(state.get("futile_attack_penalty"))
    else:
        controller.start_environment_plies = progress
    state["futile_attack_penalty"] = controller.state_dict()
    state["futile_attack_objective_migration"] = {
        "update": int(payload["update"]), "environment_plies": progress,
        "from_penalty": old, "to_penalty": target,
        "start_environment_plies": controller.start_environment_plies,
        "stop_environment_plies": controller.stop_environment_plies,
        "ema_decay": settings.futile_attack_ema_decay,
        "scope": "future_ppo_transitions_only", "perspective": "acting_learner_team",
        "evidence": "per_viewer_public_battle_history", "terminal_reward": "additive",
    }
    config["futile_attack_penalty"] = target
    config["futile_attack_ema_decay"] = settings.futile_attack_ema_decay
    config["target_environment_plies"] = settings.target_environment_plies


def reconcile_bomb_objective(payload, settings, *, adopt: bool = False):
    from .tactical_rewards import BOMB_REWARD_FIELDS
    config = payload["config"]
    old = {field: float(config.get(field, 0.0)) for field in BOMB_REWARD_FIELDS.values()}
    target = {field: float(getattr(settings, field)) for field in BOMB_REWARD_FIELDS.values()}
    if old == target and (not any(target.values()) or
                          config.get("target_environment_plies") == settings.target_environment_plies):
        return
    if not adopt:
        raise RuntimeError("bomb rewards changed across resume; use --adopt-bomb-rewards to preserve state and adopt them explicitly")
    state = payload["trainer_state"]
    state["bomb_objective_migration"] = {
        "update": int(payload["update"]),
        "environment_plies": state.get("cumulative", {}).get("environment_plies", 0),
        "from_rewards": old, "to_rewards": target,
        "from_target_environment_plies": config.get("target_environment_plies"),
        "to_target_environment_plies": settings.target_environment_plies,
        "scope": "future_mappo_transitions_only", "perspective": "bomb_owner_team",
        "schedule": "linear_global_steps_zero_at_half_budget", "terminal_reward": "additive",
    }
    config.update(target)
