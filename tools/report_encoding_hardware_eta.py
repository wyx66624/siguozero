"""Export current inputs and conditional GPU scenarios from a measured baseline.

Run from the repository root with PYTHONPATH=src. This does not train a model.
The accelerated fraction is an assumption, not measured GPU utilization.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from junqi.training.encoding import (
    ACTION_ENCODER_TYPE, ActionFeatures, BOARD_CODE_VOCAB_SIZE, MAX_BOARD_POINTS, OWN_PIECE_CODES,
    action_point_coordinates,
    encode_known_casualties, encode_visible_board, exact_piece_code,
)
from junqi.training.models import (
    GamePolicyTransformer, GameValueTransformer, ModelConfig,
    PieceConditionedLayoutPointerDecoder, PublicActionEncoder, WholeBoardEncoder,
)
from junqi.training.modes import new_game
from junqi.training.settings import TrainingSettings


HARDWARE = {
    "RTX 4090 24GB": {
        "dense_bf16_fp32_accumulate_tflops": 165.2,
        "memory_bandwidth_gb_s": 1008,
        "source": "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
    },
    "RTX PRO 6000 Blackwell Workstation 96GB": {
        "dense_bf16_fp32_accumulate_tflops": 503.8,
        "memory_bandwidth_gb_s": 1792,
        "source": "https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1.0.pdf",
    },
    "H100 SXM 80GB": {
        "dense_bf16_fp32_accumulate_tflops": 1979 / 2,
        "memory_bandwidth_gb_s": 3350,
        "source": "https://www.nvidia.com/en-us/data-center/h100/",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path,
                        default=Path("docs/benchmarks/whole_board_eta_20260910.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/benchmarks/encoding_hardware_scenarios_rev21_20260910.json"))
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline["architecture"] != "whole_board_one_hot_linear":
        raise ValueError("the baseline must use the current whole-board architecture")
    target = baseline["target_environment_steps_per_mode"]
    availability = baseline["availability"]
    if baseline["save_every_updates"] != 1:
        raise ValueError("these scenarios assume one complete save per outer update")
    for probe in baseline["probes"]:
        if hashlib.sha256(Path(probe["source"]).read_bytes()).hexdigest() != probe["source_sha256"]:
            raise ValueError("source probe changed after the baseline was generated")

    settings = TrainingSettings.from_yaml(
        Path("configs/bootstrap.yaml"), "four_dark", model_scale="main",
    )
    main_config = settings.model
    for probe in baseline["probes"]:
        measured_config = probe.get("model_config", {})
        for field in ("board_dim", "temporal_dim", "temporal_layers", "temporal_ffn_dim"):
            if measured_config.get(field) != getattr(main_config, field):
                raise ValueError(f"baseline {field} does not match the current four-player model")
    board = WholeBoardEncoder(main_config)
    action = PublicActionEncoder(main_config.board_dim)
    board_parameters = sum(p.numel() for p in board.parameters())
    action_parameters = sum(p.numel() for p in action.parameters())
    assert board.input_dim == 18009
    assert board_parameters == (board.input_dim + 1) * main_config.board_dim
    assert action_parameters == 6 * main_config.board_dim
    parameter_counts = {}
    with torch.device("meta"):
        for scale in ("bootstrap", "main", "extended"):
            config = TrainingSettings.from_yaml(
                Path("configs/bootstrap.yaml"), "four_dark", model_scale=scale,
            ).model
            counts = {name: sum(p.numel() for p in model_type(config).parameters())
                      for name, model_type in (("policy", GamePolicyTransformer),
                                               ("critic", GameValueTransformer),
                                               ("layout", PieceConditionedLayoutPointerDecoder))}
            counts["trainable_total"] = sum(counts.values())
            counts["including_frozen_layout"] = counts["trainable_total"] + counts["layout"]
            parameter_counts[scale] = counts
    codebook = {
        kind.value: [exact_piece_code(kind, owner, "four_dark") for owner in range(4)]
        for kind in OWN_PIECE_CODES
    }
    assert len(set(itertools.chain.from_iterable(codebook.values()))) == 48
    initial_examples = []
    for mode in ("four_dark", "double_open"):
        observation = new_game(mode, seed=20260910).observe(viewer=0)
        codes = encode_visible_board(observation)
        assert len(codes) == MAX_BOARD_POINTS
        assert all(0 <= code < BOARD_CODE_VOCAB_SIZE for code in codes)
        assert sum(code == 0 for code in codes) == 29
        casualties = encode_known_casualties(observation)
        assert not any(casualties)
        initial_examples.append({
            "mode": mode, "seed": 20260910, "viewer": 0,
            "board_codes": list(codes), "raw_casualty_bits": list(casualties),
            "meaning": "one reproducible legal initial layout, not a fixed layout for all games",
        })
    with torch.no_grad():
        absent = action(torch.zeros((1, 5)), torch.zeros(1, dtype=torch.bool))
        assert absent.shape == (1, main_config.board_dim) and torch.count_nonzero(absent).item() == 0

    rows = []
    for name, hardware in HARDWARE.items():
        ratios = (hardware["memory_bandwidth_gb_s"] / 1008,
                  hardware["dense_bf16_fp32_accumulate_tflops"] / 165.2)
        speedups = [1 / ((1 - fraction) + fraction / ratio)
                    for fraction, ratio in itertools.product((0.60, 0.85), ratios)]
        hardware["gpu_stage_speedup_scenarios"] = list(ratios)
        hardware["non_save_speedup_scenario_range"] = [min(speedups), max(speedups)]
        for cards in (1, 2, 4, 8, 16):
            per_mode = {}
            for probe in baseline["probes"]:
                n = probe["total_measured_environment_steps"]
                batch = probe["per_rank_transition_batch"]
                checkpoint = probe["checkpoint_seconds"]
                seconds = (probe["weighted_seconds_with_periodic_save"]
                           - len(probe["updates"]) * checkpoint)
                work = {
                    "selected_window_average": target * seconds / n,
                    "last_measured_update": (
                        target * probe["updates"][-1]["seconds_without_save"] / batch
                    ),
                }
                # Same per-rank work; global games and rollout batch grow with N.
                # Assume the measured save latency stays constant, not GPU-scaled.
                # Extra gathered rank histories may make real multi-card saves slower.
                saves = math.ceil(target / (batch * cards)) + 1
                eta = {}
                for history, work_seconds in work.items():
                    values = [
                        (work_seconds / (cards * speedup * efficiency) + saves * checkpoint)
                        / (86400 * availability)
                        for speedup, efficiency in itertools.product(
                            (min(speedups), max(speedups)),
                            (1.0,) if cards == 1 else (0.7, 0.9),
                        )
                    ]
                    eta[history] = [min(values), max(values)]
                per_mode[probe["mode"]] = eta
            envelopes = {
                history: [min(v[history][0] for v in per_mode.values()),
                          max(v[history][1] for v in per_mode.values())]
                for history in work
            }
            rows.append({
                "hardware": name, "cards": cards,
                "global_games_by_mode": {p["mode"]: p["games"] * cards for p in baseline["probes"]},
                "global_transition_batch": batch * cards,
                "calendar_days_by_mode": per_mode, "calendar_days_across_modes": envelopes,
            })

    source_files = [args.baseline, Path(__file__), Path("configs/bootstrap.yaml"),
                    Path("src/junqi/training/encoding.py"),
                    Path("src/junqi/training/models.py"), Path("src/junqi/game.py")]
    result = {
        "status": "conditional_engineering_scenarios_not_target_gpu_measurements_or_bounds",
        "encoding_config_revision": settings.raw_config["config_revision"],
        "timing_baseline_config_revisions": baseline.get("config_revisions", []),
        "current_five_input_action_timing_measured": (
            baseline.get("config_revisions") == [settings.raw_config["config_revision"]]
            and baseline.get("action_encoder_types") == [ACTION_ENCODER_TYPE]),
        "model_parameter_counts": parameter_counts,
        "budget_per_mode": target, "availability": availability,
        "measured_baseline_device": "RTX 4090 24GB",
        "hardware": HARDWARE,
        "baseline_spec": {
            "dense_bf16_fp32_accumulate_tflops": 165.2, "memory_bandwidth_gb_s": 1008,
            "source": "https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf",
        },
        "assumptions": {
            "gpu_accelerable_fraction_of_non_save_wall_time": [0.60, 0.85],
            "fraction_is_measured": False,
            "same_relative_gpu_kernel_utilization": True,
            "multi_card_efficiency": [0.7, 0.9],
            "weak_scaling": "each rank retains the selected measured games and transition batch; global batch grows with cards",
            "sixteen_card_scaling": "conditional on adequate inter-node CPU and communication capacity; not measured",
            "checkpoint_latency": "measured single-card latency held constant; rank-gather growth unmeasured",
            "formula": "S=1/((1-f)+f/r); days=(work_seconds/(N*S*efficiency)+save_count*save_seconds)/86400/availability",
            "excluded": ["extra checkpoint archives", "selection games", "changed mature-policy history distribution",
                         "periodic layout update cost when no layout backward occurred in the probe",
                         "unmeasured larger per-card rollout concurrency on higher-memory GPUs"],
        },
        "action": {"architecture": ACTION_ENCODER_TYPE, "field_count": 5,
                   "fields": ["source_x", "source_y", "destination_x", "destination_y", "actor"],
                   "learned_field_embeddings": False, "normalization": "none",
                    "projection_output_dim": main_config.board_dim, "layer_norm": False, "parameters": action_parameters,
                   "four_dark_example_0_to_120_actor_3": ActionFeatures(0, 120, 3).as_vector("four_dark"),
                   "coordinates_by_mode": {mode: [action_point_coordinates(code, mode)
                                                   for code in range(points)]
                                           for mode, points in (("four_dark", 129), ("double_open", 129),
                                                                ("two_player", 60))}},
        "board": {"points": 129, "category_vocabulary": 139, "input_dim_dead_rules_on": board.input_dim,
                   "projection_output_dim": main_config.board_dim, "parameters": board_parameters,
                  "codebook_by_relative_owner_0_1_2_3": codebook},
        "initial_board_examples": initial_examples, "scenarios": rows,
        "source_sha256": {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": [action_parameters, board_parameters],
                      "model_parameter_counts": parameter_counts,
                      "hardware": HARDWARE,
                      "scenarios": [{"hardware": row["hardware"], "cards": row["cards"],
                                     **row["calendar_days_across_modes"]} for row in rows]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
