"""Check global/per-rank launch arguments without starting CUDA training."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).parents[1] / "scripts/start_four_player_ppo.sh"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class FourPlayerLauncherTests(unittest.TestCase):
    def launch(self, gpus, *, extra_environment=None, arguments=()):
        with tempfile.TemporaryDirectory() as directory:
            fake_python = Path(directory) / "python"
            fake_python.write_text(
                "#!/usr/bin/env python3\nimport json, sys\nprint(json.dumps(sys.argv[1:]))\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o700)
            environment = {key: value for key, value in os.environ.items() if key not in {
                "NUM_GPUS", "GAMES_PER_GPU", "BASE_GAME_POOL", "ACTOR_BATCH", "MICROBATCH",
                "TRANSITION_BATCH", "TEMPORAL_CACHE_ENTRIES", "TARGET_ENVIRONMENT_PLIES",
                "ENVIRONMENT_WORKERS",
            }}
            environment.update(PYTHON_BIN=str(fake_python), NUM_GPUS=str(gpus))
            environment.update(extra_environment or {})
            return subprocess.run(
                ["bash", str(SCRIPT), "four_dark", *arguments], env=environment,
                capture_output=True, text=True, check=False,
            )

    def test_global_work_grows_with_cards_and_per_rank_work_stays_fixed(self):
        for cards in (1, 2, 8):
            with self.subTest(cards=cards):
                result = self.launch(cards)
                self.assertEqual(result.returncode, 0, result.stderr)
                args = json.loads(result.stdout)
                value = lambda flag: args[args.index(flag) + 1]
                self.assertEqual(int(value("--base-game-pool")), 48 * cards)
                self.assertEqual(int(value("--transition-batch")), 12288 * cards)
                self.assertEqual(int(value("--actor-batch")), 48)
                self.assertEqual(int(value("--temporal-cache-entries")), 576)
                self.assertEqual(int(value("--ppo-minibatch")), 512)
                self.assertEqual(int(value("--microbatch")), 32)
                self.assertEqual(int(value("--environment-workers")), 4)
                self.assertEqual(int(value("--target-environment-plies")), 3_000_000_000)
                self.assertEqual(value("--checkpoint-policy"), "periodic")
                self.assertNotIn("--checkpoint-every", args)
                self.assertEqual("torch.distributed.run" in args, cards > 1)
                if cards > 1:
                    self.assertIn(f"--nproc-per-node={cards}", args)
                    self.assertTrue(value("--run-dir").endswith(f"_{cards}gpu"))

    def test_explicit_global_budget_and_pool_overrides_are_preserved(self):
        result = self.launch(2, extra_environment={
            "GAMES_PER_GPU": "40", "BASE_GAME_POOL": "72", "TRANSITION_BATCH": "16384",
        }, arguments=("--target-environment-plies", "24576"))
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads(result.stdout)
        value = lambda flag: args[args.index(flag) + 1]
        self.assertEqual(value("--base-game-pool"), "72")
        self.assertEqual(value("--transition-batch"), "16384")
        self.assertEqual(value("--actor-batch"), "40")
        self.assertEqual(value("--temporal-cache-entries"), "480")
        self.assertEqual(args.count("--target-environment-plies"), 1)
        self.assertEqual(value("--target-environment-plies"), "24576")

    def test_invalid_card_count_is_rejected_before_launch(self):
        result = self.launch(0)
        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integers", result.stderr)
