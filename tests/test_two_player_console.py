"""Dedicated console configuration and page isolation; no GPU model required."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from junqi.web.monitor import Monitor
from junqi.web.server import Application, Handler


ROOT = Path(__file__).parents[1]


def console_config():
    return json.loads((ROOT / "configs/two_player_console.json").read_text(encoding="utf-8"))


def test_unstarted_run_has_planned_mappo_environment_budget_without_fake_metrics(tmp_path):
    monitor = Monitor(console_config(), tmp_path)
    status = monitor.run_status(monitor.specs[0])
    assert status["configuration_source"] == "planned"
    assert status["algorithm"] == "mappo"
    assert status["counter"] == "environment_plies"
    assert status["target"] == 3_000_000_000
    assert status["steps"] == status["update"] == 0
    assert status["checkpoint_steps"] is None and status["metrics"] == {}
    assert status["cycle_rate"] is None and status["processes"] == []
    assert status["next_checkpoint_steps"] == status["checkpoint_interval_steps"] == 25_000_000
    assert status["training_plan"]["historical_start_environment_plies"] == 500_000_000
    assert status["training_plan"]["draw_reward"] == -.15
    assert monitor.public_catalog() == []
    assert not monitor.specs[0]["path"].exists()
    assert not (tmp_path / "output/local_console").exists()


def test_real_run_configuration_replaces_planned_values(tmp_path):
    monitor = Monitor(console_config(), tmp_path)
    run = monitor.specs[0]["path"]
    run.mkdir(parents=True)
    (run / "resolved_config.json").write_text(json.dumps({
        "algorithm": "mappo", "target_environment_plies": 1_000_000_000,
        "draw_reward": -.2, "historical_enabled": False,
    }))
    status = monitor.run_status(monitor.specs[0])
    assert status["configuration_source"] == "resolved"
    assert status["counter"] == "environment_plies"
    assert status["target"] == 1_000_000_000
    assert status["training_plan"]["draw_reward"] == -.2
    assert status["training_plan"]["historical_enabled"] is False
    assert status["training_plan"]["historical_start_environment_plies"] is None


def page(app, path):
    replies = []
    handler = SimpleNamespace(server=SimpleNamespace(app=app), path=path,
        _local_request=lambda: True, _respond=lambda *args: replies.append(args))
    Handler.do_GET(handler)
    assert replies[0][0] == 200
    return replies[0][1].decode("utf-8")


def test_dedicated_pages_have_two_seats_and_reuse_monitor_play_and_replay_scripts(tmp_path):
    app = Application(console_config(), tmp_path)
    try:
        dashboard, play = page(app, "/"), page(app, "/play")
        assert "双人军棋" in dashboard and "/dashboard.js" in dashboard
        assert 'data-console-mode="two_player"' in play and "/play.js" in play
        assert "60 点棋盘" in play and 'id="review-controls"' in play
        assert 'value="1">北方 · 后手' in play
        assert "对家队友" not in play and "其余三个席位" not in play
        assert app.monitor.output_directory == tmp_path / "output/two_player_console"
    finally:
        app.close()


def test_dedicated_console_rejects_four_player_catalog_and_existing_console_stays_generic(tmp_path):
    wrong = console_config()
    wrong["runs"][0]["mode"] = "four_dark"
    with pytest.raises(ValueError, match="matching runs"):
        Application(wrong, tmp_path)
    generic = copy.deepcopy(wrong)
    generic.pop("console_mode")
    app = Application(generic, tmp_path)
    try:
        assert "四国 <span>SiguoZero" in page(app, "/")
        assert "对家队友" in page(app, "/play")
    finally:
        app.close()
