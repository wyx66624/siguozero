"""Contracts for reusing a managed monitor and publishing its process identity."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from junqi.web.service_control import UNIT, ensure_managed_console, write_service_metadata


class ManagedServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.config = self.root / "console.json"
        self.properties = {
            "LoadState": "loaded", "ActiveState": "active", "MainPID": "42",
            "UnitFileState": "enabled", "WorkingDirectory": str(self.root),
            "ExecStart": f"python -m junqi.web.server --config {self.config} --port 8765",
        }

    def test_uninstalled_service_keeps_explicit_launcher_available(self):
        with patch("junqi.web.service_control.unit_properties", return_value={"LoadState": "not-found"}), \
                patch("junqi.web.service_control.subprocess.run") as command:
            self.assertIsNone(ensure_managed_console(self.root, self.config, 8765))
            command.assert_not_called()

    def test_another_installation_is_not_started(self):
        for field, value in (("WorkingDirectory", "/another/project"),
                             ("ExecStart", "python -m junqi.web.server --config other.json --port 8765"),
                             ("ExecStart", f"python --config {self.config} --port 8766")):
            with self.subTest(field=field, value=value), \
                    patch("junqi.web.service_control.unit_properties", return_value={**self.properties, field:value}), \
                    patch("junqi.web.service_control.subprocess.run") as command:
                with self.assertRaisesRegex(RuntimeError, "another project"):
                    ensure_managed_console(self.root, self.config, 8765)
                command.assert_not_called()

    def test_reuses_only_the_systemd_process_that_owns_the_http_port(self):
        with patch("junqi.web.service_control.unit_properties", return_value=self.properties), \
                patch("junqi.web.service_control.subprocess.run") as command, \
                patch("junqi.web.service_control.wait_ready", return_value={"ok":True, "pid":42}) as ready:
            result = ensure_managed_console(self.root, self.config, 8765)
            self.assertEqual(result["pid"], 42)
            self.assertEqual(result["managed_by"], UNIT)
            self.assertTrue(result["enabled"])
            command.assert_called_once_with(["systemctl", "start", UNIT], check=True, timeout=30)
            ready.return_value = {"ok":True, "pid":99}
            with self.assertRaisesRegex(RuntimeError, "Port owner"):
                ensure_managed_console(self.root, self.config, 8765)

    def test_restart_replaces_metadata_with_current_process_identity(self):
        directory = self.root / "output/local_console"
        directory.mkdir(parents=True)
        (directory / "service.json").write_text('{"pid":99999999}')
        write_service_metadata(self.root, self.config, 8765, managed_by=UNIT)
        result = json.loads((directory / "service.json").read_text())
        self.assertEqual(result["pid"], os.getpid())
        self.assertEqual(result["managed_by"], UNIT)
        self.assertEqual(result["config"], str(self.config))
        self.assertEqual(list(directory.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
