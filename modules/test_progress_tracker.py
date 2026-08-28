"""Focused tests for progress-tracker status payloads."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from modules.progress_tracker import emit_final_status


class ProgressTrackerFinalStatusTests(unittest.TestCase):
    """Verify final status payload exposes both realtime measures."""

    def test_emit_final_status_includes_raw_and_total_realtime(self):
        """Emit raw and whole-run realtime values in final GUI status payload."""
        with patch("modules.progress_tracker._relay_status") as relay_status:
            emit_final_status(
                elapsed="0:02:00",
                audio="0:10:00",
                realtime="5.00x",
                realtime_total="4.00x",
                total_elapsed="0:02:30",
            )

        relay_status.assert_called_once_with(
            {
                "operation": "✅ Processing Complete!",
                "elapsed": "0:02:00",
                "audio": "0:10:00",
                "realtime": "5.00x",
                "realtime_total": "4.00x",
                "remaining": "0:00:00",
                "total_elapsed": "0:02:30",
            }
        )


if __name__ == "__main__":
    unittest.main()
