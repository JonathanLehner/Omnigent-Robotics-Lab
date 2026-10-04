"""Regression checks for timestamped rollout inspection."""

import tempfile
from pathlib import Path

import numpy as np

from lab import runner


def _frame(index, time_s, b0_xy, active, b0_z=0.05):
    return {
        "index": index,
        "time_s": time_s,
        "block_xy_m": {"b0": b0_xy, "b1": [1.0, 0.0]},
        "block_z_m": {"b0": b0_z, "b1": 0.1},
        "active_block_id": active,
    }


def test_post_release_motion_warning_names_active_placement():
    trace = {
        "frames": [
            _frame(0, 0.0, [0.0, 0.0], "b0", b0_z=0.2),
            _frame(1, 1.0, [0.0, 0.0], "b0"),
            _frame(2, 2.0, [0.0, 0.0], "b1"),
            _frame(3, 3.0, [0.064, 0.0], "b1"),
        ],
        "events": [
            {
                "kind": "release",
                "block_id": "b0",
                "time_s": 1.5,
                "block_xy_m": [0.0, 0.0],
            }
        ],
    }

    events = runner._inspection_events(trace)
    warnings = runner._motion_warnings(trace, events)

    assert len(warnings) == 1
    assert warnings[0]["distance_m"] == 0.064
    assert warnings[0]["onset_window_s"] == [2.0, 3.0]
    assert warnings[0]["placing_block_id"] == "b1"
    assert "b0 moved 6.4 cm" in warnings[0]["text"]
    assert "while placing b1" in warnings[0]["text"]


def test_contact_sheets_paginate_without_dropping_frames():
    frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(25)]
    metadata = [{"index": index, "time_s": float(index)} for index in range(25)]
    with tempfile.TemporaryDirectory() as temporary:
        paths = runner._save_contact_sheets(
            frames,
            metadata,
            Path(temporary) / "sheet",
            labels={},
        )

        assert len(paths) == 2
        assert all(Path(path).is_file() for path in paths)


if __name__ == "__main__":
    test_post_release_motion_warning_names_active_placement()
    test_contact_sheets_paginate_without_dropping_frames()
    print("runner inspection tests passed")
