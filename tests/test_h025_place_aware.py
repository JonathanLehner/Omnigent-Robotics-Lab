"""Focused tests for H-025 telemetry, limits, and support alignment."""

from types import SimpleNamespace

import numpy as np
import pytest

from lab.stages.placed_block_monitor import (
    DisplacementDetected,
    PlacedBlockMonitor,
)


class FakeWorld:
    def __init__(self):
        self.poses = {
            "b0": np.array([0.85, 0.0, 0.05]),
            "b1": np.array([0.85, 0.0, 0.15]),
            "b2": np.array([0.85, 0.2, 0.05]),
        }
        self.quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.d = SimpleNamespace(time=1.0)
        self.m = SimpleNamespace(opt=SimpleNamespace(timestep=0.002))
        self.step_observer = None

    def block_pose(self, block_id):
        return self.poses[block_id].copy(), self.quat.copy()


SPEC = [
    {"id": "b0", "pos": [0.0, 0.0, 0.05], "on": []},
    {"id": "b1", "pos": [0.0, 0.0, 0.15], "on": ["b0"]},
    {"id": "b2", "pos": [0.0, 0.2, 0.05], "on": []},
]
PAIRS = {block["id"]: block["id"] for block in SPEC}


def _monitor(*, aware):
    world, log = FakeWorld(), {}
    monitor = PlacedBlockMonitor(
        world,
        SPEC,
        PAIRS,
        {
            "params": {
                "placed_block_telemetry": True,
                "placed_block_monitor": aware,
            }
        },
        log,
    )
    return world, log, monitor


def test_passive_telemetry_flags_displacement_without_control_action():
    world, log, monitor = _monitor(aware=False)
    assert world.step_observer == monitor.observe_step
    monitor.record_release("b0")
    monitor.set_active("b1")
    monitor.set_watching(True)
    world.poses["b0"][1] += 0.02

    monitor.observe_step()

    row = log["placed_block_displacement"]["blocks"][0]
    assert row["max_displacement_m_during_later_placements"] == pytest.approx(0.02)
    assert row["final_displacement_m"] == pytest.approx(0.02)
    assert row["held_block_id_at_onset"] == "b1"
    assert log["placed_block_displacement"]["triggers"] == []


def test_aware_recovery_once_then_logs_exhausted():
    world, log, monitor = _monitor(aware=True)
    monitor.record_release("b0")
    monitor.set_active("b1")
    monitor.set_watching(True)
    world.poses["b0"][1] += 0.02

    with pytest.raises(DisplacementDetected) as event:
        monitor.observe_step()
    assert event.value.block_id == "b0"
    monitor.begin_recovery("b0")
    monitor.finish_recovery("b0", world.poses["b0"])

    world.poses["b0"][1] += 0.02
    monitor.observe_step()
    decisions = [
        row["decision"]
        for row in log["placed_block_displacement"]["triggers"]
        if row["type"] == "placed_block_displacement"
    ]
    assert decisions == ["recover", "recovery_exhausted"]
    assert monitor.recovery_counts == {"b0": 1}
    assert monitor.recovery_total == 1


def test_loaded_block_is_not_recovered():
    world, log, monitor = _monitor(aware=True)
    monitor.record_release("b0")
    monitor.record_release("b1")
    monitor.set_active("b2")
    monitor.set_watching(True)
    world.poses["b0"][1] += 0.02

    monitor.observe_step()

    trigger = next(
        row
        for row in log["placed_block_displacement"]["triggers"]
        if row["type"] == "placed_block_displacement"
    )
    assert trigger["block_id"] == "b0"
    assert trigger["decision"] == "unrecoverable_supported"
    assert monitor.recovery_total == 0


def test_episode_recovery_cap_applies_across_blocks():
    world, log, monitor = _monitor(aware=True)
    monitor.max_episode_recoveries = 2
    monitor.record_release("b0")
    monitor.record_release("b2")
    monitor.recovery_counts = {"b0": 1, "b2": 1}
    monitor.recovery_total = 2
    monitor.set_active("b1")
    monitor.set_watching(True)
    world.poses["b2"][1] += 0.02

    monitor.observe_step()

    trigger = next(
        row
        for row in log["placed_block_displacement"]["triggers"]
        if row["type"] == "placed_block_displacement"
    )
    assert trigger["block_id"] == "b2"
    assert trigger["decision"] == "recovery_exhausted"


def test_support_alignment_uses_actual_pose_and_clamps_norm():
    world, log, monitor = _monitor(aware=True)
    monitor.record_release("b0")
    world.poses["b0"][:2] += [0.018, 0.024]

    shift = monitor.support_shift("b1")

    assert np.linalg.norm(shift) == pytest.approx(0.01)
    assert shift.tolist() == pytest.approx([0.006, 0.008])
