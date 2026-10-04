"""Passive placed-block telemetry and H-025 recovery decisions.

The observer is called by :class:`lab.sim.World` after an existing simulation
step. It only reads poses: enabling telemetry adds neither simulation steps nor
randomness to the control arm.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lab import sim


DISPLACEMENT_THRESHOLD_M = 0.01
ALIGNMENT_LIMIT_M = 0.01


@dataclass
class DisplacementDetected(RuntimeError):
    """Stop the aware descent at the first recoverable displacement."""

    block_id: str


class PlacedBlockMonitor:
    """Track displacement from each block's own post-release pose."""

    def __init__(self, world, spec, pairs, cfg, log):
        prm = cfg.get("params", {})
        self.world = world
        self.enabled = bool(prm.get("placed_block_telemetry", False))
        self.aware = bool(prm.get("placed_block_monitor", False))
        self.threshold_m = float(
            prm.get("placed_block_displacement_threshold_m", DISPLACEMENT_THRESHOLD_M)
        )
        self.alignment_limit_m = float(
            prm.get("placed_block_alignment_limit_m", ALIGNMENT_LIMIT_M)
        )
        self.max_episode_recoveries = int(
            prm.get("placed_block_max_episode_recoveries", 2)
        )
        self.max_block_recoveries = int(
            prm.get("placed_block_max_recoveries_per_block", 1)
        )
        self.by_sid = {block["id"]: block for block in spec}
        self.pid_for_sid = dict(pairs)
        self.sid_for_pid = {pid: sid for sid, pid in pairs.items()}
        self.release_poses = {}
        self.rows = {}
        self.placed = set()
        self.active_pid = None
        self.watching = False
        self.recovery_counts = {}
        self.recovery_total = 0
        self.decided_events = set()
        self.log = log
        self.telemetry = {
            "schema": "h025_t_disp_v1",
            "passive": True,
            "threshold_m": self.threshold_m,
            "blocks": [],
            "triggers": [],
            "recoveries": [],
        }
        if self.enabled:
            log["placed_block_displacement"] = self.telemetry
            world.step_observer = self.observe_step

    def set_active(self, block_id):
        self.active_pid = block_id

    def set_watching(self, watching):
        self.watching = bool(watching)

    def record_release(self, block_id):
        """Use the pose immediately after release as this block's baseline."""
        pose = self.world.block_pose(block_id)
        self.release_poses[block_id] = pose
        self.placed.add(block_id)
        if block_id not in self.rows:
            sid = self.sid_for_pid.get(block_id)
            row = {
                "block_id": block_id,
                "spec_id": sid,
                "release_position_xyz_m": pose[0].tolist(),
                "release_quaternion_wxyz": pose[1].tolist(),
                "max_displacement_m_during_later_placements": 0.0,
                "final_displacement_m": 0.0,
                "onset_step": None,
                "held_block_id_at_onset": None,
            }
            self.rows[block_id] = row
            self.telemetry["blocks"].append(row)

    def support_shift(self, sid):
        """Mean actual-minus-nominal support XY shift, norm-clamped to 1 cm."""
        support_sids = list(self.by_sid[sid].get("on", []))
        offsets = []
        for support_sid in support_sids:
            support_pid = self.pid_for_sid.get(support_sid)
            if support_pid not in self.placed:
                continue
            actual, _ = self.world.block_pose(support_pid)
            nominal = sim.SITE + np.asarray(self.by_sid[support_sid]["pos"], dtype=float)
            offsets.append(actual[:2] - nominal[:2])
        if not offsets:
            return np.zeros(2)
        shift = np.mean(offsets, axis=0)
        norm = float(np.linalg.norm(shift))
        if norm > self.alignment_limit_m:
            shift = shift * (self.alignment_limit_m / norm)
        self.telemetry["triggers"].append(
            {
                "type": "support_alignment",
                "held_block_id": self.pid_for_sid.get(sid),
                "support_block_ids": [
                    self.pid_for_sid[s]
                    for s in support_sids
                    if self.pid_for_sid.get(s) in self.placed
                ],
                "shift_xy_m": shift.tolist(),
            }
        )
        return shift

    def _is_loaded(self, block_id):
        sid = self.sid_for_pid.get(block_id)
        if sid is None:
            return False
        return any(
            placed_sid is not None
            and sid in self.by_sid[placed_sid].get("on", [])
            for pid in self.placed
            if pid != block_id
            for placed_sid in [self.sid_for_pid.get(pid)]
        )

    def _decision(self, block_id):
        if self._is_loaded(block_id):
            return "unrecoverable_supported"
        if (
            self.recovery_counts.get(block_id, 0) >= self.max_block_recoveries
            or self.recovery_total >= self.max_episode_recoveries
        ):
            return "recovery_exhausted"
        return "recover"

    def observe_step(self):
        """Read every placed pose after the world's existing control step."""
        active = self.active_pid
        for block_id, release_pose in tuple(self.release_poses.items()):
            actual, _ = self.world.block_pose(block_id)
            displacement = float(np.linalg.norm(actual - release_pose[0]))
            row = self.rows[block_id]
            row["final_displacement_m"] = displacement
            if active is not None and active != block_id:
                row["max_displacement_m_during_later_placements"] = max(
                    row["max_displacement_m_during_later_placements"],
                    displacement,
                )
                if displacement > self.threshold_m and row["onset_step"] is None:
                    row["onset_step"] = int(
                        round(self.world.d.time / self.world.m.opt.timestep)
                    )
                    row["held_block_id_at_onset"] = active

            if (
                not self.aware
                or not self.watching
                or active is None
                or active == block_id
                or displacement <= self.threshold_m
                or block_id in self.decided_events
            ):
                continue
            decision = self._decision(block_id)
            self.decided_events.add(block_id)
            self.telemetry["triggers"].append(
                {
                    "type": "placed_block_displacement",
                    "block_id": block_id,
                    "held_block_id": active,
                    "displacement_m": displacement,
                    "step": int(
                        round(self.world.d.time / self.world.m.opt.timestep)
                    ),
                    "decision": decision,
                }
            )
            if decision == "recover":
                raise DisplacementDetected(block_id)

    def begin_recovery(self, block_id):
        self.recovery_counts[block_id] = self.recovery_counts.get(block_id, 0) + 1
        self.recovery_total += 1
        self.telemetry["recoveries"].append(
            {
                "block_id": block_id,
                "held_block_id": self.active_pid,
                "attempt": self.recovery_counts[block_id],
                "outcome": "started",
            }
        )

    def finish_recovery(self, block_id, target_position, outcome="replaced"):
        actual, _ = self.world.block_pose(block_id)
        row = self.telemetry["recoveries"][-1]
        row["outcome"] = outcome
        row["position_error_m"] = float(
            np.linalg.norm(actual - np.asarray(target_position, dtype=float))
        )
        # Recovery establishes a new post-release baseline for T-disp. The
        # event remains present in max/onset telemetry but is not a final DISP.
        self.release_poses[block_id] = self.world.block_pose(block_id)
        release = self.release_poses[block_id]
        self.rows[block_id]["release_position_xyz_m"] = release[0].tolist()
        self.rows[block_id]["release_quaternion_wxyz"] = release[1].tolist()
        self.rows[block_id]["final_displacement_m"] = 0.0
        # A later displacement must be observed and logged as exhausted rather
        # than silently suppressed after the one allowed recovery.
        self.decided_events.discard(block_id)

    def close(self):
        self.set_watching(False)
        self.active_pid = None
