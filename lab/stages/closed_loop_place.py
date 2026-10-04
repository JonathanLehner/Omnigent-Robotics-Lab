"""Closed-loop weld placement for rung 4.

This keeps the rung-0 pick, carry, weld grasp, and release sequence, but
measures the block pose at the nominal place waypoint and corrects the
translation before release.
"""

import numpy as np

from lab import sim
from lab.pipeline import stage
from lab.stages.placed_block_monitor import (
    DisplacementDetected,
    PlacedBlockMonitor,
)


def _closed_loop_correction(
    world,
    pid,
    target,
    commanded_tip,
    quat,
    passes,
    correction_tol,
    max_step,
    correction_seg,
    correction_dwell,
):
    corrections = []
    for _ in range(passes):
        actual_pos, _ = world.block_pose(pid)
        residual = target - actual_pos
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= correction_tol:
            break
        correction = residual
        if residual_norm > max_step:
            correction = correction * (max_step / residual_norm)
        commanded_tip = commanded_tip + correction
        q_target, ik_residual, _ = world.ik(commanded_tip, quat)
        world.move_joints(q_target, correction_seg, dwell=correction_dwell)
        corrected_pos, _ = world.block_pose(pid)
        corrections.append(
            {
                "command_m": correction.tolist(),
                "ik_residual_m": ik_residual,
                "remaining_err_m": float(np.linalg.norm(corrected_pos - target)),
            }
        )
    return commanded_tip, corrections


def _recover_displaced_block(
    world,
    monitor,
    displaced_pid,
    held_pick_block_pos,
    held_type,
    quat,
    back,
    seg,
    lift,
    targets,
    block_types,
    passes,
    correction_tol,
    max_step,
    correction_seg,
    correction_dwell,
    clear,
):
    """Park the held block, replace the displaced block, then let the caller retry."""
    monitor.set_watching(False)
    monitor.begin_recovery(displaced_pid)

    tip, _ = world.tip_pose()
    world.move_joints(world.ik(tip + [0.0, 0.0, 0.03], quat)[0], seg)
    held_handle = sim.handle_point(held_type)
    held_pick = held_pick_block_pos + held_handle
    world.move_joints(
        world.ik([held_pick[0], held_pick[1], max(held_pick[2] + lift, tip[2] + 0.03)], quat)[0],
        seg,
    )
    world.move_joints(world.ik(held_pick, quat)[0], seg)
    world.release()
    world.set_gripper(True)
    world.move_joints(world.ik(held_pick + back + [0.0, 0.0, 0.05], quat)[0], seg * 0.7)

    displaced_pos, _ = world.block_pose(displaced_pid)
    displaced_handle = sim.handle_point(block_types[displaced_pid])
    displaced_pick = displaced_pos + displaced_handle
    displaced_target = targets[displaced_pid]
    displaced_place = displaced_target + displaced_handle
    displaced_pitch = next(
        (
            angle
            for angle in sim.GRASP_PITCH_DEG
            if world.ik(displaced_pick, sim.pitch_quat(angle))[1] < 0.01
            and world.ik(displaced_place, sim.pitch_quat(angle))[1] < 0.01
        ),
        None,
    )
    if displaced_pitch is None:
        monitor.finish_recovery(
            displaced_pid, displaced_target, outcome="unreachable"
        )
        return
    displaced_quat = sim.pitch_quat(displaced_pitch)
    displaced_back = sim.rot(displaced_quat, [-0.10, 0.0, 0.0])
    world.set_gripper(True)
    for pos in (
        displaced_pick + displaced_back + [0.0, 0.0, 0.05],
        displaced_pick + displaced_back,
        displaced_pick,
    ):
        world.move_joints(world.ik(pos, displaced_quat)[0], seg)
    world.set_gripper(False)
    world.grasp(displaced_pid)
    for pos in (
        displaced_pick + [0.0, 0.0, lift],
        displaced_place + [0.0, 0.0, 0.04],
        displaced_place + [0.0, 0.0, clear],
    ):
        world.move_joints(world.ik(pos, displaced_quat)[0], seg)
    desired = displaced_target + [0.0, 0.0, clear]
    commanded_tip, _ = world.tip_pose()
    _closed_loop_correction(
        world,
        displaced_pid,
        desired,
        commanded_tip,
        displaced_quat,
        passes,
        correction_tol,
        max_step,
        correction_seg,
        correction_dwell,
    )
    world.release()
    monitor.release_poses[displaced_pid] = world.block_pose(displaced_pid)
    world.set_gripper(True)
    for pos in (
        displaced_place + displaced_back,
        displaced_place + displaced_back + [0.0, 0.0, 0.1],
    ):
        world.move_joints(world.ik(pos, displaced_quat)[0], seg * 0.7)
    monitor.finish_recovery(displaced_pid, displaced_target)


@stage("build", "closed_loop_weld_place")
def build_closed_loop_weld_place(world, spec, order, pairs, cfg, log):
    """Run v0's scripted motion with measured pre-release position correction."""
    prm = cfg.get("params", {})
    seg = prm.get("segment_s", 1.0)
    clear = prm.get("place_clearance_m", 0.005)
    lift = prm.get("carry_height_m", 0.15)
    passes = int(prm.get("place_correction_passes", 2))
    correction_seg = prm.get("place_correction_segment_s", 0.6)
    correction_dwell = prm.get("place_correction_dwell_s", 0.2)
    correction_tol = prm.get("place_correction_tolerance_m", 0.001)
    max_step = prm.get("place_correction_max_step_m", 0.04)
    aware = bool(prm.get("placed_block_monitor", False))
    contact_band = float(prm.get("placed_block_contact_band_m", 0.02))
    slow_factor = float(prm.get("placed_block_descent_slow_factor", 4.0))

    by_id = {block["id"]: block for block in spec}
    targets = {
        pairs[sid]: sim.SITE + np.asarray(block["pos"], dtype=float)
        for sid, block in by_id.items()
        if sid in pairs
    }
    block_types = {
        pairs[sid]: block["type"] for sid, block in by_id.items() if sid in pairs
    }
    monitor = PlacedBlockMonitor(world, spec, pairs, cfg, log)
    top = 0.0
    for sid in order:
        if sid not in pairs:
            continue
        block, pid = by_id[sid], pairs[sid]
        handle = sim.handle_point(block["type"])
        block_pos, _ = world.block_pose(pid)
        target = sim.SITE + np.asarray(block["pos"])
        pick = block_pos + handle
        place = target + handle
        pitch = next(
            (
                angle
                for angle in sim.GRASP_PITCH_DEG
                if world.ik(pick, sim.pitch_quat(angle))[1] < 0.01
                and world.ik(place, sim.pitch_quat(angle))[1] < 0.01
            ),
            None,
        )
        if pitch is None:
            log["failures"].append("unreachable")
            continue

        quat = sim.pitch_quat(pitch)
        back = sim.rot(quat, [-0.10, 0, 0])
        carry_z = max(top, target[2]) + lift
        pick_block_pos = block_pos.copy()
        monitor.set_active(pid)
        while True:
            block_pos, _ = world.block_pose(pid)
            pick = block_pos + handle
            world.set_gripper(True)
            for pos in (pick + back + [0, 0, 0.05], pick + back, pick):
                world.move_joints(world.ik(pos, quat)[0], seg)
            world.set_gripper(False)
            world.grasp(pid)
            for pos in (
                pick + [0, 0, lift],
                [place[0], place[1], carry_z],
                place + [0, 0, 0.04],
            ):
                world.move_joints(world.ik(pos, quat)[0], seg)

            shift_xy = monitor.support_shift(sid) if aware else np.zeros(2)
            aware_place = place.copy()
            aware_place[:2] += shift_xy
            try:
                monitor.set_watching(True)
                if aware:
                    world.move_joints(
                        world.ik(
                            aware_place + [0.0, 0.0, contact_band], quat
                        )[0],
                        seg,
                    )
                    world.move_joints(
                        world.ik(aware_place + [0.0, 0.0, clear], quat)[0],
                        seg * slow_factor,
                    )
                else:
                    world.move_joints(
                        world.ik(place + [0.0, 0.0, clear], quat)[0], seg
                    )

                desired_pre_release = target + [
                    shift_xy[0],
                    shift_xy[1],
                    clear,
                ]
                initial_pos, _ = world.block_pose(pid)
                commanded_tip, _ = world.tip_pose()
                commanded_tip, corrections = _closed_loop_correction(
                    world,
                    pid,
                    desired_pre_release,
                    commanded_tip,
                    quat,
                    passes,
                    correction_tol,
                    max_step,
                    correction_seg,
                    correction_dwell,
                )
                monitor.set_watching(False)
                break
            except DisplacementDetected as event:
                _recover_displaced_block(
                    world,
                    monitor,
                    event.block_id,
                    pick_block_pos,
                    block["type"],
                    quat,
                    back,
                    seg,
                    lift,
                    targets,
                    block_types,
                    passes,
                    correction_tol,
                    max_step,
                    correction_seg,
                    correction_dwell,
                    clear,
                )

        pre_release_pos, _ = world.block_pose(pid)
        world.release()
        monitor.record_release(pid)
        world.set_gripper(True)
        retreat_place = aware_place if aware else place
        for pos in (
            retreat_place + back,
            retreat_place + back + [0, 0, 0.1],
        ):
            world.move_joints(world.ik(pos, quat)[0], seg * 0.7)
        log["stages"].append(
            {
                "block": pid,
                "pitch": pitch,
                "initial_pre_release_err_m": float(
                    np.linalg.norm(initial_pos - target)
                ),
                "pre_release_err_m": float(
                    np.linalg.norm(pre_release_pos - target)
                ),
                "correction_passes": corrections,
            }
        )
        top = max(
            top,
            target[2] + sim.BLOCK_TYPES[block["type"]][2] / 2,
        )
    monitor.close()
