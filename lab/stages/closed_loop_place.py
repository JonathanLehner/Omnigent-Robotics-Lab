"""Closed-loop weld placement for rung 4.

This keeps the rung-0 pick, carry, weld grasp, and release sequence, but
measures the block pose at the nominal place waypoint and corrects the
translation before release.
"""

import numpy as np

from lab import sim
from lab.pipeline import stage


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

    by_id = {block["id"]: block for block in spec}
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
        world.set_gripper(True)
        for pos in (pick + back + [0, 0, 0.05], pick + back, pick):
            world.move_joints(world.ik(pos, quat)[0], seg)
        world.set_gripper(False)
        world.grasp(pid)
        for pos in (
            pick + [0, 0, lift],
            [place[0], place[1], carry_z],
            place + [0, 0, 0.04],
            place + [0, 0, clear],
        ):
            world.move_joints(world.ik(pos, quat)[0], seg)

        desired_pre_release = target + [0.0, 0.0, clear]
        initial_pos, _ = world.block_pose(pid)
        commanded_tip, _ = world.tip_pose()
        corrections = []
        for _ in range(passes):
            actual_pos, _ = world.block_pose(pid)
            residual = desired_pre_release - actual_pos
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= correction_tol:
                break
            correction = residual
            if residual_norm > max_step:
                correction = correction * (max_step / residual_norm)
            # Integrate measured block error into the command. Recomputing from
            # actual_tip + residual would send the same nominal target on every
            # pass and cannot remove the position servos' steady-state sag.
            commanded_tip = commanded_tip + correction
            q_target, ik_residual, _ = world.ik(commanded_tip, quat)
            world.move_joints(
                q_target,
                correction_seg,
                dwell=correction_dwell,
            )
            corrected_pos, _ = world.block_pose(pid)
            corrections.append(
                {
                    "command_m": correction.tolist(),
                    "ik_residual_m": ik_residual,
                    "remaining_err_m": float(
                        np.linalg.norm(corrected_pos - desired_pre_release)
                    ),
                }
            )

        pre_release_pos, _ = world.block_pose(pid)
        world.release()
        world.set_gripper(True)
        for pos in (place + back, place + back + [0, 0, 0.1]):
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
