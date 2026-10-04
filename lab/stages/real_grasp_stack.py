"""Stack-aware contact grasp for rung 3.

This is deliberately a separate build implementation so the evaluated
``v3_grasp`` control path remains unchanged.  It uses the simulator's true
block pose for pick localization and optional pre-release pose correction.
"""

import numpy as np

from lab import sim
from lab.pipeline import stage
from lab.stages.real_grasp import _block_contact_state


def _ramp_gripper(world, target, seconds):
    actuator = world._act("arm_f1x")
    start = float(world.d.ctrl[actuator])
    steps = max(1, int(seconds / world.m.opt.timestep))
    for i in range(steps):
        phase = (i + 1) / steps
        blend = 3 * phase**2 - 2 * phase**3
        world.d.ctrl[actuator] = start + blend * (target - start)
        world.step(1)


@stage("build", "scripted_real_grasp_stack")
def build_scripted_real_grasp_stack(world, spec, order, pairs, cfg, log):
    """Physical handle grasp with opt-in stack placement and release fixes."""
    prm = cfg.get("params", {})
    seg = prm.get("segment_s", 1.4)
    clear = prm.get("place_clearance_m", 0.008)
    lift = prm.get("carry_height_m", 0.15)
    grasp_offset = np.asarray(prm.get("grasp_tip_offset_m", [0.02, 0.0, 0.01]))
    handle_friction = np.asarray(prm.get("handle_friction", [1.0, 0.01, 0.001]))
    pose_correction = bool(prm.get("oracle_place_correction", False))
    ramped_release = bool(prm.get("ramped_release", False))
    vertical_clear = bool(prm.get("vertical_clearance_before_retreat", False))
    by_id = {b["id"]: b for b in spec}
    top = 0.0

    for sid in order:
        if sid not in pairs:
            continue
        block, pid = by_id[sid], pairs[sid]
        hp = sim.handle_point(block["type"])
        block_pos, block_quat = world.block_pose(pid)
        body_id = world.m.body(pid).id
        handle_geom = world.m.body_geomadr[body_id] + world.m.body_geomnum[body_id] - 1
        world.m.geom_friction[handle_geom] = handle_friction
        target = sim.SITE + np.asarray(block["pos"])
        pick = block_pos + sim.rot(block_quat, hp) + grasp_offset
        place = target + hp + grasp_offset
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
        start_pos = block_pos.copy()
        world.set_gripper(True)
        for pos in (pick + back + [0, 0, 0.05], pick + back, pick):
            world.move_joints(world.ik(pos, quat)[0], seg)

        before_close, _ = world.block_pose(pid)
        approach_displacement = float(np.linalg.norm(before_close - start_pos))
        _ramp_gripper(
            world,
            prm.get("grasp_close_command", -0.5),
            prm.get("grasp_close_dwell_s", 0.8),
        )
        handle_contacts, body_contacts = _block_contact_state(world, pid)
        closed_pos, _ = world.block_pose(pid)
        closed_rel = closed_pos - world.tip_pose()[0]

        transport_samples = []
        for waypoint, pos in (
            ("lift", pick + [0, 0, lift]),
            ("carry", [place[0], place[1], carry_z]),
            ("preplace", place + [0, 0, 0.04]),
            ("place", place + [0, 0, clear]),
        ):
            world.move_joints(world.ik(pos, quat)[0], seg)
            sample_pos, _ = world.block_pose(pid)
            transport_samples.append(
                {
                    "waypoint": waypoint,
                    "pos": sample_pos,
                    "relative_drift_m": float(
                        np.linalg.norm((sample_pos - world.tip_pose()[0]) - closed_rel)
                    ),
                }
            )

        if pose_correction:
            for _ in range(int(prm.get("place_correction_passes", 2))):
                actual, _ = world.block_pose(pid)
                desired = target + [0.0, 0.0, clear]
                correction = desired - actual
                tip, _ = world.tip_pose()
                world.move_joints(
                    world.ik(tip + correction, quat)[0],
                    prm.get("place_correction_segment_s", 0.8),
                    dwell=prm.get("place_correction_dwell_s", 0.15),
                )

        pre_release_pos, pre_release_quat = world.block_pose(pid)
        pre_release_tip, _ = world.tip_pose()
        relative_at_release = pre_release_pos - pre_release_tip
        slip_vector = relative_at_release - closed_rel
        pre_release_error = pre_release_pos - target
        # At the target pose the block bottom coincides with its support top.
        support_gap = float(pre_release_error[2])
        contacts_before_open = _block_contact_state(world, pid)

        if ramped_release:
            _ramp_gripper(
                world,
                prm.get("grasp_open_command", -1.0),
                prm.get("release_open_s", 0.6),
            )
        else:
            world.set_gripper(True)
        world.step(max(1, int(prm.get("release_dwell_s", 0.5) / world.m.opt.timestep)))
        after_open_pos, _ = world.block_pose(pid)
        contacts_after_open = _block_contact_state(world, pid)

        if vertical_clear:
            tip, _ = world.tip_pose()
            clear_vector = np.asarray(
                prm.get(
                    "release_clear_vector_m",
                    [0.0, 0.0, prm.get("release_vertical_clearance_m", 0.10)],
                )
            )
            world.move_joints(
                world.ik(tip + clear_vector, quat)[0],
                prm.get("release_clear_segment_s", 0.7),
                dwell=0.0,
            )
            after_clear_pos, _ = world.block_pose(pid)
            retreat_points = (
                (place + back + [0.0, 0.0, 0.10],)
                if prm.get("release_final_retreat", True)
                else ()
            )
        else:
            after_clear_pos = after_open_pos
            retreat_points = (place + back, place + back + [0, 0, 0.1])
        for pos in retreat_points:
            world.move_joints(world.ik(pos, quat)[0], seg * 0.7)
        after_retreat_pos, _ = world.block_pose(pid)

        max_relative_drift = max(
            (sample["relative_drift_m"] for sample in transport_samples), default=0.0
        )
        log["stages"].append(
            {
                "block": pid,
                "pitch": pitch,
                "grasp": "contact",
                "handle_contacts_after_close": handle_contacts,
                "block_contacts_after_close": body_contacts,
                "approach_displacement_m": approach_displacement,
                "max_relative_drift_m": max_relative_drift,
                "slip_vector_at_release_m": slip_vector.tolist(),
                "pre_release_error_xyz_m": pre_release_error.tolist(),
                "support_gap_m": support_gap,
                "release_open_displacement_xyz_m": (
                    after_open_pos - pre_release_pos
                ).tolist(),
                "release_clear_displacement_xyz_m": (
                    after_clear_pos - after_open_pos
                ).tolist(),
                "retreat_displacement_xyz_m": (
                    after_retreat_pos - after_clear_pos
                ).tolist(),
                "contacts_before_open": list(contacts_before_open),
                "contacts_after_open": list(contacts_after_open),
                "fixes": {
                    "oracle_place_correction": pose_correction,
                    "ramped_release": ramped_release,
                    "vertical_clearance_before_retreat": vertical_clear,
                },
            }
        )
        top = max(top, target[2] + sim.BLOCK_TYPES[block["type"]][2] / 2)
