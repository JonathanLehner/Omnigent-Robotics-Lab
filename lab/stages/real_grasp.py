"""Contact-based handle grasp for rung 3.

Unlike ``scripted_weld``, this stage never calls ``World.grasp``.  The block is
supported solely by MuJoCo contacts and the commanded gripper force.
"""

import mujoco
import numpy as np

from lab import sim
from lab.pipeline import stage


def _block_contact_state(world, block_id):
    """Return gripper contacts on the handle and on the block body."""
    body_id = world.m.body(block_id).id
    first_geom = world.m.body_geomadr[body_id]
    handle_geom = first_geom + world.m.body_geomnum[body_id] - 1
    block_geoms = set(range(first_geom, handle_geom))
    gripper_bodies = {
        world.m.body("arm_link_wr1").id,
        world.m.body("arm_link_fngr").id,
    }
    handle_contacts = body_contacts = 0
    for i in range(world.d.ncon):
        contact = world.d.contact[i]
        for block_geom, other_geom in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            if world.m.geom_bodyid[other_geom] not in gripper_bodies:
                continue
            handle_contacts += int(block_geom == handle_geom)
            body_contacts += int(block_geom in block_geoms)
    return handle_contacts, body_contacts


def _block_gripper_contact_forces(world, block_id):
    """Count block/gripper contacts and sum their instantaneous force magnitudes."""
    body_id = world.m.body(block_id).id
    first_geom = world.m.body_geomadr[body_id]
    handle_geom = first_geom + world.m.body_geomnum[body_id] - 1
    block_geoms = set(range(first_geom, handle_geom))
    gripper_bodies = {
        world.m.body("arm_link_wr1").id,
        world.m.body("arm_link_fngr").id,
    }
    sample = {
        "handle_contacts": 0,
        "body_contacts": 0,
        "handle_force_n": 0.0,
        "body_force_n": 0.0,
    }
    contact_force = np.zeros(6)
    for i in range(world.d.ncon):
        contact = world.d.contact[i]
        for block_geom, other_geom in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            if world.m.geom_bodyid[other_geom] not in gripper_bodies:
                continue
            mujoco.mj_contactForce(world.m, world.d, i, contact_force)
            force = float(np.linalg.norm(contact_force[:3]))
            if block_geom == handle_geom:
                sample["handle_contacts"] += 1
                sample["handle_force_n"] += force
            elif block_geom in block_geoms:
                sample["body_contacts"] += 1
                sample["body_force_n"] += force
    return sample


@stage("build", "scripted_real_grasp")
def build_scripted_real_grasp(world, spec, order, pairs, cfg, log):
    """Use scripted IK motion but close the physical fingers on each handle."""
    prm = cfg.get("params", {})
    seg = prm.get("segment_s", 1.4)
    clear = prm.get("place_clearance_m", 0.008)
    lift = prm.get("carry_height_m", 0.15)
    close_dwell = prm.get("grasp_close_dwell_s", 0.8)
    release_vertical_lift = float(prm.get("release_vertical_lift_m", 0.0))
    release_vertical_lift_steps = int(prm.get("release_vertical_lift_steps", 12))
    release_phase_settle = float(prm.get("release_phase_settle_s", 0.0))
    release_seat_lower = float(prm.get("release_seat_lower_m", 0.0))
    release_open_s = float(prm.get("release_open_s", 0.0))
    release_open_retreat = float(prm.get("release_open_retreat_m", 0.0))
    grasp_open_command = float(prm.get("grasp_open_command", -1.0))
    release_place_bias = np.asarray(
        prm.get("release_place_bias_m", [0.0, 0.0, 0.0])
    )
    grasp_offset = np.asarray(prm.get("grasp_tip_offset_m", [0.02, 0.0, 0.01]))
    handle_friction = np.asarray(prm.get("handle_friction", [1.0, 0.01, 0.001]))
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
        # The Menagerie Spot fingertip frame is slightly behind/below the
        # physical pinch surfaces; this calibrated offset centers the handle
        # between those surfaces instead of driving a jaw into the block.
        pick = block_pos + sim.rot(block_quat, hp) + grasp_offset
        place = target + hp + grasp_offset + release_place_bias
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
        approach_displacement_m = float(np.linalg.norm(before_close - start_pos))
        # Closing all the way in one control tick injects enough impulse to
        # launch the light block.  Ramp to a force-producing pinch pose.
        gripper_act = world._act("arm_f1x")
        start_command = float(world.d.ctrl[gripper_act])
        close_command = prm.get("grasp_close_command", -0.5)
        close_steps = max(1, int(close_dwell / world.m.opt.timestep))
        for i in range(close_steps):
            phase = (i + 1) / close_steps
            blend = 3 * phase**2 - 2 * phase**3
            world.d.ctrl[gripper_act] = (
                start_command + blend * (close_command - start_command)
            )
            world.step(1)
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
                    "z": float(sample_pos[2]),
                    "relative_drift_m": float(
                        np.linalg.norm((sample_pos - world.tip_pose()[0]) - closed_rel)
                    ),
                }
            )

        pre_release = world.block_pose(pid)
        pre_release_pos = pre_release[0]
        transported_m = float(np.linalg.norm(pre_release_pos - before_close))
        max_relative_drift_m = max(
            (sample["relative_drift_m"] for sample in transport_samples), default=0.0
        )
        dropped = (
            transport_samples
            and transport_samples[0]["z"] > prm.get("lifted_z_m", 0.09)
            and transport_samples[1]["z"] < prm.get("dropped_z_m", 0.075)
        )
        handle_collision = approach_displacement_m > prm.get(
            "handle_collision_displacement_m", 0.02
        )
        failure_mode = None
        if transported_m < prm.get("min_transport_m", 0.08):
            failure_mode = "drop" if dropped else "slip"
        elif max_relative_drift_m > prm.get("max_relative_slip_m", 0.04):
            failure_mode = "drop" if dropped else "slip"
        elif handle_collision and not handle_contacts:
            failure_mode = "handle_collision"
        if failure_mode and (
            transported_m < prm.get("min_transport_m", 0.08)
            or np.linalg.norm(pre_release_pos - target)
            > prm.get("max_pre_release_error_m", 0.05)
        ):
            log["failures"].append(failure_mode)
        tip_before_seat, _ = world.tip_pose()
        seat_steps = max(1, int(prm.get("release_seat_steps", 4)))
        for step in range(1, seat_steps + 1):
            if release_seat_lower <= 0.0:
                break
            seat_waypoint = tip_before_seat + [
                0.0,
                0.0,
                -release_seat_lower * step / seat_steps,
            ]
            world.move_joints(
                world.ik(seat_waypoint, quat)[0],
                prm.get("release_seat_segment_s", 0.4) / seat_steps,
                dwell=0.0,
            )
        after_seat_pos, _ = world.block_pose(pid)

        gripper_act = world._act("arm_f1x")
        gripper_joint = world.m.joint("arm_f1x")
        gripper_qpos = gripper_joint.qposadr[0]
        gripper_dof = gripper_joint.dofadr[0]
        open_start_command = float(world.d.ctrl[gripper_act])
        open_start_qpos = float(world.d.qpos[gripper_qpos])
        open_start_tip, _ = world.tip_pose()
        arm_actuators = [world._act(joint) for joint in sim.ARM_JOINTS]
        open_start_arm = world.d.ctrl[arm_actuators].copy()
        open_target_arm = open_start_arm
        if release_open_retreat > 0.0:
            retreat_xy = np.asarray([back[0], back[1], 0.0])
            retreat_xy /= np.linalg.norm(retreat_xy)
            open_target_tip = open_start_tip + release_open_retreat * retreat_xy
            open_target_arm = world.ik(open_target_tip, quat)[0]
        open_steps = max(1, int(release_open_s / world.m.opt.timestep))
        open_samples = []
        max_open_speed = 0.0
        for step in range(open_steps):
            if release_open_s > 0.0:
                phase = (step + 1) / open_steps
                blend = 3 * phase**2 - 2 * phase**3
                world.d.ctrl[gripper_act] = (
                    open_start_command
                    + blend * (grasp_open_command - open_start_command)
                )
            else:
                blend = 1.0
                world.d.ctrl[gripper_act] = grasp_open_command
            if release_open_retreat > 0.0:
                world.d.ctrl[arm_actuators] = (
                    open_start_arm + blend * (open_target_arm - open_start_arm)
                )
            world.step(1)
            open_samples.append(_block_gripper_contact_forces(world, pid))
            max_open_speed = max(
                max_open_speed, abs(float(world.d.qvel[gripper_dof]))
            )
        dwell_steps = max(
            0, int(prm.get("release_dwell_s", 0.5) / world.m.opt.timestep)
        )
        for _ in range(dwell_steps):
            world.step(1)
            open_samples.append(_block_gripper_contact_forces(world, pid))
            max_open_speed = max(
                max_open_speed, abs(float(world.d.qvel[gripper_dof]))
            )
        open_end_qpos = float(world.d.qpos[gripper_qpos])
        open_end_tip, _ = world.tip_pose()
        after_open_pos, _ = world.block_pose(pid)
        tip_after_open, _ = world.tip_pose()
        lift_tip_samples = [tip_after_open]

        if release_vertical_lift > 0.0:
            # Lift in world z before introducing any horizontal motion.  The
            # legacy retreat below follows the pitched tool axis; keeping it
            # as the zero-default path preserves prior method results.
            # World.move_joints interpolates in joint space, so use short
            # Cartesian waypoints to keep the tool path vertical as well as
            # making its endpoint vertical.
            lift_steps = max(1, release_vertical_lift_steps)
            for step in range(1, lift_steps + 1):
                lift_waypoint = tip_after_open + [
                    0.0,
                    0.0,
                    release_vertical_lift * step / lift_steps,
                ]
                world.move_joints(
                    world.ik(lift_waypoint, quat)[0],
                    seg * 0.7 / lift_steps,
                    dwell=0.0,
                )
                lift_tip_samples.append(world.tip_pose()[0])
            after_lift_pos, _ = world.block_pose(pid)

            # Once the open jaws are vertically clear, retreat horizontally
            # by the x/y component of the calibrated tool-axis offset.
            tip_after_lift, _ = world.tip_pose()
            retreat = tip_after_lift + [back[0], back[1], 0.0]
            world.move_joints(world.ik(retreat, quat)[0], seg * 0.7)
        else:
            after_lift_pos = after_open_pos
            for pos in (place + back, place + back + [0, 0, 0.1]):
                world.move_joints(world.ik(pos, quat)[0], seg * 0.7)
        after_retreat_pos, _ = world.block_pose(pid)

        if release_phase_settle > 0.0:
            world.step(max(1, int(release_phase_settle / world.m.opt.timestep)))
        after_settle_pos, _ = world.block_pose(pid)
        tip_lift_displacement = lift_tip_samples[-1] - lift_tip_samples[0]
        lift_lateral_deviation = max(
            (
                float(np.linalg.norm(sample[:2] - lift_tip_samples[0][:2]))
                for sample in lift_tip_samples
            ),
            default=0.0,
        )

        phase_displacements = {
            "pre_release_from_target": pre_release_pos - target,
            "after_seat_from_pre_release": after_seat_pos - pre_release_pos,
            "after_open_from_pre_release": after_open_pos - pre_release_pos,
            "after_lift_from_after_open": after_lift_pos - after_open_pos,
            "after_retreat_from_after_lift": after_retreat_pos - after_lift_pos,
            "after_settle_from_after_retreat": after_settle_pos - after_retreat_pos,
        }

        log["stages"].append(
            {
                "block": pid,
                "pitch": pitch,
                "grasp": "contact",
                "handle_contacts_after_close": handle_contacts,
                "block_contacts_after_close": body_contacts,
                "approach_displacement_m": approach_displacement_m,
                "transported_m": transported_m,
                "max_relative_drift_m": max_relative_drift_m,
                "grasp_failure_mode": failure_mode,
                "pre_release_err_m": float(np.linalg.norm(pre_release_pos - target)),
                "release_vertical_lift_m": release_vertical_lift,
                "release_tip_lift_displacement_xyz_m": (
                    tip_lift_displacement.tolist()
                ),
                "release_tip_lift_max_lateral_deviation_m": lift_lateral_deviation,
                "release_open_diagnostics": {
                    "command_start": open_start_command,
                    "command_end": grasp_open_command,
                    "joint_position_start": open_start_qpos,
                    "joint_position_end": open_end_qpos,
                    "max_joint_speed_rad_s": max_open_speed,
                    "tip_displacement_xyz_m": (
                        open_end_tip - open_start_tip
                    ).tolist(),
                    "max_handle_contacts": max(
                        (sample["handle_contacts"] for sample in open_samples),
                        default=0,
                    ),
                    "max_body_contacts": max(
                        (sample["body_contacts"] for sample in open_samples),
                        default=0,
                    ),
                    "handle_contact_impulse_ns": float(
                        world.m.opt.timestep
                        * sum(sample["handle_force_n"] for sample in open_samples)
                    ),
                    "body_contact_impulse_ns": float(
                        world.m.opt.timestep
                        * sum(sample["body_force_n"] for sample in open_samples)
                    ),
                    "max_handle_force_n": max(
                        (sample["handle_force_n"] for sample in open_samples),
                        default=0.0,
                    ),
                    "max_body_force_n": max(
                        (sample["body_force_n"] for sample in open_samples),
                        default=0.0,
                    ),
                },
                "release_phase_displacement_xyz_m": {
                    phase: displacement.tolist()
                    for phase, displacement in phase_displacements.items()
                },
                "release_phase_displacement_m": {
                    phase: float(np.linalg.norm(displacement))
                    for phase, displacement in phase_displacements.items()
                },
            }
        )
        top = max(top, target[2] + sim.BLOCK_TYPES[block["type"]][2] / 2)
