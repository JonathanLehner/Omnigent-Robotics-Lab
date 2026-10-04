"""Opt-in grasp release with gravity-compensated hold and Cartesian lift."""

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
    for index in range(world.d.ncon):
        contact = world.d.contact[index]
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
    """Count block/gripper contacts and sum instantaneous force magnitudes."""
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
    for index in range(world.d.ncon):
        contact = world.d.contact[index]
        for block_geom, other_geom in (
            (contact.geom1, contact.geom2),
            (contact.geom2, contact.geom1),
        ):
            if world.m.geom_bodyid[other_geom] not in gripper_bodies:
                continue
            mujoco.mj_contactForce(
                world.m, world.d, index, contact_force
            )
            force = float(np.linalg.norm(contact_force[:3]))
            if block_geom == handle_geom:
                sample["handle_contacts"] += 1
                sample["handle_force_n"] += force
            elif block_geom in block_geoms:
                sample["body_contacts"] += 1
                sample["body_force_n"] += force
    return sample


def _smoothstep(phase):
    return 3 * phase**2 - 2 * phase**3


def _close_command(world, actuator, configured):
    """Resolve the flagged maximum-grip command against the model ctrlrange."""
    maximum = float(world.m.actuator_ctrlrange[actuator, 1])
    if configured == "max":
        return maximum
    command = float(configured)
    if not np.isclose(command, maximum):
        raise ValueError(
            "v3_grasp_hold requires grasp_close_command=max "
            f"(model maximum is {maximum}, got {configured!r})"
        )
    return maximum


def _hold_step(world, arm_actuators, arm_dofs, joint_target):
    """Hold the captured joints and cancel their current MuJoCo bias forces."""
    world.d.ctrl[arm_actuators] = joint_target
    world.d.qfrc_applied[arm_dofs] = world.d.qfrc_bias[arm_dofs]
    compensation_norm = float(np.linalg.norm(world.d.qfrc_bias[arm_dofs]))
    world.step(1)
    return compensation_norm


@stage("build", "scripted_real_grasp_hold")
def build_scripted_real_grasp_hold(world, spec, order, pairs, cfg, log):
    """Use a maximum-force pinch and a held, gradual, vertical release."""
    prm = cfg.get("params", {})
    seg = float(prm.get("segment_s", 1.4))
    clear = float(prm.get("place_clearance_m", 0.008))
    lift = float(prm.get("carry_height_m", 0.15))
    close_dwell = float(prm.get("grasp_close_dwell_s", 0.8))
    release_open_s = float(prm.get("release_open_s", 1.5))
    release_vertical_lift = float(prm.get("release_vertical_lift_m", 0.15))
    release_lift_s = float(prm.get("release_lift_s", seg * 0.7))
    release_xy_hold_gain = float(prm.get("release_xy_hold_gain", 1.0))
    grasp_open_command = float(prm.get("grasp_open_command", -1.0))
    grasp_offset = np.asarray(
        prm.get("grasp_tip_offset_m", [0.02, 0.0, 0.01]), dtype=float
    )
    handle_friction = np.asarray(
        prm.get("handle_friction", [1.0, 0.01, 0.001]), dtype=float
    )
    release_place_bias = np.asarray(
        prm.get("release_place_bias_m", [0.0, 0.0, 0.0]), dtype=float
    )
    by_id = {block["id"]: block for block in spec}
    top = 0.0

    for sid in order:
        if sid not in pairs:
            continue
        block, pid = by_id[sid], pairs[sid]
        handle_point = sim.handle_point(block["type"])
        block_pos, block_quat = world.block_pose(pid)
        body_id = world.m.body(pid).id
        handle_geom = (
            world.m.body_geomadr[body_id] + world.m.body_geomnum[body_id] - 1
        )
        world.m.geom_friction[handle_geom] = handle_friction
        target = sim.SITE + np.asarray(block["pos"])
        pick = (
            block_pos
            + sim.rot(block_quat, handle_point)
            + grasp_offset
        )
        place = target + handle_point + grasp_offset + release_place_bias
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
        back = sim.rot(quat, [-0.10, 0.0, 0.0])
        carry_z = max(top, target[2]) + lift
        start_pos = block_pos.copy()
        world.set_gripper(True)
        for pos in (pick + back + [0, 0, 0.05], pick + back, pick):
            world.move_joints(world.ik(pos, quat)[0], seg)

        before_close, _ = world.block_pose(pid)
        approach_displacement_m = float(np.linalg.norm(before_close - start_pos))
        gripper_act = world._act("arm_f1x")
        start_command = float(world.d.ctrl[gripper_act])
        close_command = _close_command(
            world, gripper_act, prm.get("grasp_close_command", "max")
        )
        close_steps = max(1, int(close_dwell / world.m.opt.timestep))
        for step in range(close_steps):
            blend = _smoothstep((step + 1) / close_steps)
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
                        np.linalg.norm(
                            (sample_pos - world.tip_pose()[0]) - closed_rel
                        )
                    ),
                }
            )

        pre_release_pos, _ = world.block_pose(pid)
        transported_m = float(np.linalg.norm(pre_release_pos - before_close))
        carry_slip_m = max(
            (
                sample["relative_drift_m"]
                for sample in transport_samples
            ),
            default=0.0,
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
        if (
            transported_m < prm.get("min_transport_m", 0.08)
            or carry_slip_m > prm.get("max_relative_slip_m", 0.04)
        ):
            failure_mode = "drop" if dropped else "slip"
        elif handle_collision and not handle_contacts:
            failure_mode = "handle_collision"
        if failure_mode and (
            transported_m < prm.get("min_transport_m", 0.08)
            or np.linalg.norm(pre_release_pos - target)
            > prm.get("max_pre_release_error_m", 0.05)
        ):
            log["failures"].append(failure_mode)

        gripper_joint = world.m.joint("arm_f1x")
        gripper_qpos = gripper_joint.qposadr[0]
        gripper_dof = gripper_joint.dofadr[0]
        open_start_command = float(world.d.ctrl[gripper_act])
        open_start_qpos = float(world.d.qpos[gripper_qpos])
        open_start_tip, open_start_quat = world.tip_pose()
        open_start_block, _ = world.block_pose(pid)
        arm_actuators = np.asarray(
            [world._act(joint) for joint in sim.ARM_JOINTS], dtype=int
        )
        arm_dofs = np.asarray(world.arm_v, dtype=int)
        # Capture the realized placement joints, not their old command. With
        # qfrc_bias feed-forward, these are an equilibrium target rather than
        # an error-dependent approximation to the captured hand pose.
        hold_joint_target = world.d.qpos[world.arm_q].copy()
        open_steps = max(1, int(release_open_s / world.m.opt.timestep))
        open_samples = []
        open_tip_samples = [open_start_tip]
        max_open_speed = 0.0
        max_compensation_norm = 0.0
        for step in range(open_steps):
            blend = _smoothstep((step + 1) / open_steps)
            world.d.ctrl[gripper_act] = (
                open_start_command
                + blend * (grasp_open_command - open_start_command)
            )
            max_compensation_norm = max(
                max_compensation_norm,
                _hold_step(
                    world,
                    arm_actuators,
                    arm_dofs,
                    hold_joint_target,
                ),
            )
            open_tip_samples.append(world.tip_pose()[0])
            open_samples.append(_block_gripper_contact_forces(world, pid))
            max_open_speed = max(
                max_open_speed, abs(float(world.d.qvel[gripper_dof]))
            )

        dwell_steps = max(
            0,
            int(
                float(prm.get("release_dwell_s", 0.5))
                / world.m.opt.timestep
            ),
        )
        for _ in range(dwell_steps):
            max_compensation_norm = max(
                max_compensation_norm,
                _hold_step(
                    world,
                    arm_actuators,
                    arm_dofs,
                    hold_joint_target,
                ),
            )
            open_tip_samples.append(world.tip_pose()[0])
            open_samples.append(_block_gripper_contact_forces(world, pid))
            max_open_speed = max(
                max_open_speed, abs(float(world.d.qvel[gripper_dof]))
            )
        world.d.qfrc_applied[arm_dofs] = 0.0

        open_end_qpos = float(world.d.qpos[gripper_qpos])
        open_end_tip, _ = world.tip_pose()
        after_open_pos, _ = world.block_pose(pid)
        hand_open_displacement = open_end_tip - open_start_tip
        hand_open_max_displacement_m = max(
            (
                float(np.linalg.norm(sample - open_start_tip))
                for sample in open_tip_samples
            ),
            default=0.0,
        )

        # Move the desired Cartesian pose upward while fixing x/y to the
        # measured lift-start position. Re-solving IK before every simulator
        # step makes the XY correction continuous at the physics rate.
        lift_start_tip, _ = world.tip_pose()
        lift_tip_samples = [lift_start_tip]
        lift_steps = max(1, int(release_lift_s / world.m.opt.timestep))
        for step in range(lift_steps):
            blend = _smoothstep((step + 1) / lift_steps)
            desired_tip = lift_start_tip + [
                0.0,
                0.0,
                release_vertical_lift * blend,
            ]
            current_tip, _ = world.tip_pose()
            lift_target = desired_tip.copy()
            lift_target[:2] += release_xy_hold_gain * (
                desired_tip[:2] - current_tip[:2]
            )
            world.d.ctrl[arm_actuators] = world.ik(
                lift_target, open_start_quat
            )[0]
            world.step(1)
            lift_tip_samples.append(world.tip_pose()[0])
        after_lift_pos, _ = world.block_pose(pid)
        lateral_drift_during_lift_m = max(
            (
                float(
                    np.linalg.norm(
                        sample[:2] - lift_start_tip[:2]
                    )
                )
                for sample in lift_tip_samples
            ),
            default=0.0,
        )
        tip_after_lift, _ = world.tip_pose()
        retreat = tip_after_lift + [back[0], back[1], 0.0]
        world.move_joints(world.ik(retreat, open_start_quat)[0], seg * 0.7)
        after_retreat_pos, _ = world.block_pose(pid)
        block_open_to_retreat = after_retreat_pos - open_start_block

        release_telemetry = {
            "hand_displacement_during_opening_xyz_m": (
                hand_open_displacement.tolist()
            ),
            "hand_displacement_during_opening_m": float(
                np.linalg.norm(hand_open_displacement)
            ),
            "hand_max_displacement_during_opening_m": (
                hand_open_max_displacement_m
            ),
            "lateral_drift_during_lift_m": (
                lateral_drift_during_lift_m
            ),
            "block_displacement_open_to_retreat_xyz_m": (
                block_open_to_retreat.tolist()
            ),
            "block_displacement_open_to_retreat_m": float(
                np.linalg.norm(block_open_to_retreat)
            ),
            "carry_slip_m": carry_slip_m,
        }
        log["stages"].append(
            {
                "block": pid,
                "pitch": pitch,
                "grasp": "contact",
                "handle_contacts_after_close": handle_contacts,
                "block_contacts_after_close": body_contacts,
                "grasp_close_command": close_command,
                "grasp_close_ctrlrange": (
                    world.m.actuator_ctrlrange[gripper_act].tolist()
                ),
                "approach_displacement_m": approach_displacement_m,
                "transported_m": transported_m,
                "max_relative_drift_m": carry_slip_m,
                "carry_slip_m": carry_slip_m,
                "grasp_failure_mode": failure_mode,
                "pre_release_err_m": float(
                    np.linalg.norm(pre_release_pos - target)
                ),
                "release_open_s": release_open_s,
                "release_vertical_lift_m": release_vertical_lift,
                "release_tip_lift_displacement_xyz_m": (
                    lift_tip_samples[-1] - lift_tip_samples[0]
                ).tolist(),
                "release_tip_lift_max_lateral_deviation_m": (
                    lateral_drift_during_lift_m
                ),
                "release_open_diagnostics": {
                    "command_start": open_start_command,
                    "command_end": grasp_open_command,
                    "joint_position_start": open_start_qpos,
                    "joint_position_end": open_end_qpos,
                    "max_joint_speed_rad_s": max_open_speed,
                    "tip_displacement_xyz_m": (
                        hand_open_displacement.tolist()
                    ),
                    "max_tip_displacement_m": (
                        hand_open_max_displacement_m
                    ),
                    "gravity_compensation": "qfrc_bias",
                    "max_gravity_compensation_norm": (
                        max_compensation_norm
                    ),
                    "max_handle_contacts": max(
                        (
                            sample["handle_contacts"]
                            for sample in open_samples
                        ),
                        default=0,
                    ),
                    "max_body_contacts": max(
                        (
                            sample["body_contacts"]
                            for sample in open_samples
                        ),
                        default=0,
                    ),
                    "handle_contact_impulse_ns": float(
                        world.m.opt.timestep
                        * sum(
                            sample["handle_force_n"]
                            for sample in open_samples
                        )
                    ),
                    "body_contact_impulse_ns": float(
                        world.m.opt.timestep
                        * sum(
                            sample["body_force_n"]
                            for sample in open_samples
                        )
                    ),
                    "max_handle_force_n": max(
                        (
                            sample["handle_force_n"]
                            for sample in open_samples
                        ),
                        default=0.0,
                    ),
                    "max_body_force_n": max(
                        (
                            sample["body_force_n"]
                            for sample in open_samples
                        ),
                        default=0.0,
                    ),
                },
                "release_telemetry": release_telemetry,
                "release_phase_displacement_xyz_m": {
                    "after_open_from_open_start": (
                        after_open_pos - open_start_block
                    ).tolist(),
                    "after_lift_from_after_open": (
                        after_lift_pos - after_open_pos
                    ).tolist(),
                    "after_retreat_from_after_lift": (
                        after_retreat_pos - after_lift_pos
                    ).tolist(),
                    "after_retreat_from_open_start": (
                        block_open_to_retreat.tolist()
                    ),
                },
            }
        )
        top = max(
            top,
            target[2] + sim.BLOCK_TYPES[block["type"]][2] / 2,
        )
