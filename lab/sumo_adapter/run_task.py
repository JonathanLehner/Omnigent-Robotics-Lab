"""Run Sumo MPC episodes headless (Relic whole-body policy in the loop). Executes INSIDE Sumo's pixi env:

  pixi run --manifest-path ~/src/sumo/pyproject.toml python lab/sumo_adapter/run_task.py --task spot_tire_stack

--task-module path/to/my_task.py registers agent-written tasks: the module must call
judo.tasks.register_task(...) (see sumo/tasks/__init__.py; reuse sumo.tasks._SPOT_REGISTRATION_KWARGS for Spot).
Upstream sumo.run_mpc builds its Controller without the task's registered rollout backend, so Spot tasks fall back
to plain MuJoCo and crash; this adapter passes it explicitly. Prints one JSON line with the results.
"""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from math import atan2
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _yaw_deg(quat: np.ndarray) -> float:
    """Return planar yaw in degrees from a wxyz quaternion."""
    quat = np.asarray(quat, dtype=float)
    quat /= np.linalg.norm(quat)
    return float(
        np.degrees(
            atan2(
                2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2),
            )
        )
    )


def base_arrival_state(task, qpos: np.ndarray, qvel: np.ndarray) -> dict:
    """Return the true terminal base pose, velocity, and approach errors."""
    config = getattr(task, "config", None)
    if not (
        hasattr(task, "body_pose_idx")
        and hasattr(config, "goal_x")
        and hasattr(config, "goal_y")
    ):
        return {}
    pose_idx = int(task.body_pose_idx)
    qpos = np.asarray(qpos)
    reset = np.asarray(task.reset_pose)
    terminal_pose = qpos[pose_idx : pose_idx + 7]
    goal_pose = reset[pose_idx : pose_idx + 7].copy()
    goal_pose[:2] = [config.goal_x, config.goal_y]
    goal_pose[3:7] = [1.0, 0.0, 0.0, 0.0]
    goal_xy = np.array([config.goal_x, config.goal_y])
    position_error = float(np.linalg.norm(qpos[pose_idx : pose_idx + 2] - goal_xy))
    quat = qpos[pose_idx + 3 : pose_idx + 7]
    vel_idx = int(task.model.jnt_dofadr[task.model.joint("base").id])
    base_speed = float(np.linalg.norm(qvel[vel_idx : vel_idx + 2]))
    # Spot's approach target has zero roll, pitch, and yaw. Report the full
    # quaternion angular distance, not only heading, so tilted arrivals show.
    orientation_error = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
    return {
        "terminal_base_pose": terminal_pose.tolist(),
        "terminal_base_xy": terminal_pose[:2].tolist(),
        "terminal_base_yaw_deg": round(_yaw_deg(quat), 3),
        "terminal_base_speed_m_s": round(base_speed, 4),
        "nominal_base_goal_pose": goal_pose.tolist(),
        "base_position_error_m": round(position_error, 4),
        "base_orientation_error_deg": round(float(np.degrees(orientation_error)), 2),
    }


def effective_cost_weights(config) -> dict:
    """Return the cost weights actually active on the task config."""
    return {
        key: value
        for key, value in vars(config).items()
        if key.startswith("w_")
    }


def save_video(task, qpos_traj: np.ndarray, path_stem: Path) -> dict:
    """Render a recorded headless trajectory to GIF and, when available, MP4."""
    import mujoco
    from PIL import Image

    path_stem.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(task.sim_model, height=360, width=480)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [-1.1, 0.0, 0.35] if "walk" in task.name else [0.0, 0.0, 0.35]
    camera.azimuth, camera.elevation = 145.0, -24.0
    camera.distance = 4.3 if "walk" in task.name else 2.7
    frames = []
    # run_single_episode records at 50 Hz; 10 Hz is enough for inspection.
    for qpos in qpos_traj[::5]:
        task.data.qpos[:] = qpos
        mujoco.mj_forward(task.sim_model, task.data)
        renderer.update_scene(task.data, camera=camera)
        frames.append(renderer.render().copy())
    renderer.close()
    gif = Path(f"{path_stem}.gif")
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(gif, save_all=True, append_images=images[1:], duration=100, loop=0)
    sheet = Path(f"{path_stem}.png")
    columns = 4
    selected = [
        frames[index]
        for index in np.linspace(
            0, len(frames) - 1, min(12, len(frames)), dtype=int
        )
    ]
    blank = np.zeros_like(selected[0])
    rows = [
        np.concatenate(
            list(selected[i : i + columns])
            + [blank] * (columns - len(selected[i : i + columns])),
            axis=1,
        )
        for i in range(0, len(selected), columns)
    ]
    Image.fromarray(np.concatenate(rows, axis=0)).save(sheet)
    result = {"gif": str(gif), "contact_sheet": str(sheet)}
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        mp4 = Path(f"{path_stem}.mp4")
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "480x360",
                "-r",
                "10",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                str(mp4),
            ],
            input=np.stack(frames).tobytes(),
            check=True,
            timeout=120,
        )
        result["mp4"] = str(mp4)
    return result


def main():
    started_at = int(time.time())
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-module")
    ap.add_argument("--optimizer", default="cem")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--episode-length-s", type=float, default=10.0)
    ap.add_argument("--num-rollouts", type=int)
    ap.add_argument("--horizon", type=float)
    ap.add_argument("--max-opt-iters-per-step", type=int)
    ap.add_argument("--step-log-path")
    ap.add_argument("--task-config-json")
    ap.add_argument("--settle-phase-s", type=float, default=0.0)
    ap.add_argument("--settle-config-json")
    ap.add_argument("--object-start-pose", type=float, nargs=7)
    ap.add_argument("--object-goal-pose", type=float, nargs=7)
    ap.add_argument("--object-size", type=float, nargs=3)
    ap.add_argument("--object-mass", type=float)
    ap.add_argument("--video-dir")
    a = ap.parse_args()

    import sumo.controller  # noqa: F401 - registers overrides
    import sumo.tasks  # noqa: F401 - registers tasks
    import judo.simulation.hierarchical_mj_simulation as hierarchical_sim
    import judo.utils.hierarchical_mj_rollout_backend as hierarchical_rollout
    from judo.optimizers import get_registered_optimizers
    from judo.tasks import get_registered_tasks
    from sumo.controller import Controller, ControllerConfig
    from sumo.run_mpc.run_mpc import RunMPCConfig, _create_sim, run_single_episode
    from sumo.utils.mujoco import G1RolloutBackend

    # The native backend supports an infinite cutoff. Use it so every sampled
    # trajectory executes the full fixed-step horizon, independent of host load.
    hierarchical_rollout.DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME = float("inf")
    hierarchical_sim.DEFAULT_SPOT_ROLLOUT_CUTOFF_TIME = float("inf")

    if a.task_module:
        spec = importlib.util.spec_from_file_location("agent_task", a.task_module)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    reg = get_registered_tasks()[a.task]
    sim = _create_sim(a.task)
    task = sim.task
    approach_config = (
        json.loads(a.task_config_json) if a.task_config_json else {}
    )
    settle_config = (
        json.loads(a.settle_config_json) if a.settle_config_json else {}
    )

    def apply_task_config(values: dict) -> None:
        unknown = sorted(set(values) - set(vars(task.config)))
        if unknown:
            raise ValueError(f"unknown task config fields: {unknown}")
        for key, value in values.items():
            setattr(task.config, key, value)

    apply_task_config(approach_config)
    if a.object_size is not None:
        size = np.asarray(a.object_size, dtype=float)
        if hasattr(task, "configure_object_proxy") and a.object_mass is not None:
            task.configure_object_proxy(size, float(a.object_mass))
        else:
            import mujoco

            half_size = size / 2.0
            for geom_name in ("box_collision", "box_visual"):
                task.sim_model.geom(geom_name).size[:] = half_size
            if a.object_mass is not None:
                body_id = task.sim_model.body("box_body").id
                mass = float(a.object_mass)
                task.sim_model.body_mass[body_id] = mass
                task.sim_model.body_inertia[body_id] = (
                    mass
                    / 3.0
                    * np.array(
                        [
                            half_size[1] ** 2 + half_size[2] ** 2,
                            half_size[0] ** 2 + half_size[2] ** 2,
                            half_size[0] ** 2 + half_size[1] ** 2,
                        ]
                    )
                )
            task.config.object_half_height = float(half_size[2])
            mujoco.mj_setConst(task.sim_model, task.data)
    # The hierarchical simulator creates native model copies in _create_sim.
    # H-018 changes proxy geometry after that point, so refresh only its native
    # systems. Existing tasks retain their original initialization path.
    if hasattr(task, "weld_eq_id") and hasattr(sim, "_init_cpp_systems"):
        sim._init_cpp_systems(reg.locomotion_policy_path)
    object_proxy = None
    if all(
        name in {task.sim_model.geom(i).name for i in range(task.sim_model.ngeom)}
        for name in ("box_collision", "box_visual")
    ):
        body_id = task.sim_model.body("box_body").id
        object_proxy = {
            "full_size_m": (
                2.0 * task.sim_model.geom("box_collision").size
            ).tolist(),
            "mass_kg": float(task.sim_model.body_mass[body_id]),
            "inertia_kg_m2": task.sim_model.body_inertia[body_id].tolist(),
        }
    if a.object_start_pose is not None:
        start = np.asarray(a.object_start_pose, dtype=float)
        start_quat = start[3:7] / np.linalg.norm(start[3:7])
        task.config.start_x = float(start[0])
        task.config.start_y = float(start[1])
        task.config.start_z = float(start[2])
        task.config.start_yaw_deg = float(
            np.degrees(
                atan2(
                    2.0
                    * (
                        start_quat[0] * start_quat[3]
                        + start_quat[1] * start_quat[2]
                    ),
                    1.0
                    - 2.0
                    * (
                        start_quat[2] * start_quat[2]
                        + start_quat[3] * start_quat[3]
                    ),
                )
            )
        )
    if a.object_goal_pose is not None:
        goal = np.asarray(a.object_goal_pose, dtype=float)
        goal_quat = goal[3:7] / np.linalg.norm(goal[3:7])
        task.config.goal_x = float(goal[0])
        task.config.goal_y = float(goal[1])
        task.config.goal_z = float(goal[2])
        task.config.goal_yaw_deg = float(
            np.degrees(
                atan2(
                    2.0
                    * (
                        goal_quat[0] * goal_quat[3]
                        + goal_quat[1] * goal_quat[2]
                    ),
                    1.0
                    - 2.0
                    * (
                        goal_quat[2] * goal_quat[2]
                        + goal_quat[3] * goal_quat[3]
                    ),
                )
            )
        )
    opt_cls, opt_cfg_cls = get_registered_optimizers()[a.optimizer]
    ocfg = opt_cfg_cls()
    ocfg.set_override(a.task)
    if a.num_rollouts:
        ocfg.num_rollouts = a.num_rollouts
    ccfg = ControllerConfig()
    ccfg.set_override(a.task)
    if a.horizon:
        ccfg.horizon = a.horizon
    if a.max_opt_iters_per_step is not None:
        if a.max_opt_iters_per_step < 1:
            raise ValueError("--max-opt-iters-per-step must be >= 1")
        ccfg.max_opt_iters = a.max_opt_iters_per_step
    ctrl = Controller(ccfg, task, opt_cls(ocfg, task.nu), rollout_backend=reg.rollout_backend,
                      rollout_backend_registry={"mujoco_g1": G1RolloutBackend})
    approach_state = dict(vars(task.config))
    walk_cost_weights = effective_cost_weights(task.config)
    apply_task_config(settle_config)
    settle_cost_weights = effective_cost_weights(task.config)
    apply_task_config(approach_state)
    cfg = RunMPCConfig(
        init_task=a.task,
        visualize=False,
        num_episodes=a.episodes,
        episode_length_s=a.episode_length_s + a.settle_phase_s,
        save_results=False,
    )
    out = []
    for i in range(a.episodes):
        apply_task_config(approach_state)
        phase_state = {
            "settle": False,
            "walk_end_qpos": None,
            "walk_end_qvel": None,
        }
        post_sim_step = task.post_sim_step
        release_refresh_state = {"done": False}

        def post_sim_step_with_phase_switch():
            post_sim_step()
            if (
                hasattr(task, "release_time_s")
                and task.release_time_s is not None
                and not release_refresh_state["done"]
            ):
                # eq_active is data state. Native hierarchical systems own
                # private MjData instances, so rebuild from the model whose
                # eq_active0 the task switched off at release.
                if hasattr(sim, "_init_cpp_systems"):
                    sim._init_cpp_systems(reg.locomotion_policy_path)
                backend = ctrl.rollout_backend
                if hasattr(backend, "_setup_mujoco_extensions"):
                    backend._setup_mujoco_extensions(
                        task.sim_model,
                        backend._policy_path,
                        backend.num_threads,
                    )
                release_refresh_state["done"] = True
            if (
                a.settle_phase_s > 0.0
                and not phase_state["settle"]
                and task.data.time >= a.episode_length_s
            ):
                phase_state["walk_end_qpos"] = np.asarray(
                    task.data.qpos
                ).copy()
                phase_state["walk_end_qvel"] = np.asarray(
                    task.data.qvel
                ).copy()
                apply_task_config(settle_config)
                phase_state["settle"] = True

        task.post_sim_step = post_sim_step_with_phase_switch
        reset = np.asarray(task.reset_pose)
        actual_start_pose = reset[
            task.body_pose_idx : task.body_pose_idx + 7
        ].tolist()
        episode_seed = a.episode_seed + i
        t = time.time()
        planning_wall_s = 0.0
        update_action = ctrl.update_action
        update_iteration = ctrl._update_iteration
        planning_rows = []
        iteration_count = 0

        def counted_update_iteration():
            nonlocal iteration_count
            iteration_count += 1
            return update_iteration()

        def timed_update_action():
            nonlocal planning_wall_s
            planning_started = time.perf_counter()
            before_iterations = iteration_count
            try:
                result = update_action()
            finally:
                planning_wall_s += time.perf_counter() - planning_started
            if hasattr(task, "cost_terms"):
                terms = task.cost_terms(
                    ctrl.states,
                    ctrl.sensors,
                    ctrl.rollout_controls,
                    ctrl.system_metadata,
                )
                best = int(np.argmax(ctrl.rewards))
                physics = (
                    task.physics_log[-1]
                    if getattr(task, "physics_log", None)
                    else {}
                )
                pose = task.data.qpos[
                    task.object_pose_idx : task.object_pose_idx + 7
                ]
                goal = np.array(
                    [
                        task.config.goal_x,
                        task.config.goal_y,
                        task.config.goal_z,
                    ]
                )
                error = pose[:3] - goal
                planning_rows.append(
                    {
                        "planning_step": len(planning_rows),
                        "time_s": physics.get("time_s", 0.0),
                        "samples": int(len(ctrl.rewards)),
                        "iterations_run": iteration_count
                        - before_iterations,
                        "cost_terms": {
                            name: {
                                "executed_best": float(values[best]),
                                "sample_mean": float(np.mean(values)),
                                "sample_min": float(np.min(values)),
                            }
                            for name, values in terms.items()
                        },
                        "block_goal_error": {
                            "xyz_m": error.tolist(),
                            "position_m": float(np.linalg.norm(error)),
                            "xy_m": float(np.linalg.norm(error[:2])),
                            "yaw_deg": float(task._yaw_error_deg(pose[3:7])),
                        },
                        "weld_active": physics.get("weld_active"),
                        "base_xy_displacement_m": physics.get(
                            "base_xy_displacement_m"
                        ),
                        "base_yaw_change_deg": physics.get(
                            "base_yaw_change_deg"
                        ),
                        "robot_block_contacts_since_previous_plan": int(
                            sum(
                                row["robot_block_contacts"]
                                for row in getattr(task, "physics_log", [])
                                if row["time_s"]
                                > (
                                    planning_rows[-1]["time_s"]
                                    if planning_rows
                                    else -1.0
                                )
                            )
                        ),
                        "block_floor_contacts_since_previous_plan": int(
                            sum(
                                row["block_floor_contacts"]
                                for row in getattr(task, "physics_log", [])
                                if row["time_s"]
                                > (
                                    planning_rows[-1]["time_s"]
                                    if planning_rows
                                    else -1.0
                                )
                            )
                        ),
                    }
                )
            return result

        ctrl._update_iteration = counted_update_iteration
        ctrl.update_action = timed_update_action
        try:
            ep = run_single_episode(
                cfg, task, ctrl, sim, episode_idx=episode_seed
            )
        finally:
            ctrl.update_action = update_action
            ctrl._update_iteration = update_iteration
            task.post_sim_step = post_sim_step
        final_qpos = np.asarray(task.data.qpos).copy()
        final_qvel = np.asarray(task.data.qvel).copy()
        walk_end_qpos = phase_state["walk_end_qpos"]
        walk_end_qvel = phase_state["walk_end_qvel"]
        if walk_end_qpos is None:
            walk_end_qpos = final_qpos
            walk_end_qvel = final_qvel
        walk_end = base_arrival_state(
            task, walk_end_qpos.copy(), walk_end_qvel.copy()
        )
        post_settle = (
            base_arrival_state(task, final_qpos.copy(), final_qvel.copy())
            if phase_state["settle"]
            else {}
        )
        sim_dt = task.sim_model.opt.timestep
        plan_dt = 1.0 / ctrl.controller_cfg.control_freq
        num_steps = int(cfg.episode_length_s / sim_dt) + 1
        steps_per_plan = max(1, int(plan_dt / sim_dt))
        planning_steps = (num_steps - 1) // steps_per_plan + 1
        row = {"episode": i, "sumo_seed": episode_seed,
               "sumo_deterministic": True,
               "rollout_cutoff_mode": "full_fixed_step_horizon",
               "eigen_threads": 1, "onnx_inter_op_threads": 1,
               "success": bool(ep["success"]), "failure": bool(ep["failure"]),
               "length_s": float(ep["length"]),
               "mean_reward": float(np.mean(ep["rewards"])) if len(ep["rewards"]) else None,
               "final_qpos": final_qpos.tolist(),
               "actual_start_pose": actual_start_pose,
               "settle_phase_s": a.settle_phase_s,
               "settle_phase_applied": phase_state["settle"],
               "walk_end_base_xy": walk_end.get("terminal_base_xy"),
               "walk_end_base_yaw_deg": walk_end.get("terminal_base_yaw_deg"),
               "walk_end_base_speed_m_s": walk_end.get(
                   "terminal_base_speed_m_s"
               ),
               "post_settle_base_xy": post_settle.get("terminal_base_xy"),
               "post_settle_base_yaw_deg": post_settle.get(
                   "terminal_base_yaw_deg"
               ),
               "post_settle_base_speed_m_s": post_settle.get(
                   "terminal_base_speed_m_s"
               ),
               "mpc_planning_steps": planning_steps,
               "max_opt_iters_per_step": ctrl.controller_cfg.max_opt_iters,
               "mpc_iterations_completed": iteration_count,
               "planning_wall_s": round(planning_wall_s, 3),
               "wall_s": round(time.time() - t, 1)}
        if hasattr(task, "weld_eq_id"):
            row.update(
                {
                    "object_pose_qpos_index": int(task.object_pose_idx),
                    "release_time_s": task.release_time_s,
                    "forced_release": bool(task.forced_release),
                    "release_reason": task.release_reason,
                    "error_at_release": task.release_error,
                    "object_pose_at_release": task.release_pose,
                    "object_pose_at_release_plus_post_release_s": (
                        task.release_plus_post_pose
                    ),
                    "release_transitions": int(task.release_transitions),
                    "physics_steps": task.physics_log,
                    "collision_manifest": task.collision_manifest(),
                    "runtime_idealization_manifest": (
                        task.runtime_idealization_manifest()
                    ),
                    "planning_steps": planning_rows,
                }
            )
            if task.release_pose is not None:
                release_gripper_distance = (
                    task.post_release_block_gripper_distance_m
                )
                row["post_release_block_gripper_distance_m"] = (
                    release_gripper_distance
                )
                if release_gripper_distance is not None:
                    row[
                        "post_release_block_gripper_distance_growth_m"
                    ] = (
                        release_gripper_distance
                        - float(task.release_block_gripper_distance_m)
                    )
        if a.step_log_path:
            step_log_path = Path(a.step_log_path).resolve()
            step_log_path.parent.mkdir(parents=True, exist_ok=True)
            with step_log_path.open("w") as stream:
                for planning_row in planning_rows:
                    stream.write(
                        json.dumps(planning_row, sort_keys=True) + "\n"
                    )
            row["step_log_path"] = str(step_log_path)
            row["step_log_rows"] = len(planning_rows)
        row.update(base_arrival_state(task, final_qpos, final_qvel))
        if a.video_dir and len(ep["qpos_traj"]):
            row["video"] = save_video(
                task,
                ep["qpos_traj"],
                Path(a.video_dir).resolve() / f"{a.task}_s{i}",
            )
        out.append(row)
    object_goal_pose = None
    if all(
        hasattr(task.config, name)
        for name in ("goal_x", "goal_y", "goal_z", "goal_yaw_deg")
    ):
        half_yaw = np.radians(task.config.goal_yaw_deg) / 2.0
        object_goal_pose = [
            task.config.goal_x,
            task.config.goal_y,
            task.config.goal_z,
            float(np.cos(half_yaw)),
            0.0,
            0.0,
            float(np.sin(half_yaw)),
        ]
    reset = np.asarray(task.reset_pose)
    object_start_pose = reset[
        task.object_pose_idx : task.object_pose_idx + 7
    ].tolist() if hasattr(task, "object_pose_idx") else None
    robot_start_pose = reset[
        task.body_pose_idx : task.body_pose_idx + 7
    ].tolist() if hasattr(task, "body_pose_idx") else None
    print(json.dumps({"run_id": f"sumo-{started_at}-{a.task}", "task": a.task,
                      "sumo_deterministic": True,
                      "rollout_cutoff_mode": "full_fixed_step_horizon",
                      "eigen_threads": 1, "onnx_inter_op_threads": 1,
                      "optimizer": a.optimizer, "num_rollouts": ocfg.num_rollouts,
                      "task_config": approach_config,
                      "settle_phase_s": a.settle_phase_s,
                      "settle_config": settle_config,
                      "phase_cost_weights": {
                          "walk": walk_cost_weights,
                          "settle": (
                              settle_cost_weights
                              if a.settle_phase_s > 0.0
                              else None
                          ),
                      },
                      "controller_horizon_s": ctrl.controller_cfg.horizon,
                      "max_opt_iters_per_step": ctrl.controller_cfg.max_opt_iters,
                      "backend": type(ctrl.rollout_backend).__name__,
                      "object_proxy": object_proxy,
                      "object_start_pose": object_start_pose,
                      "robot_start_pose": robot_start_pose,
                      "object_goal_pose": object_goal_pose, "episodes": out}))


if __name__ == "__main__":
    main()
