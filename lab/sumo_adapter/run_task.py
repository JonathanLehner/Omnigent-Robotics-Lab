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
import time
from math import atan2
from pathlib import Path

import numpy as np


def base_arrival_state(task, qpos: np.ndarray) -> dict:
    """Return the true terminal base pose, its nominal goal, and approach errors."""
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
    goal_xy = np.array([config.goal_x, config.goal_y])
    position_error = float(np.linalg.norm(qpos[pose_idx : pose_idx + 2] - goal_xy))
    quat = qpos[pose_idx + 3 : pose_idx + 7]
    # Spot's approach target has zero roll, pitch, and yaw. Report the full
    # quaternion angular distance, not only heading, so tilted arrivals show.
    orientation_error = 2.0 * np.arccos(np.clip(abs(quat[0]), 0.0, 1.0))
    return {
        "terminal_base_pose": terminal_pose.tolist(),
        "nominal_base_goal_pose": goal_pose.tolist(),
        "base_position_error_m": round(position_error, 4),
        "base_orientation_error_deg": round(float(np.degrees(orientation_error)), 2),
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
    result = {"gif": str(gif)}
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
    ctrl = Controller(ccfg, task, opt_cls(ocfg, task.nu), rollout_backend=reg.rollout_backend,
                      rollout_backend_registry={"mujoco_g1": G1RolloutBackend})
    cfg = RunMPCConfig(init_task=a.task, visualize=False, num_episodes=a.episodes,
                       episode_length_s=a.episode_length_s, save_results=False)
    out = []
    for i in range(a.episodes):
        episode_seed = a.episode_seed + i
        t = time.time()
        planning_wall_s = 0.0
        update_action = ctrl.update_action

        def timed_update_action():
            nonlocal planning_wall_s
            planning_started = time.perf_counter()
            try:
                return update_action()
            finally:
                planning_wall_s += time.perf_counter() - planning_started

        ctrl.update_action = timed_update_action
        try:
            ep = run_single_episode(
                cfg, task, ctrl, sim, episode_idx=episode_seed
            )
        finally:
            ctrl.update_action = update_action
        final_qpos = np.asarray(ep["qpos_traj"][-1]) if len(ep["qpos_traj"]) else None
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
               "final_qpos": final_qpos.tolist() if final_qpos is not None else None,
               "mpc_planning_steps": planning_steps,
               "max_opt_iters_per_step": ctrl.controller_cfg.max_opt_iters,
               "mpc_iterations_completed": planning_steps * ctrl.controller_cfg.max_opt_iters,
               "planning_wall_s": round(planning_wall_s, 3),
               "wall_s": round(time.time() - t, 1)}
        if final_qpos is not None:
            row.update(base_arrival_state(task, final_qpos))
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
                      "controller_horizon_s": ctrl.controller_cfg.horizon,
                      "backend": type(ctrl.rollout_backend).__name__,
                      "object_proxy": object_proxy,
                      "object_start_pose": object_start_pose,
                      "robot_start_pose": robot_start_pose,
                      "object_goal_pose": object_goal_pose, "episodes": out}))


if __name__ == "__main__":
    main()
