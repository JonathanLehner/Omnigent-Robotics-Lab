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
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--task-module")
    ap.add_argument("--optimizer", default="cem")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--episode-length-s", type=float, default=10.0)
    ap.add_argument("--num-rollouts", type=int)
    ap.add_argument("--horizon", type=float)
    a = ap.parse_args()

    import sumo.controller  # noqa: F401 - registers overrides
    import sumo.tasks  # noqa: F401 - registers tasks
    from judo.optimizers import get_registered_optimizers
    from judo.tasks import get_registered_tasks
    from sumo.controller import Controller, ControllerConfig
    from sumo.run_mpc.run_mpc import RunMPCConfig, _create_sim, run_single_episode
    from sumo.utils.mujoco import G1RolloutBackend

    if a.task_module:
        spec = importlib.util.spec_from_file_location("agent_task", a.task_module)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))

    reg = get_registered_tasks()[a.task]
    sim = _create_sim(a.task)
    task = sim.task
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
        t = time.time()
        ep = run_single_episode(cfg, task, ctrl, sim, episode_idx=i)
        out.append({"episode": i, "success": bool(ep["success"]), "failure": bool(ep["failure"]),
                    "length_s": float(ep["length"]), "mean_reward": float(np.mean(ep["rewards"])) if ep["rewards"] else None,
                    "final_qpos": np.asarray(ep["qpos_traj"][-1]).round(4).tolist() if ep["qpos_traj"] else None,
                    "wall_s": round(time.time() - t, 1)})
    print(json.dumps({"task": a.task, "optimizer": a.optimizer, "num_rollouts": ocfg.num_rollouts,
                      "backend": type(ctrl.rollout_backend).__name__, "episodes": out}))


if __name__ == "__main__":
    main()
