"""Watch one episode live in the MuJoCo viewer (macOS needs mjpython):

  uv run mjpython -m lab.view T3-dev-03            # method v0, seed 0
  uv run mjpython -m lab.view T3-dev-03 v1_support_sort 4

Same scene, method and seed as a batch episode -> the same rollout the batch scored.
"""

import sys
import time

import mujoco
import mujoco.viewer

from lab import pipeline, scenes, sim


def main(scene_id, method="v0", seed=0, speed=1.0):
    viewer = {}
    step = sim.World.step

    def step_and_show(self, n):
        for _ in range(n):
            step(self, 1)
            if "v" not in viewer:
                viewer["v"] = mujoco.viewer.launch_passive(self.m, self.d)
                viewer["v"].cam.lookat[:] = sim.SITE + [-0.25, 0, 0.2]
                viewer["v"].cam.distance, viewer["v"].cam.azimuth, viewer["v"].cam.elevation = 1.9, 215, -28
            if int(self.d.time / self.m.opt.timestep) % 10 == 0:
                viewer["v"].sync()
                time.sleep(10 * self.m.opt.timestep / speed)

    sim.World.step = step_and_show
    ep = pipeline.run_episode(scenes.load_scene(scene_id), pipeline.load_method(method), int(seed))
    print({k: ep[k] for k in ("success", "failures", "order")}, [round(b["pos_err"] * 100, 1) for b in ep["blocks"]], "cm")
    while viewer["v"].is_running():
        time.sleep(0.1)


if __name__ == "__main__":
    main(*sys.argv[1:])
