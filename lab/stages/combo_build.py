"""H-020 composite: v2 Relic walk followed by v47 closed-loop placement."""

from lab.pipeline import stage
from lab.stages.closed_loop_place import build_closed_loop_weld_place
from lab.stages.h020_telemetry import walk_end_telemetry
from lab.stages.sumo_build import (
    _bridge_walked_base_pose,
    _episode,
    _run_sumo,
)


@stage("build", "sumo_walk_then_closed_loop_weld_place")
def build_sumo_walk_then_closed_loop_weld_place(world, spec, order, pairs, cfg, log):
    """Run v2's approach and bridge, then v47's measured weld placement."""
    result = _run_sumo(cfg, episode_seed=int(log["seed"]))
    episode = _episode(result)
    log["stages"].append({"phase": "walk_approach", **episode})
    log["walk_end"] = walk_end_telemetry(episode)
    if not episode["success"]:
        log["failures"].append("walk_approach_failed")
        return
    bridge = _bridge_walked_base_pose(world, episode)
    log["stages"].append({"phase": "sumo_to_assembly_state_bridge", **bridge})
    log["walk_end"] = walk_end_telemetry(episode, bridge)
    build_closed_loop_weld_place(world, spec, order, pairs, cfg, log)
