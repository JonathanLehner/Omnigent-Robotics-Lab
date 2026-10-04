"""H-025 G1: deterministic replay against R-103 and R-104 episode logs.

This is a read-only development gate. It reuses each episode's logged VLM parse
and logged deterministic Sumo terminal episode, then verifies that adding
passive T-disp telemetry leaves final poses and simulation time bit-identical.
It does not call the experiment runner or create a run record.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from lab import pipeline, scenes
from lab.stages import combo_build
from lab.stages.vlm_measure_anchor import measure_blocks_from_picture


R103_PAIRS = (
    ("T2-dev-12", 4023),
    ("T2-dev-01", 4000),
    ("T2-dev-11", 4018),
    ("T2-dev-25", 4036),
    ("T3-dev-01", 4100),
    ("T3-dev-42", 4136),
)
R104_PAIRS = (
    ("T2-dev-01", 4000),
    ("T2-dev-12", 4023),
    ("T3-dev-42", 4136),
)


def _episodes(path: Path) -> dict[tuple[str, int], dict]:
    return {
        (row["scene"], int(row["seed"])): row
        for row in (
            json.loads(line) for line in path.read_text().splitlines() if line
        )
    }


def _assert_bit_identical(expected: dict, actual: dict) -> None:
    assert actual["sim_time_s"] == expected["sim_time_s"]
    assert actual["success"] == expected["success"]
    expected_blocks = {row["id"]: row for row in expected["blocks"]}
    actual_blocks = {row["id"]: row for row in actual["blocks"]}
    assert actual_blocks.keys() == expected_blocks.keys()
    for block_id in expected_blocks:
        assert (
            actual_blocks[block_id]["final_pose"]
            == expected_blocks[block_id]["final_pose"]
        ), block_id


def _replay_v6(expected: dict) -> dict:
    logged_blocks = copy.deepcopy(expected["perceive"]["parsed_blocks"])
    logged_vlm_blocks = [
        {
            key: copy.deepcopy(block[key])
            for key in (
                "id",
                "color",
                "type",
                "layer",
                "supported_by",
                "rough_y_cm",
            )
        }
        for block in logged_blocks
    ]
    logged_walk = copy.deepcopy(expected["stages"][0])

    def replay_perception(scene, _cfg):
        blocks, info = measure_blocks_from_picture(
            logged_vlm_blocks,
            pipeline.ROOT / scene["image_views"]["robot side"],
        )
        assert blocks == logged_blocks
        return blocks, info | {
            "cached": False,
            "cache_mode": "logged_vlm_response",
            "replayed_logged_vlm_response": True,
            "parsed_blocks": copy.deepcopy(blocks),
        }

    pipeline.STAGES["perceive"]["vlm_measure_anchored"] = replay_perception
    combo_build._run_sumo = lambda _cfg, episode_seed: {"seed": episode_seed}
    combo_build._episode = lambda _result: copy.deepcopy(logged_walk)
    return pipeline.run_episode(
        scenes.load_scene(expected["scene"]),
        pipeline.load_method("v6_combo"),
        int(expected["seed"]),
    )


def _replay_v0(expected: dict) -> dict:
    return pipeline.run_episode(
        scenes.load_scene(expected["scene"]),
        pipeline.load_method("v0"),
        int(expected["seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runs-root",
        type=Path,
        required=True,
        help="directory containing the recorded R-103 and R-104 run folders",
    )
    args = parser.parse_args()
    r103 = _episodes(args.runs_root / "1791114791_v6_combo" / "episodes.jsonl")
    r104 = _episodes(args.runs_root / "1791115477_v0" / "episodes.jsonl")

    rows = []
    for pair in R103_PAIRS:
        expected = r103[pair]
        actual = _replay_v6(expected)
        _assert_bit_identical(expected, actual)
        telemetry = actual["placed_block_displacement"]
        rows.append(
            {
                "run": "R-103",
                "scene": pair[0],
                "seed": pair[1],
                "bit_identical": True,
                "sim_time_s": actual["sim_time_s"],
                "disp_blocks": [
                    row["block_id"]
                    for row in telemetry["blocks"]
                    if row["max_displacement_m_during_later_placements"] > 0.01
                ],
            }
        )
        if pair == ("T2-dev-12", 4023):
            b0 = next(
                row for row in telemetry["blocks"] if row["block_id"] == "b0"
            )
            assert b0["max_displacement_m_during_later_placements"] > 0.01
            assert b0["final_displacement_m"] > 0.01

    for pair in R104_PAIRS:
        expected = r104[pair]
        actual = _replay_v0(expected)
        _assert_bit_identical(expected, actual)
        rows.append(
            {
                "run": "R-104",
                "scene": pair[0],
                "seed": pair[1],
                "bit_identical": True,
                "sim_time_s": actual["sim_time_s"],
            }
        )

    print(json.dumps({"ok": True, "replays": rows}, indent=2))


if __name__ == "__main__":
    main()
