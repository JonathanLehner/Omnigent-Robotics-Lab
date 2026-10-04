"""Offline H-019 Gate 0 using synthetic calibration and old dev scenes only."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from lab import pipeline, scenes, sim
from lab.stages.vlm_measure_anchor import measure_blocks_from_picture


SCENE_IDS = (
    [f"T2-dev-{index:02d}" for index in range(11, 24)]
    + [f"T3-dev-{index:02d}" for index in range(13, 25)]
)
SYNTHETIC_Y_M = (-0.12, -0.06, 0.0, 0.04, 0.10)


def _parsed_from_truth(blocks: list[dict]) -> list[dict]:
    return [
        {
            "id": block["id"],
            "color": block["color"],
            "type": block["type"],
            "layer": round((block["pos"][2] - 0.05) / 0.10),
            "supported_by": list(block.get("on", [])),
            "rough_y_cm": float(block["pos"][1]) * 100.0,
        }
        for block in blocks
    ]


def _measure_case(case: str, blocks: list[dict], image_path: str) -> list[dict]:
    measured, _ = measure_blocks_from_picture(
        _parsed_from_truth(blocks), image_path
    )
    truth_by_id = {block["id"]: block for block in blocks}
    rows = []
    for block in measured:
        truth = truth_by_id[block["id"]]
        row = {
            "case": case,
            "block": block["id"],
            "color": block["color"],
            "type": block["type"],
            "layer": block["layer"],
            "true_y_cm": float(truth["pos"][1]) * 100.0,
            "perceived_y_cm": float(block["pos"][1]) * 100.0,
            "e_cm": (
                float(block["pos"][1]) - float(truth["pos"][1])
            ) * 100.0,
            "fallback": block["lateral_measurement_fallback"],
        }
        rows.append(row)
        print(json.dumps(row, sort_keys=True))
    return rows


def _synthetic_gate(output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    colors = ["red", "blue", "green", "yellow", "orange"]
    for index, (lateral_y, color) in enumerate(zip(SYNTHETIC_Y_M, colors)):
        block = {
            "id": "b0",
            "type": "cube",
            "color": color,
            "pos": [0.0, lateral_y, 0.05],
            "yaw": 0.0,
            "on": [],
        }
        paths = sim.render_structure_views(
            [scenes.to_world(block)],
            str(output_dir / f"single_{index}"),
        )
        rows.extend(
            _measure_case(f"synthetic-single-{lateral_y:+.2f}", [block], paths["robot side"])
        )

    stack = [
        {
            "id": "b0",
            "type": "cube",
            "color": "purple",
            "pos": [0.0, 0.07, 0.05],
            "yaw": 0.0,
            "on": [],
        },
        {
            "id": "b1",
            "type": "cube",
            "color": "cyan",
            "pos": [0.0, 0.07, 0.15],
            "yaw": 0.0,
            "on": ["b0"],
        },
    ]
    paths = sim.render_structure_views(
        [scenes.to_world(block) for block in stack],
        str(output_dir / "stack_y_plus_007"),
    )
    rows.extend(_measure_case("synthetic-stack-+0.07", stack, paths["robot side"]))
    return rows


def _old_scene_gate() -> list[dict]:
    rows = []
    for scene_id in SCENE_IDS:
        scene = scenes.load_scene(scene_id)
        rows.extend(
            _measure_case(
                scene_id,
                scene["target"]["blocks"],
                str(pipeline.ROOT / scene["image_views"]["robot side"]),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=pipeline.ROOT / "gate0" / "v47_calibration",
    )
    args = parser.parse_args()

    synthetic = _synthetic_gate(args.output_dir)
    old = _old_scene_gate()
    synthetic_max = max(abs(row["e_cm"]) for row in synthetic)
    old_abs = [abs(row["e_cm"]) for row in old]
    scene_means = {
        scene_id: statistics.mean(
            row["e_cm"] for row in old if row["case"] == scene_id
        )
        for scene_id in ("T2-dev-20", "T2-dev-23")
    }
    checks = {
        "G0a_max_abs_e_cm_le_0.3": synthetic_max <= 0.3,
        "G0b_median_abs_e_cm_le_0.5": statistics.median(old_abs) <= 0.5,
        "G0b_max_abs_e_cm_le_1.5": max(old_abs) <= 1.5,
        "G0b_T2_dev_20_mean_e_within_1.0": abs(scene_means["T2-dev-20"]) <= 1.0,
        "G0b_T2_dev_23_mean_e_within_1.0": abs(scene_means["T2-dev-23"]) <= 1.0,
    }
    summary = {
        "synthetic_blocks": len(synthetic),
        "synthetic_max_abs_e_cm": synthetic_max,
        "old_scenes": len(SCENE_IDS),
        "old_blocks": len(old),
        "old_median_abs_e_cm": statistics.median(old_abs),
        "old_max_abs_e_cm": max(old_abs),
        "scene_mean_e_cm": scene_means,
        "checks": checks,
        "passed": all(checks.values()),
    }
    print("GATE0 " + json.dumps(summary, sort_keys=True))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
