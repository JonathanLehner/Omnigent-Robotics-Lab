"""Offline H-013 Gate 0. This is the only v46 file allowed to read dev ground truth."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from lab import pipeline, scenes
from lab.stages import vlm_measure


SCENE_IDS = (
    [f"T1-dev-{index:02d}" for index in range(1, 13)]
    + [f"T2-dev-{index:02d}" for index in range(11, 15)]
    + [f"T3-dev-{index:02d}" for index in range(13, 17)]
)


def _oracle_structure(scene):
    blocks = scene["target"]["blocks"]
    by_id = {block["id"]: block for block in blocks}
    return {
        block["color"]: {
            "type": block["type"],
            "layer": round((block["pos"][2] - 0.05) / 0.10),
            "supporter_colors": sorted(by_id[sid]["color"] for sid in block["on"]),
            "y": block["pos"][1],
        }
        for block in blocks
    }


def _predicted_structure(blocks):
    by_id = {block["id"]: block for block in blocks}
    return {
        block["color"]: {
            "type": block["type"],
            "layer": block["layer"],
            "supporter_colors": sorted(
                by_id[sid]["color"] for sid in block["supported_by"] if sid in by_id
            ),
            "y": block["pos"][1],
        }
        for block in blocks
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument(
        "--reuse-recorded",
        action="store_true",
        help="reuse prior fresh v2 samples, sampling only missing scene/sample pairs",
    )
    args = parser.parse_args()
    method = pipeline.load_method("v46_vlm_measure")
    recorded = defaultdict(list)
    if args.reuse_recorded:
        for line in (pipeline.ROOT / "record/model_calls.jsonl").read_text().splitlines():
            call = json.loads(line)
            if call.get("prompt") == "prompts/perceive/v2.md":
                recorded[str(Path(call["image"]).resolve())].append(call["key"])
    original_call_model = vlm_measure.call_model
    errors_cm = []
    structure_correct = [0] * args.samples
    rows = []
    for scene_id in SCENE_IDS:
        scene = scenes.load_scene(scene_id)
        oracle = _oracle_structure(scene)
        for sample in range(args.samples):
            image = str((pipeline.ROOT / scene["image_multiview"]).resolve())
            replayed = sample < len(recorded[image])
            if replayed:
                key = recorded[image][sample]
                output = json.loads(
                    (pipeline.ROOT / f"record/model_cache/{key}.json").read_text()
                )
                vlm_measure.call_model = lambda *a, _key=key, _output=output, **k: {
                    "output": _output, "key": _key, "cached": False
                }
            else:
                vlm_measure.call_model = original_call_model
            blocks, info = pipeline.STAGES["perceive"]["vlm_measure"](scene, method)
            vlm_measure.call_model = original_call_model
            predicted = _predicted_structure(blocks)
            comparable = {
                color: {key: value for key, value in block.items() if key != "y"}
                for color, block in predicted.items()
            }
            expected = {
                color: {key: value for key, value in block.items() if key != "y"}
                for color, block in oracle.items()
            }
            correct = comparable == expected
            structure_correct[sample] += int(correct)
            matched_errors = {
                color: abs(predicted[color]["y"] - truth["y"]) * 100.0
                for color, truth in oracle.items()
                if color in predicted
            }
            errors_cm.extend(matched_errors.values())
            rows.append({
                "scene": scene_id,
                "sample": sample,
                "structure_correct": correct,
                "max_lateral_error_cm": max(matched_errors.values(), default=None),
                "cached": info["cached"],
                "recorded_replay": replayed,
                "vlm_output_sha256": info["vlm_output_sha256"],
                "fallback_blocks": info["fallback_blocks"],
            })
            print(json.dumps(rows[-1], sort_keys=True), flush=True)
    summary = {
        "scenes": len(SCENE_IDS),
        "samples_per_scene": args.samples,
        "matched_blocks": len(errors_cm),
        "max_lateral_error_cm": max(errors_cm),
        "median_lateral_error_cm": statistics.median(errors_cm),
        "structure_correct_by_sample": structure_correct,
        # model_calls.jsonl contains only cache misses, so replayed rows also
        # originated as fresh samples even though this invocation did not resample.
        "fresh_origin_samples": sum(not row["cached"] for row in rows),
        "fresh_calls_this_run": sum(
            not row["cached"] and not row["recorded_replay"] for row in rows
        ),
        "cached_calls": sum(row["cached"] for row in rows),
        "recorded_replays": sum(row["recorded_replay"] for row in rows),
        "fallback_blocks": sum(row["fallback_blocks"] for row in rows),
    }
    print("GATE0 " + json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
