"""Offline E1/E2 and V5 replay harness for H-026.

E1 is intentionally an explicit CLI action; importing this module never calls a
model or runs simulation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from lab import pipeline, scenes
from lab.method_models import call_model
from lab.pipeline import ROOT
from lab.stages.v47b_footprint_override import apply_v47b_to_response
from lab.stages.vlm_measure import _sample_text
from lab.stages.vlm_measure_anchor import measure_blocks_from_picture

TARGET_MANIFEST = ROOT / "methods" / "v6_combo_v47b.targets.json"
DEFAULT_R103_EPISODES = (
    ROOT.parent
    / "Omnigent-Robotics-Lab"
    / "runs"
    / "1791114791_v6_combo"
    / "episodes.jsonl"
)
DEFAULT_R104_EPISODES = (
    ROOT.parent
    / "Omnigent-Robotics-Lab"
    / "runs"
    / "1791115477_v0"
    / "episodes.jsonl"
)


def _layer(block: dict) -> int:
    if "layer" in block:
        return int(block["layer"])
    return round((float(block["pos"][2]) - 0.05) / 0.10)


def targeted_scene_ids_from_specs(scene_dir: Path | None = None) -> list[str]:
    """Apply H-026's target rule to JSON specs, without loading image pixels."""
    scene_dir = scene_dir or ROOT / "scenes" / "dev"
    selected = []
    for path in sorted(scene_dir.glob("T*-dev-*.json")):
        scene = json.loads(path.read_text())
        blocks = scene["target"]["blocks"]
        types_by_color: dict[str, set[str]] = {}
        for block in blocks:
            types_by_color.setdefault(block["color"], set()).add(block["type"])
        qualifies = (
            any(block["color"] == "white" for block in blocks)
            or any(
                {"cube", "brick"} <= block_types
                for block_types in types_by_color.values()
            )
            or any(block["color"] == "cyan" for block in blocks)
        )
        if qualifies:
            selected.append(scene["id"])
    return selected


def validate_target_manifest() -> list[str]:
    manifest = json.loads(TARGET_MANIFEST.read_text())
    filed = manifest["scene_ids"]
    derived = targeted_scene_ids_from_specs()
    if filed != derived:
        raise AssertionError(
            f"target manifest differs from H-026 spec rule: filed={filed}, derived={derived}"
        )
    return filed


def _match_by_color_and_pose(
    reference: list[dict], candidate: list[dict]
) -> dict[int, int]:
    """Match identities without using type, so type errors remain observable."""
    choices = []
    for reference_index, left in enumerate(reference):
        for candidate_index, right in enumerate(candidate):
            if left.get("color") != right.get("color"):
                continue
            left_y = float(
                left.get("rough_y_cm", float(left.get("pos", [0, 0])[1]) * 100)
            )
            right_y = float(
                right.get("rough_y_cm", float(right.get("pos", [0, 0])[1]) * 100)
            )
            score = 1000 * abs(_layer(left) - _layer(right)) + abs(left_y - right_y)
            choices.append((score, reference_index, candidate_index))
    matched_reference: set[int] = set()
    matched_candidate: set[int] = set()
    result = {}
    for _score, reference_index, candidate_index in sorted(choices):
        if (
            reference_index in matched_reference
            or candidate_index in matched_candidate
        ):
            continue
        result[reference_index] = candidate_index
        matched_reference.add(reference_index)
        matched_candidate.add(candidate_index)
    return result


def score_structure(parsed: list[dict], truth: list[dict]) -> dict:
    """Score H-026 structure fields with a matcher independent of block type."""
    matched = _match_by_color_and_pose(truth, parsed)
    wrong_type = []
    wrong_layer = []
    rows = []
    for truth_index, parsed_index in sorted(matched.items()):
        true = truth[truth_index]
        seen = parsed[parsed_index]
        type_wrong = true["type"] != seen.get("type")
        layer_wrong = _layer(true) != _layer(seen)
        if type_wrong:
            wrong_type.append(true["id"])
        if layer_wrong:
            wrong_layer.append(true["id"])
        rows.append(
            {
                "true_id": true["id"],
                "perceived_id": seen.get("id"),
                "color": true["color"],
                "true_type": true["type"],
                "perceived_type": seen.get("type"),
                "true_layer": _layer(true),
                "perceived_layer": _layer(seen),
                "type_wrong": type_wrong,
                "layer_wrong": layer_wrong,
            }
        )
    missing = [
        truth[index]["id"] for index in range(len(truth)) if index not in matched
    ]
    matched_parsed = set(matched.values())
    spurious = [
        parsed[index].get("id")
        for index in range(len(parsed))
        if index not in matched_parsed
    ]
    return {
        "structure_mismatch": bool(missing or spurious or wrong_type or wrong_layer),
        "type_errors": len(wrong_type),
        "wrong_type_true_ids": wrong_type,
        "wrong_layer_true_ids": wrong_layer,
        "missing_true_ids": missing,
        "spurious_perceived_ids": spurious,
        "matched_blocks": rows,
    }


def _false_override_count(
    baseline: list[dict], overridden: list[dict], truth: list[dict]
) -> int:
    before = _match_by_color_and_pose(truth, baseline)
    after = _match_by_color_and_pose(truth, overridden)
    count = 0
    for truth_index in set(before) & set(after):
        true_type = truth[truth_index]["type"]
        if (
            baseline[before[truth_index]].get("type") == true_type
            and overridden[after[truth_index]].get("type") != true_type
        ):
            count += 1
    return count


def majority_vote_types(responses: list[list[dict]]) -> list[dict]:
    """E2 per-block majority in a disjoint group of five; ties keep parse 1."""
    if len(responses) != 5:
        raise ValueError(f"E2 requires groups of five responses, got {len(responses)}")
    voted = copy.deepcopy(responses[0])
    for reference_index, block in enumerate(voted):
        votes = [block["type"]]
        for response in responses[1:]:
            match = _match_by_color_and_pose([block], response)
            if 0 in match:
                votes.append(response[match[0]]["type"])
        counts = Counter(votes)
        winners = [
            block_type
            for block_type, count in counts.items()
            if count == max(counts.values())
        ]
        block["type"] = (
            winners[0] if len(winners) == 1 else responses[0][reference_index]["type"]
        )
    return voted


def _json_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def run_e1_e2(output_dir: Path) -> dict:
    """Run the filed E1/E2 design. This is not invoked by build validation."""
    output_dir.mkdir(parents=True, exist_ok=False)
    calls_path = output_dir / "e1_calls.jsonl"
    e2_path = output_dir / "e2_groups.jsonl"
    cfg = pipeline.load_method("v6_combo_v47b")
    scene_ids = validate_target_manifest()
    prompt_hashes = set()
    call_count = 0
    with calls_path.open("x") as calls_file, e2_path.open("x") as e2_file:
        for scene_id in scene_ids:
            scene = scenes.load_scene(scene_id)
            vlm_picture = ROOT / scene["image_multiview"]
            measurement_picture = ROOT / scene["image_views"]["robot side"]
            responses = []
            for draw in range(20):
                result = call_model(
                    cfg["perceive"]["model"],
                    cfg["perceive"]["prompt"],
                    text=_sample_text(cfg["perceive"]),
                    image=str(vlm_picture),
                    use_cache=False,
                )
                if result["cached"]:
                    raise AssertionError(f"{scene_id} draw {draw} was cached")
                raw = copy.deepcopy(result["output"]["blocks"])
                baseline, _baseline_info = measure_blocks_from_picture(
                    raw, measurement_picture
                )
                overridden, override_info = apply_v47b_to_response(
                    raw, measurement_picture, cfg
                )
                before_score = score_structure(
                    baseline, scene["target"]["blocks"]
                )
                after_score = score_structure(
                    overridden, scene["target"]["blocks"]
                )
                row = {
                    "hypothesis": "H-026",
                    "experiment": "E1",
                    "scene": scene_id,
                    "draw": draw,
                    "cached": False,
                    "prompt_sha256": result["prompt_sha256"],
                    "image_sha256": result["image_sha256"],
                    "response_sha256": result["response_sha256"],
                    "vlm_blocks": raw,
                    "baseline_score": before_score,
                    "override_score": after_score,
                    "false_overrides": _false_override_count(
                        baseline, overridden, scene["target"]["blocks"]
                    ),
                    "type_measurements": override_info["type_measurements"],
                }
                calls_file.write(json.dumps(row, sort_keys=True) + "\n")
                calls_file.flush()
                responses.append(raw)
                prompt_hashes.add(result["prompt_sha256"])
                call_count += 1

            for group_index in range(4):
                group = responses[group_index * 5 : (group_index + 1) * 5]
                voted_raw = majority_vote_types(group)
                voted, _voted_info = measure_blocks_from_picture(
                    voted_raw, measurement_picture
                )
                voted_overridden, override_info = apply_v47b_to_response(
                    voted_raw, measurement_picture, cfg
                )
                e2_file.write(
                    json.dumps(
                        {
                            "hypothesis": "H-026",
                            "experiment": "E2",
                            "scene": scene_id,
                            "group": group_index,
                            "draws": list(
                                range(group_index * 5, (group_index + 1) * 5)
                            ),
                            "sc_score": score_structure(
                                voted, scene["target"]["blocks"]
                            ),
                            "sc_override_score": score_structure(
                                voted_overridden, scene["target"]["blocks"]
                            ),
                            "false_overrides": _false_override_count(
                                voted, voted_overridden, scene["target"]["blocks"]
                            ),
                            "type_measurements": override_info["type_measurements"],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                e2_file.flush()

    summary = {
        "hypothesis": "H-026",
        "targeted_scenes": scene_ids,
        "targeted_scene_count": len(scene_ids),
        "uncached_calls_per_scene": 20,
        "e1_calls": call_count,
        "e2_groups": len(scene_ids) * 4,
        "prompt_sha256": sorted(prompt_hashes),
        "calls_path": str(calls_path),
        "e2_path": str(e2_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _read_episodes(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _canonical_episode(episode: dict) -> dict:
    episode = copy.deepcopy(episode)
    for key in ("video", "wall_s", "run_fingerprint"):
        episode.pop(key, None)
    for stage_row in episode.get("stages", []):
        for key in ("sumo_run_id", "planning_wall_s", "wall_s"):
            stage_row.pop(key, None)
    return episode


def _first_difference(left, right, path="$") -> str | None:
    if type(left) is not type(right):
        return f"{path}: types {type(left).__name__} != {type(right).__name__}"
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path}: keys {sorted(set(left) ^ set(right))}"
        for key in left:
            difference = _first_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: lengths {len(left)} != {len(right)}"
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            difference = _first_difference(
                left_value, right_value, f"{path}[{index}]"
            )
            if difference:
                return difference
        return None
    return None if left == right else f"{path}: {left!r} != {right!r}"


def _assert_bit_for_bit(expected: dict, actual: dict, label: str) -> dict:
    expected = _canonical_episode(expected)
    actual = _canonical_episode(actual)
    expected_bytes = json.dumps(
        expected, sort_keys=True, separators=(",", ":")
    ).encode()
    actual_bytes = json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    if expected_bytes != actual_bytes:
        raise AssertionError(
            f"{label} replay mismatch: {_first_difference(expected, actual)}"
        )
    return {
        "pair": [expected["scene"], expected["seed"]],
        "sha256": hashlib.sha256(expected_bytes).hexdigest(),
    }


def run_v5_reproduction(
    r103_path: Path = DEFAULT_R103_EPISODES,
    r104_path: Path = DEFAULT_R104_EPISODES,
    pairs: int = 6,
) -> dict:
    """Reproduce six R-104 v0 and six R-103 logged-response pairs."""
    r103 = _read_episodes(r103_path)[:pairs]
    r104 = _read_episodes(r104_path)[:pairs]
    if len(r103) != pairs or len(r104) != pairs:
        raise AssertionError("V5 source logs do not contain the requested pairs")
    if [
        (episode["scene"], episode["seed"]) for episode in r103
    ] != [
        (episode["scene"], episode["seed"]) for episode in r104
    ]:
        raise AssertionError("R-103 and R-104 V5 pairs differ")

    result = {"r104_v0": [], "r103_v6_combo_logged_vlm": []}
    v0_cfg = pipeline.load_method("v0")
    for expected in r104:
        actual = pipeline.run_episode(
            scenes.load_scene(expected["scene"]), v0_cfg, expected["seed"]
        )
        result["r104_v0"].append(
            _assert_bit_for_bit(expected, actual, "R-104/v0")
        )

    v6_cfg = pipeline.load_method("v6_combo")
    original_perceive = pipeline.STAGES["perceive"]["vlm_measure_anchored"]
    try:
        for expected in r103:
            parsed_blocks = copy.deepcopy(expected["perceive"]["parsed_blocks"])
            base_info = copy.deepcopy(expected["perceive"])
            base_info["block_measurements"] = [
                {
                    key: row.get(key)
                    for key in (
                        "perceived_id",
                        "u_px",
                        "measurement_point",
                        "d_m",
                        "y_anchored",
                        "y_recentred",
                        "ppm_camera",
                        "ppm_blocks",
                    )
                }
                for row in expected["perceive"]["block_position_errors"]
                if row.get("perceived_id") is not None
            ]
            for key in (
                "n_blocks",
                "block_position_errors",
                "spurious_perceived_blocks",
                "n_missing",
                "n_spurious",
                "block_pose_errors",
                "structure_matches_true",
            ):
                base_info.pop(key, None)

            def replay_perception(_scene, _cfg, blocks=parsed_blocks, info=base_info):
                return copy.deepcopy(blocks), copy.deepcopy(info)

            pipeline.STAGES["perceive"][
                "vlm_measure_anchored"
            ] = replay_perception
            actual = pipeline.run_episode(
                scenes.load_scene(expected["scene"]), v6_cfg, expected["seed"]
            )
            result["r103_v6_combo_logged_vlm"].append(
                _assert_bit_for_bit(expected, actual, "R-103/v6_combo")
            )
    finally:
        pipeline.STAGES["perceive"][
            "vlm_measure_anchored"
        ] = original_perceive
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("targets", help="validate and print the frozen target list")
    e1 = commands.add_parser("e1", help="run the post-freeze E1/E2 offline replay")
    e1.add_argument("--output-dir", type=Path, required=True)
    v5 = commands.add_parser("v5", help="run the six-pair V5 reproduction gate")
    v5.add_argument("--r103", type=Path, default=DEFAULT_R103_EPISODES)
    v5.add_argument("--r104", type=Path, default=DEFAULT_R104_EPISODES)
    args = parser.parse_args()

    if args.command == "targets":
        ids = validate_target_manifest()
        print(json.dumps({"count": len(ids), "scene_ids": ids}, indent=2))
    elif args.command == "e1":
        print(json.dumps(run_e1_e2(args.output_dir), indent=2))
    elif args.command == "v5":
        print(json.dumps(run_v5_reproduction(args.r103, args.r104), indent=2))


if __name__ == "__main__":
    main()
