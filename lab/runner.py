"""Batch evaluation: gates, then parallel episodes, then a Run object in the record.

Gates (run before any budget is spent): method config loads and names registered stages, prompt files and
schemas parse, eval code matches its frozen hash, held-out scenes only with final_eval, a one-episode smoke
test completes. Episodes run in parallel processes (one per core, capped).
"""

import hashlib
import json
import os
import copy
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from lab import metrics, record, scenes

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FROZEN = ROOT / "record" / "frozen_eval.sha256"
BUDGET = record.DB_PATH.parent / "budget.json"  # per record, so the single-agent baseline has its own
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))
SUMO_ENV = Path.home() / "src" / "sumo" / ".pixi" / "envs" / "default"
METHOD_ARM_CAPS = {"v47_vlm_anchor": 40, "v0": 16}


def eval_hash() -> str:
    return hashlib.sha256((ROOT / "lab" / "metrics.py").read_bytes()).hexdigest()


def freeze_eval():
    """Human-run only (`uv run python -m lab.runner freeze`): records the current metrics.py hash."""
    FROZEN.parent.mkdir(parents=True, exist_ok=True)
    FROZEN.write_text(eval_hash())


def budget() -> dict:
    if not BUDGET.exists():
        BUDGET.write_text(json.dumps({"episodes_total": 1500, "episodes_used": 0}))
    b = json.loads(BUDGET.read_text())
    return b | {"episodes_left": b["episodes_total"] - b["episodes_used"]}


def _charge(n):
    b = budget()
    b["episodes_used"] += n
    BUDGET.write_text(json.dumps({k: b[k] for k in ("episodes_total", "episodes_used")}))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str, text: bool = True):
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=text,
    ).stdout


def _workspace_hash() -> tuple[str, list[str]]:
    """Hash tracked and untracked method inputs, including deleted-file markers."""
    raw = _git(
        "ls-files",
        "-z",
        "--cached",
        "--others",
        "--exclude-standard",
        "--",
        "lab",
        "methods",
        "prompts",
        text=False,
    )
    paths = sorted(filter(None, raw.split(b"\0")))
    digest = hashlib.sha256()
    names = []
    for encoded in paths:
        relative = encoded.decode("utf-8", errors="surrogateescape")
        names.append(relative)
        digest.update(encoded)
        digest.update(b"\0")
        path = ROOT / relative
        if path.is_file():
            digest.update(path.read_bytes())
        else:
            digest.update(b"<deleted>")
        digest.update(b"\0")
    return digest.hexdigest(), names


def _sumo_native_extensions() -> list[dict]:
    """Identify native code loaded by Sumo's default pixi environment without importing it."""
    candidates = {
        path
        for pattern in ("policy_rollout_pybind*", "_g1_extensions*")
        for path in SUMO_ENV.glob(f"lib/python*/site-packages/**/{pattern}")
        if path.is_file() and path.suffix in {".so", ".dylib", ".pyd"}
    }
    return [
        {
            "path": str(path.relative_to(SUMO_ENV)),
            "mtime_ns": path.stat().st_mtime_ns,
            "sha256": _sha256_file(path),
        }
        for path in sorted(candidates)
    ]


def capture_run_fingerprint() -> dict:
    """Capture immutable batch-start provenance for source and native code."""
    captured_at_ns = time.time_ns()
    workspace_sha256, workspace_files = _workspace_hash()
    source_state = {
        "git_head": _git("rev-parse", "HEAD").strip() or "nocommit",
        "workspace_sha256": workspace_sha256,
        "workspace_files": workspace_files,
        "sumo_native_extensions": _sumo_native_extensions(),
    }
    fingerprint_sha256 = hashlib.sha256(
        json.dumps(source_state, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "id": fingerprint_sha256,
        "captured_at": datetime.fromtimestamp(
            captured_at_ns / 1_000_000_000, tz=timezone.utc
        ).isoformat(),
        "captured_at_ns": captured_at_ns,
        **source_state,
    }


def gates(method: str, scene_ids: list[str], final_eval: bool) -> dict:
    """Cheap checks; any failure returns to the engineer without consuming budget."""
    from lab import pipeline

    if not FROZEN.exists() or FROZEN.read_text().strip() != eval_hash():
        raise RuntimeError("gate:frozen_eval: lab/metrics.py differs from its frozen hash; human sign-off needed")
    cfg = pipeline.load_method(method)
    for kind in ("perceive", "plan_order"):
        p = cfg.get(kind) or {}
        if "prompt" in p:
            json.loads((ROOT / p["prompt"]).with_suffix(".schema.json").read_text())
            (ROOT / p["prompt"]).read_text()
    loaded = [scenes.load_scene(s) for s in scene_ids]
    expected_scene_files = cfg.get("telemetry", {}).get("scene_file_count")
    if expected_scene_files is not None:
        for scene in loaded:
            _check_scene_hash_coverage(scene, int(expected_scene_files))
    held = [s["id"] for s in loaded if s["split"] == "heldout"]
    if held and not final_eval:
        raise RuntimeError(f"gate:heldout: {held} are held-out; only the human-approved final evaluation may use them")
    return {"cfg": cfg, "scenes": loaded}


def _check_scene_hash_coverage(scene: dict, expected_count: int) -> None:
    """Require a digest for every scene byte source consumed by an episode."""
    expected = {
        f"scenes/{scene['split']}/{scene['id']}.json",
        scene["image"],
        scene.get("image_multiview"),
        *(scene.get("image_views") or {}).values(),
    }
    expected.discard(None)
    actual = set(scene.get("_loaded_scene_files_sha256", {}))
    if len(expected) != expected_count or actual != expected:
        raise RuntimeError(
            f"gate:scene_hash_coverage:{scene['id']}: expected "
            f"{expected_count} distinct files and exact loader coverage; "
            f"declared={len(expected)}, hashed={len(actual)}, "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _method_arm_cap(
    experiment_id: str, method: str, planned_episodes: int | None = None
) -> int | None:
    if experiment_id == "X-049" and method in {
        "v6_combo",
        "v6_combo_t0",
        "v0",
    }:
        # X-049 is 192 episodes under Branch A and 96 under Branch B. Keep the
        # arms matched even if the record still carries the pre-branch total.
        half = (planned_episodes or 192) // 2
        ceiling = 48 if method == "v6_combo_t0" else 96
        return min(ceiling, half)
    return METHOD_ARM_CAPS.get(method)


VIDEO_FRAME_S = 0.4  # sim seconds between video frames; played back at 4x speed


def save_video(frames, path_stem: Path) -> dict:
    """Animated GIF (plays inline in the Omnigent web UI) + MP4 via ffmpeg when available (for the demo)."""
    from PIL import Image

    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(f"{path_stem}.gif", save_all=True, append_images=ims[1:], duration=int(VIDEO_FRAME_S * 250), loop=0)
    out = {"gif": str(Path(f"{path_stem}.gif").relative_to(ROOT))}
    h, w = frames[0].shape[:2]
    try:
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                        "-r", str(round(4 / VIDEO_FRAME_S)), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264",
                        f"{path_stem}.mp4"], input=np.stack(frames).tobytes(), check=True, timeout=120)
        out["mp4"] = str(Path(f"{path_stem}.mp4").relative_to(ROOT))
    except (OSError, subprocess.SubprocessError):
        pass  # ponytail: GIF only when ffmpeg is missing
    return out


def _episode(args):
    """args: (scene, cfg, seed, video_dir, keep_video). Frames are rendered only when keep_video is set."""
    scene, cfg, seed, video_dir, keep_video = args
    from lab import pipeline

    t = time.time()
    try:
        ep = pipeline.run_episode(scene, cfg, seed, frame_every_s=VIDEO_FRAME_S if video_dir and keep_video else None)
    except Exception:  # noqa: BLE001 - a crash is a recorded failure, not a lost batch
        ep = {"scene": scene["id"], "tier": scene["tier"], "seed": seed, "success": False, "blocks": [],
              "scene_files_sha256": dict(scene.get("_loaded_scene_files_sha256", {})),
              "failures": ["crash"], "error": traceback.format_exc()[-1500:]}
    frames = ep.pop("_frames", None)
    if frames:
        ep["video"] = save_video(frames, Path(video_dir) / f"{scene['id']}_s{seed}")
    ep["wall_s"] = round(time.time() - t, 2)
    return ep


def write_run_page(
    run_dir: Path,
    run_id: str,
    method: str,
    experiment_id: str,
    eps: list,
    summary: dict,
    fingerprint: dict,
) -> str:
    """runs/<run>/RUN.md: summary + embedded episode videos, viewable in the Omnigent web UI file viewer."""
    lo, hi = summary["success_ci95"]
    lines = [f"# {run_id}: {method} ({experiment_id})", "",
             f"Success {summary['success_rate']:.2f} (95% CI {lo:.2f}-{hi:.2f}), n={summary['episodes']}, "
             f"median placement error {summary['median_pos_err_cm']:.1f} cm, failures {summary['failure_categories'] or 'none'}.",
             "", "## Run fingerprint", "", "```json", json.dumps(fingerprint, indent=2, sort_keys=True), "```", "",
             "| scene | seed | success | block errors (cm) | failures | video |", "|---|---|---|---|---|---|"]
    for e in eps:
        errs = ", ".join(f"{b['pos_err'] * 100:.1f}" for b in e["blocks"])
        vid = f"[mp4]({Path(e['video']['mp4']).name})" if e.get("video", {}).get("mp4") else ""
        lines.append(f"| {e['scene']} | {e['seed']} | {'yes' if e['success'] else '**no**'} | {errs} | "
                     f"{', '.join(e['failures'])} | {vid} |")
    lines += ["", "## Videos (failures first)", ""]
    for e in sorted((e for e in eps if e.get("video")), key=lambda e: e["success"]):
        lines += [f"### {e['scene']} seed {e['seed']}: {'success' if e['success'] else 'FAILED ' + ', '.join(e['failures'])}",
                  "", f"![{e['scene']} seed {e['seed']}]({Path(e['video']['gif']).name})", ""]
    (run_dir / "RUN.md").write_text("\n".join(lines) + "\n")
    return str((run_dir / "RUN.md").relative_to(ROOT))


def run_sim_batch(experiment_id: str, scene_ids: list[str], method: str, episodes_per_scene: int = 3,
                  seed0: int = 0, author: str = "experiment_runner", final_eval: bool = False,
                  max_workers: int | None = None, pairs: list | None = None) -> dict:
    """pairs: optional explicit [[scene_id, seed], ...] (e.g. one episode per seed, round-robin over scenes);
    when given, scene_ids/episodes_per_scene/seed0 are ignored for job construction."""
    fingerprint = capture_run_fingerprint()
    exp = record.get(experiment_id)
    if exp is None or exp["kind"] != "experiment":
        raise ValueError(f"{experiment_id!r} is not an experiment in the record")
    if exp["data"].get("status") != "selected" and not final_eval:
        raise ValueError(f"experiment {experiment_id} has status {exp['data'].get('status')!r}; the planner must select it first")
    if pairs:
        pairs = [(str(sc), int(sd)) for sc, sd in pairs]
        scene_ids = sorted({sc for sc, _ in pairs})
    g = gates(method, scene_ids, final_eval)
    n = len(pairs) if pairs else len(scene_ids) * episodes_per_scene
    prior = record.query("run", contains=f'"experiment_id": "{experiment_id}"', limit=1000)
    # A batch call evaluates one method. A parity/verification run made under
    # the same experiment id for another method is still globally charged, but
    # must not consume this method arm's planned scene/seed tuples.
    done = set()
    for r in prior:
        if r["data"]["method"] != method:
            continue
        if r["data"].get("pairs"):  # explicit (scene, seed) runs
            done |= {(sc, int(sd)) for sc, sd in r["data"]["pairs"]}
        else:
            done |= {(scene_id, seed) for scene_id in r["data"].get("scene_ids", [])
                     for seed in range(r["data"]["seeds"][0], r["data"]["seeds"][-1] + 1)}
    requested = set(pairs) if pairs else {(scene_id, seed0 + i)
                                          for scene_id in scene_ids
                                          for i in range(episodes_per_scene)}
    # Count episodes, not just unique pairs: a repeated batch (same pairs again) must not slip past the plan.
    episodes_so_far = sum(int(r["data"]["episodes"]) for r in prior)
    method_episodes_so_far = sum(
        int(r["data"]["episodes"])
        for r in prior
        if r["data"]["method"] == method
    )
    arm_cap = _method_arm_cap(
        experiment_id, method, int(exp["data"]["episodes"])
    )
    if (
        not final_eval
        and arm_cap is not None
        and method_episodes_so_far + n > arm_cap
    ):
        raise RuntimeError(
            f"experiment {experiment_id} caps method {method} at "
            f"{arm_cap} episodes; {method_episodes_so_far} already "
            f"run, {n} more requested"
        )
    if not final_eval and episodes_so_far + n > int(exp["data"]["episodes"]):
        raise RuntimeError(f"experiment {experiment_id} planned {exp['data']['episodes']} episodes in total; "
                           f"{episodes_so_far} already run (all arms), {n} more requested. "
                           f"Propose a new experiment (or an amendment) for more.")
    if not final_eval and len(done | requested) > int(exp["data"]["episodes"]):
        raise RuntimeError(f"experiment {experiment_id} planned {exp['data']['episodes']} unique episodes; "
                           f"{len(done)} already run for {method}, "
                           f"{len(requested - done)} new episodes requested. "
                           f"Propose a new experiment for more.")
    if n > budget()["episodes_left"]:
        raise RuntimeError(f"budget: {n} episodes requested, {budget()['episodes_left']} left")
    smoke = _episode((g["scenes"][0], g["cfg"], 10_000, None, False))
    if "crash" in smoke["failures"] or any(f.startswith("stage_error") for f in smoke["failures"]):
        return {"gate": "smoke_test_failed", "episode": smoke, "budget_charged": 0}

    t0 = time.time()
    run_dir = RUNS / f"{int(t0)}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # video for the first seed of every scene, plus every failed episode
    by_id = {sc["id"]: sc for sc in g["scenes"]}
    if pairs:
        seen = set()
        jobs = []
        for sc, sd in pairs:
            jobs.append((by_id[sc], g["cfg"], sd, str(run_dir), sc not in seen))
            seen.add(sc)
    else:
        jobs = [(s, g["cfg"], seed0 + i, str(run_dir), i == 0) for s in g["scenes"] for i in range(episodes_per_scene)]
    # Sumo-based stages are load-sensitive (RS-011: walk outcomes changed under CPU contention), so they run
    # one episode at a time unless the method config or the caller says otherwise.
    sumo = "sumo" in str(g["cfg"]["stages"].get("build", ""))
    workers = max_workers or g["cfg"].get("max_workers") or (1 if sumo else MAX_WORKERS)
    with ProcessPoolExecutor(workers) as ex:
        eps = list(ex.map(_episode, jobs))
        wall = time.time() - t0
        # failed episodes without a video: replay them with frames (episodes are deterministic per seed)
        redo = [i for i, e in enumerate(eps) if not e["success"] and "video" not in e]
        by_scene = {s["id"]: s for s in g["scenes"]}
        for i, e in zip(redo, ex.map(_episode, [(by_scene[eps[i]["scene"]], g["cfg"], eps[i]["seed"], str(run_dir), True)
                                                for i in redo])):
            eps[i]["video"] = e.get("video")
    _charge(n)

    for episode in eps:
        episode["run_fingerprint"] = fingerprint
    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps))
    summary = metrics.summarize(eps) | {"wall_s": round(wall, 1), "workers": workers,
                                         "episodes_per_hour": round(n / wall * 3600)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "method.yaml").write_text((ROOT / "methods" / f"{method}.yaml").read_text())
    run_id = record.add("run", author, {
        "experiment_id": experiment_id, "method": method,
        "commit": f"{fingerprint['git_head'][:7]}+state.{fingerprint['id'][:8]}",
        "run_fingerprint": fingerprint, "scene_ids": scene_ids,
        "seeds": sorted({sd for _, sd in pairs}) if pairs else [seed0, seed0 + episodes_per_scene - 1], "episodes": n,
        "pairs": [list(x) for x in pairs] if pairs else None, "results_path": str(run_dir.relative_to(ROOT)),
        "final_eval": final_eval, "idealizations": g["cfg"].get("idealizations", []), "summary": summary})
    page = write_run_page(run_dir, run_id, method, experiment_id, eps, summary, fingerprint)
    videos = [{"scene": e["scene"], "seed": e["seed"], "success": e["success"], **e["video"]} for e in eps if e.get("video")]
    return {"run_id": run_id, "summary": summary, "run_page": page, "videos": videos}


def render_rollout(run_id: str, scene_id: str, seed: int, every_s: float = 1.0) -> dict:
    """Re-simulate one episode of a run with frames; save a contact sheet PNG and a GIF for the analyst."""
    from PIL import Image

    from lab import pipeline

    run = record.get(run_id)
    if run is None:
        raise ValueError(f"unknown run {run_id}")
    scene = scenes.load_scene(scene_id)
    if scene["split"] == "heldout" and not run["data"].get("final_eval"):
        raise RuntimeError("held-out scenes can only be rendered for the final evaluation run")
    cfg = yaml.safe_load((ROOT / run["data"]["results_path"] / "method.yaml").read_text())
    episode_path = ROOT / run["data"]["results_path"] / "episodes.jsonl"
    logged = None
    for line in episode_path.read_text().splitlines():
        row = json.loads(line)
        if row["scene"] == scene_id and int(row["seed"]) == int(seed):
            logged = row
            break
    if logged and logged.get("perceive", {}).get("parsed_blocks"):
        cfg = copy.deepcopy(cfg)
        cfg["stages"]["perceive"] = "logged_parse"
        cfg["perceive"]["logged_blocks"] = logged["perceive"]["parsed_blocks"]
    ep = pipeline.run_episode(scene, cfg, seed, frame_every_s=every_s)
    frames = ep.pop("_frames")
    out = ROOT / run["data"]["results_path"] / f"rollout_{scene_id}_s{seed}"
    rows = [np.concatenate(frames[i:i + 4] + [np.zeros_like(frames[0])] * (4 - len(frames[i:i + 4])), 1)
            for i in range(0, len(frames), 4)]
    Image.fromarray(np.concatenate(rows[:6], 0)).save(f"{out}.png")
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(f"{out}.gif", save_all=True, append_images=ims[1:], duration=int(every_s * 250), loop=0)
    return {"contact_sheet": f"{out}.png", "gif": f"{out}.gif", "episode": ep}


if __name__ == "__main__":
    import sys

    if sys.argv[1:] == ["freeze"]:
        freeze_eval()
        print("frozen", eval_hash())
