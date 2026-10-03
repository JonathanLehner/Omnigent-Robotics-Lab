"""Batch evaluation: gates, then parallel episodes, then a Run object in the record.

Gates (run before any budget is spent): method config loads and names registered stages, prompt files and
schemas parse, eval code matches its frozen hash, held-out scenes only with final_eval, a one-episode smoke
test completes. Episodes run in parallel processes (one per core, capped).
"""

import hashlib
import json
import os
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from lab import metrics, record, scenes

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
FROZEN = ROOT / "record" / "frozen_eval.sha256"
BUDGET = record.DB_PATH.parent / "budget.json"  # per record, so the single-agent baseline has its own
MAX_WORKERS = max(1, min(8, (os.cpu_count() or 2) - 2))


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


def git_version() -> str:
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    head = git("rev-parse", "--short", "HEAD") or "nocommit"
    diff = git("diff", "HEAD", "--", "lab", "methods", "prompts")
    return head + (f"+dirty.{hashlib.sha256(diff.encode()).hexdigest()[:8]}" if diff else "")


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
    held = [s["id"] for s in loaded if s["split"] == "heldout"]
    if held and not final_eval:
        raise RuntimeError(f"gate:heldout: {held} are held-out; only the human-approved final evaluation may use them")
    return {"cfg": cfg, "scenes": loaded}


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
              "failures": ["crash"], "error": traceback.format_exc()[-1500:]}
    frames = ep.pop("_frames", None)
    if frames:
        ep["video"] = save_video(frames, Path(video_dir) / f"{scene['id']}_s{seed}")
    ep["wall_s"] = round(time.time() - t, 2)
    return ep


def write_run_page(run_dir: Path, run_id: str, method: str, experiment_id: str, eps: list, summary: dict) -> str:
    """runs/<run>/RUN.md: summary + embedded episode videos, viewable in the Omnigent web UI file viewer."""
    lo, hi = summary["success_ci95"]
    lines = [f"# {run_id}: {method} ({experiment_id})", "",
             f"Success {summary['success_rate']:.2f} (95% CI {lo:.2f}-{hi:.2f}), n={summary['episodes']}, "
             f"median placement error {summary['median_pos_err_cm']:.1f} cm, failures {summary['failure_categories'] or 'none'}.", "",
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
                  seed0: int = 0, author: str = "experiment_runner", final_eval: bool = False) -> dict:
    exp = record.get(experiment_id)
    if exp is None or exp["kind"] != "experiment":
        raise ValueError(f"{experiment_id!r} is not an experiment in the record")
    if exp["data"].get("status") != "selected" and not final_eval:
        raise ValueError(f"experiment {experiment_id} has status {exp['data'].get('status')!r}; the planner must select it first")
    g = gates(method, scene_ids, final_eval)
    n = len(scene_ids) * episodes_per_scene
    done = sum(r["data"]["episodes"] for r in record.query("run", contains=f'"experiment_id": "{experiment_id}"', limit=1000))
    if not final_eval and done + n > int(exp["data"]["episodes"]):
        raise RuntimeError(f"experiment {experiment_id} planned {exp['data']['episodes']} episodes; {done} already run, "
                           f"{n} more requested. Propose a new experiment for more.")
    if n > budget()["episodes_left"]:
        raise RuntimeError(f"budget: {n} episodes requested, {budget()['episodes_left']} left")
    smoke = _episode((g["scenes"][0], g["cfg"], 10_000, None, False))
    if "crash" in smoke["failures"] or any(f.startswith("stage_error") for f in smoke["failures"]):
        return {"gate": "smoke_test_failed", "episode": smoke, "budget_charged": 0}

    t0 = time.time()
    run_dir = RUNS / f"{int(t0)}_{method}"
    run_dir.mkdir(parents=True, exist_ok=True)
    # video for the first seed of every scene, plus every failed episode
    jobs = [(s, g["cfg"], seed0 + i, str(run_dir), i == 0) for s in g["scenes"] for i in range(episodes_per_scene)]
    with ProcessPoolExecutor(MAX_WORKERS) as ex:
        eps = list(ex.map(_episode, jobs))
        wall = time.time() - t0
        # failed episodes without a video: replay them with frames (episodes are deterministic per seed)
        redo = [i for i, e in enumerate(eps) if not e["success"] and "video" not in e]
        by_scene = {s["id"]: s for s in g["scenes"]}
        for i, e in zip(redo, ex.map(_episode, [(by_scene[eps[i]["scene"]], g["cfg"], eps[i]["seed"], str(run_dir), True)
                                                for i in redo])):
            eps[i]["video"] = e.get("video")
    _charge(n)

    (run_dir / "episodes.jsonl").write_text("\n".join(json.dumps(e) for e in eps))
    summary = metrics.summarize(eps) | {"wall_s": round(wall, 1), "workers": MAX_WORKERS,
                                         "episodes_per_hour": round(n / wall * 3600)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (run_dir / "method.yaml").write_text((ROOT / "methods" / f"{method}.yaml").read_text())
    run_id = record.add("run", author, {
        "experiment_id": experiment_id, "method": method, "commit": git_version(), "scene_ids": scene_ids,
        "seeds": [seed0, seed0 + episodes_per_scene - 1], "episodes": n, "results_path": str(run_dir.relative_to(ROOT)),
        "final_eval": final_eval, "idealizations": g["cfg"].get("idealizations", []), "summary": summary})
    page = write_run_page(run_dir, run_id, method, experiment_id, eps, summary)
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
