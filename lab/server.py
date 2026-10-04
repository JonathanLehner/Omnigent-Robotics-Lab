"""MCP server exposing the research record and lab equipment as tools (stdio).

Every agent launches its own instance; state lives in the SQLite record, so instances share it. Which tools an
agent may call is set by the `tools:` allow-list in its agents/**/config.yaml, not here.
"""

import json
import os
import subprocess
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml
from mcp.server.mcpserver import MCPServer

from lab import record, runner, scenes, sim

ROOT = Path(__file__).resolve().parent.parent
SUMO_DIR = Path(os.environ.get("SUMO_DIR", Path.home() / "src" / "sumo"))
PIXI = Path(os.environ.get("PIXI", Path.home() / ".pixi" / "bin" / "pixi"))
mcp = MCPServer("lab")


def _j(x):
    return json.loads(json.dumps(x, default=str))


def _fresh():
    """Reload lab code so long-lived tool servers always run what is on disk: no manual reload, ever.
    Order matters: pipeline re-creates its stage registry, then every stage plugin re-registers into it."""
    import importlib
    import sys

    from lab import method_models, pipeline, report, status

    for mod in (sim, scenes, method_models, pipeline):
        importlib.reload(mod)
    for name in sorted(n for n in sys.modules if n.startswith("lab.stages.")):
        importlib.reload(sys.modules[name])
    pipeline._load_plugins()
    for mod in (runner, status, report):
        importlib.reload(mod)


# --- research record ----------------------------------------------------------
@mcp.tool()
def add_evidence(author: str, source: str, claim: str, relevance: str, quote: str = "") -> str:
    """Record a cited fact. source must be a resolvable link or identifier (arXiv id, DOI, URL)."""
    return record.add("evidence", author, {"source": source, "claim": claim, "relevance": relevance, "quote": quote})


@mcp.tool()
def add_hypothesis(author: str, statement: str, predicted_effect: str, evidence_ids: list[str],
                   rung: int | None = None) -> str:
    """Record an agent-generated hypothesis (labeled with its author). Cite the evidence it rests on."""
    return record.add("hypothesis", author, {"statement": statement, "predicted_effect": predicted_effect,
                                             "evidence_ids": evidence_ids, "rung": rung, "status": "open",
                                             "label": f"agent hypothesis by {author}"})


@mcp.tool()
def set_hypothesis_status(author: str, hypothesis_id: str, status: str, reason: str, result_ids: list[str]) -> dict:
    """Move a hypothesis to supported | refuted | reopened | open. A surprising result should REOPEN an earlier one."""
    return _j(record.update(hypothesis_id, author, {"status": status, "reason": reason, "result_ids": result_ids}))


@mcp.tool()
def propose_experiment(author: str, hypothesis_ids: list[str], scene_ids: list[str], method: str, episodes: int,
                       expected_learning: str, cost: str, feasibility: str, alternative_to: list[str] | None = None) -> str:
    """Propose a candidate test. Propose at least two competing candidates per decision; link them with alternative_to."""
    return record.add("experiment", author, {"hypothesis_ids": hypothesis_ids, "scene_ids": scene_ids, "method": method,
                                             "episodes": episodes, "expected_learning": expected_learning, "cost": cost,
                                             "feasibility": feasibility, "alternative_to": alternative_to or [],
                                             "status": "candidate"})


@mcp.tool()
def select_experiment(author: str, experiment_id: str, rejected_ids: list[str], rationale: str) -> dict:
    """Select one candidate and reject its alternatives, scored on expected learning, feasibility and cost."""
    if not rejected_ids:
        raise ValueError("select between at least two candidates: list the rejected alternatives")
    for r in rejected_ids:
        record.update(r, author, {"status": "rejected", "rejected_for": experiment_id})
    return _j(record.update(experiment_id, author, {"status": "selected", "selection_rationale": rationale,
                                                    "rejected_ids": rejected_ids}))


@mcp.tool()
def log_result(author: str, run_ids: list[str], interpretation: str, failure_analysis: str,
               supports: list[str] | None = None, refutes: list[str] | None = None, surprise: str = "") -> str:
    """Analyst: interpret run(s). Metrics are copied from the runs (not typed in by hand)."""
    runs = [record.get(r) for r in run_ids]
    m = {r["id"]: r["data"]["summary"] for r in runs if r}
    return record.add("result", author, {"run_ids": run_ids, "metrics": m, "interpretation": interpretation,
                                         "failure_analysis": failure_analysis, "supports": supports or [],
                                         "refutes": refutes or [], "surprise": surprise, "review": None})


@mcp.tool()
def review_result(author: str, result_id: str, verdict: str, reasons: str) -> dict:
    """Reviewer: verdict accept | reject. Check baseline present, held-out untouched, enough runs, CI reported."""
    if verdict not in ("accept", "reject"):
        raise ValueError("verdict must be accept or reject")
    return _j(record.update(result_id, author, {"review": {"verdict": verdict, "reasons": reasons, "by": author}}))


@mcp.tool()
def record_decision(author: str, result_ids: list[str], change: str, rationale: str, next_experiment: str) -> str:
    """PI only: final decision on what changes next. Must cite reviewed result ids."""
    unreviewed = [r for r in result_ids if not (record.get(r) or {}).get("data", {}).get("review")]
    if unreviewed:
        raise ValueError(f"results {unreviewed} have no reviewer verdict yet")
    return record.add("decision", author, {"result_ids": result_ids, "change": change, "rationale": rationale,
                                           "next_experiment": next_experiment})


@mcp.tool()
def query_record(kind: str = "", contains: str = "", limit: int = 30) -> list:
    """List record objects, newest first. kind: evidence|scene|hypothesis|experiment|run|result|decision."""
    return _j(record.query(kind or None, contains or None, limit))


@mcp.tool()
def get_object(obj_id: str) -> dict:
    """Fetch one record object by id."""
    return _j(record.get(obj_id))


@mcp.tool()
def lab_status() -> str:
    """The plan and where the lab is: ladder x tiers grid (latest success, CI, run), unfinished experiments,
    hypotheses, latest decision, budget, recent run pages. Markdown; post it in chat so the human sees it."""
    _fresh()
    from lab import status

    return Path(status.build()).read_text()


@mcp.tool()
def export_record() -> dict:
    """Export the full record to record/export.json and build REPORT.md (for judges and reconstruction)."""
    _fresh()
    from lab import report

    return {"export": record.export(), "report": report.build()}


# --- literature ---------------------------------------------------------------
@mcp.tool()
def arxiv_search(query: str, max_results: int = 8) -> list:
    """Search arXiv. Returns id, title, published date, abstract."""
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": f"all:{query}", "max_results": max_results})
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(urllib.request.urlopen(url, timeout=30).read())
    return [{"id": e.findtext("a:id", "", ns).split("/abs/")[-1], "title": " ".join(e.findtext("a:title", "", ns).split()),
             "published": e.findtext("a:published", "", ns)[:10],
             "abstract": " ".join(e.findtext("a:summary", "", ns).split())[:1200]} for e in root.findall("a:entry", ns)]


@mcp.tool()
def openalex_search(query: str, max_results: int = 8) -> list:
    """Search OpenAlex works. Returns doi, title, year, citation count."""
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode({"search": query, "per-page": max_results})
    res = json.loads(urllib.request.urlopen(url, timeout=30).read())
    return [{"id": w["id"], "doi": w.get("doi"), "title": w.get("title"), "year": w.get("publication_year"),
             "cited_by": w.get("cited_by_count")} for w in res.get("results", [])]


# --- equipment ----------------------------------------------------------------
@mcp.tool()
def list_equipment() -> dict:
    """The research goal (question, tiers, ladder, success target, budget) and the equipment registry."""
    reg = yaml.safe_load((ROOT / "equipment.yaml").read_text())
    reg["goal"] = {"path": reg["goal"], **yaml.safe_load((ROOT / reg["goal"]).read_text())}
    return reg


@mcp.tool()
def list_scenes(tier: str = "", split: str = "dev") -> list:
    """Benchmark scenes. split=dev for development; held-out scenes are locked until final evaluation."""
    out = []
    for f in sorted((ROOT / "scenes" / split).glob("*.json")):
        s = json.loads(f.read_text())
        if not tier or s["tier"] == tier:
            out.append({"id": s["id"], "tier": s["tier"], "split": split, "structure": s["structure"],
                        "n_blocks": len(s["target"]["blocks"]), "image_path": s["image"]})
    return out


@mcp.tool()
def get_scene(scene_id: str) -> dict:
    """Full scene: oracle target spec, oracle order, start layout, target image path. Dev scenes only."""
    s = scenes.load_scene(scene_id)
    if s["split"] == "heldout":
        raise PermissionError("held-out scenes are locked")
    return s


@mcp.tool()
def make_dev_scene(author: str, tier: str, seed: int, permute_blocks: bool = False,
                   scene_id: str = "", layout: dict | None = None) -> dict:
    """Generate a DEV scene, optionally with an explicit id and {structure, blocks} target layout."""
    _fresh()
    sid = scene_id or f"{tier}-dev-x{seed}"
    sc = scenes.make_scene(sid, tier, "dev", seed=10_000 + seed, permute_blocks=permute_blocks,
                           layout=layout)
    data = {"tier": tier, "split": "dev", "structure": sc["structure"],
            "spec_path": f"scenes/dev/{sid}.json", "image_path": sc["image"],
            "n_blocks": len(sc["target"]["blocks"])}
    if permute_blocks:
        data["block_permutation"] = sc["block_permutation"]
    record.add("scene", author, data, obj_id=sid)
    return {"id": sid, "structure": sc["structure"], "image": sc["image"],
            **({"block_permutation": sc["block_permutation"]} if permute_blocks else {})}


@mcp.tool()
def check_stability(blocks: list[dict], seconds: float = 3.0) -> dict:
    """Settle test without the robot. blocks: [{id, type: cube|brick, color, pos: [x,y,z] structure frame, yaw}]."""
    _fresh()
    return sim.settle([scenes.to_world(b) for b in blocks], seconds)


@mcp.tool()
def list_methods() -> dict:
    """Method versions (methods/*.yaml) and the registered stage implementations."""
    _fresh()
    from lab import pipeline

    return {"methods": {p.stem: yaml.safe_load(p.read_text()) for p in sorted((ROOT / "methods").glob("*.yaml"))},
            "stages": {k: sorted(v) for k, v in pipeline.STAGES.items()}}


@mcp.tool()
def validate_method(method: str, scene_ids: list[str]) -> dict:
    """Run the pre-compute gates (config, prompts, frozen eval, held-out lock) without spending budget."""
    _fresh()
    g = runner.gates(method, scene_ids, final_eval=False)
    return {"ok": True, "stages": g["cfg"]["stages"], "idealizations": g["cfg"].get("idealizations", [])}


@mcp.tool()
def budget_status() -> dict:
    """Episode budget for the session: total, used, left."""
    _fresh()
    return runner.budget()


@mcp.tool()
def run_sim_batch(experiment_id: str, scene_ids: list[str], method: str, episodes_per_scene: int = 3,
                  seed0: int = 0, final_eval: bool = False, max_workers: int = 0, pairs: list[list] | None = None) -> dict:
    """Runner only: run a SELECTED experiment. Gates + one-episode smoke test first (free), then parallel episodes.
    final_eval=True unlocks held-out scenes and requires human approval. max_workers: parallel episodes (0 = default:
    8, or 1 for Sumo-based methods, which are load-sensitive); set 1 to run serially. pairs: explicit
    [[scene_id, seed], ...] for designs that are not scenes x consecutive seeds (one batch instead of many calls)."""
    _fresh()
    return _j(runner.run_sim_batch(experiment_id, scene_ids, method, episodes_per_scene, seed0, final_eval=final_eval,
                                   max_workers=max_workers or None, pairs=pairs))


@mcp.tool()
def render_rollout(run_id: str, scene_id: str, seed: int) -> dict:
    """Re-simulate one episode of a run with frames. Returns a contact-sheet PNG + GIF path to inspect visually."""
    _fresh()
    return _j(runner.render_rollout(run_id, scene_id, seed))


@mcp.tool()
def compute_metrics(run_ids: list[str]) -> dict:
    """Pool episodes across runs and recompute metrics (success rate with 95% Wilson CI, failures, per tier)."""
    _fresh()
    from lab import metrics

    eps = []
    for r in run_ids:
        eps += [json.loads(line) for line in (ROOT / record.get(r)["data"]["results_path"] / "episodes.jsonl").read_text().splitlines()]
    return metrics.summarize(eps)


@mcp.tool()
def call_model(model: str, prompt_path: str, text: str = "", image: str = "") -> dict:
    """Method model call (codex | claude-haiku | claude-sonnet ...) on a versioned prompt; cached and logged."""
    from lab.method_models import call_model as cm

    return cm(model, prompt_path, text, image or None)


@mcp.tool()
def eval_prompt(model: str, prompt_path: str, scene_ids: list[str], image: str = "main") -> dict:
    """Cheap offline test without simulation: parse each dev scene's target image and score against the oracle spec
    (block count, color/type match, position error after matching). image: main | multiview (2x2 labeled views:
    main, top, robot side, left side)."""
    _fresh()
    import numpy as np

    from lab.method_models import call_model as cm

    rows = []
    for sid in scene_ids:
        sc = get_scene(sid)
        try:
            out = cm(model, prompt_path, image=str(ROOT / sc["image" if image == "main" else "image_multiview"]))["output"]["blocks"]
        except Exception as e:  # noqa: BLE001
            rows.append({"scene": sid, "error": str(e)[:300]})
            continue
        truth = sc["target"]["blocks"]
        free, errs, matched = list(truth), [], 0
        for b in out:
            hit = next((t for t in free if t["color"] == b.get("color") and t["type"] == b.get("type")), None)
            if hit:
                free.remove(hit)
                matched += 1
                errs.append(float(np.linalg.norm(np.subtract(hit["pos"], b["pos"]))))
        rows.append({"scene": sid, "n_true": len(truth), "n_pred": len(out), "matched": matched,
                     "exact_count": len(out) == len(truth), "max_pos_err_cm": round(max(errs) * 100, 1) if errs else None})
    ok = [r for r in rows if "error" not in r]
    return {"per_scene": rows, "count_accuracy": sum(r["exact_count"] and r["matched"] == r["n_true"] for r in ok) / max(1, len(rows)),
            "within_3cm": sum(r["max_pos_err_cm"] is not None and r["max_pos_err_cm"] <= 3 for r in ok) / max(1, len(rows))}


@mcp.tool()
def run_sumo_task(task: str, episodes: int = 1, episode_length_s: float = 10.0, optimizer: str = "cem",
                  task_module: str = "", num_rollouts: int = 0, video_dir: str = "") -> dict:
    """Run Sumo MPC (Relic whole-body policy in the loop) headless in its pixi env. task_module: path to an
    agent-written module that registers new tasks (staged costs). ~2.5x slower than real time on this Mac."""
    cmd = [str(PIXI), "run", "--manifest-path", str(SUMO_DIR / "pyproject.toml"), "python",
           str(ROOT / "lab" / "sumo_adapter" / "run_task.py"), "--task", task, "--episodes", str(episodes),
           "--episode-length-s", str(episode_length_s), "--optimizer", optimizer]
    if task_module:
        cmd += ["--task-module", str((ROOT / task_module).resolve())]
    if num_rollouts:
        cmd += ["--num-rollouts", str(num_rollouts)]
    if video_dir:
        cmd += ["--video-dir", str((ROOT / video_dir).resolve())]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=SUMO_DIR)
    last = [line for line in res.stdout.splitlines() if line.startswith("{")]
    if res.returncode or not last:
        return {"error": res.stderr[-3000:]}
    return json.loads(last[-1])


if __name__ == "__main__":
    mcp.run()
