"""LAB_STATUS.md: the plan (ladder x tiers) and where the lab is in it. Rebuilt on every record write;
open it in the Omnigent web UI file viewer."""

import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
TIERS = ["T0", "T1", "T2", "T3"]
CAPABILITY = {0: "Baseline (all oracles)", 1: "Planned build order", 2: "Walking approach",
              3: "Real gripper grasp", 4: "Lab-chosen placement (MPC)", 5: "Structure from picture"}


def _rung_of(run) -> int | None:
    try:
        return yaml.safe_load((ROOT / run["data"]["results_path"] / "method.yaml").read_text()).get("rung")
    except (OSError, AttributeError):
        return None


def build(path: Path = ROOT / "LAB_STATUS.md") -> str:
    from lab import record

    objs = list(reversed(record.query(limit=100000)))  # oldest first
    kind = lambda k: [o for o in objs if o["kind"] == k]  # noqa: E731
    goal = yaml.safe_load((ROOT / "goals" / "spot_assembly.yaml").read_text())
    runs = [r for r in kind("run") if not r["data"].get("final_eval")]

    # latest run per (rung, tier)
    cell = {}
    for r in runs:
        rung = _rung_of(r)
        for tier, v in r["data"]["summary"].get("by_tier", {}).items():
            if rung is not None and v["n"] >= 6:
                cell[(rung, tier)] = (r, v)

    done_eps = {}
    for r in kind("run"):
        done_eps[r["data"]["experiment_id"]] = done_eps.get(r["data"]["experiment_id"], 0) + r["data"]["episodes"]
    running = set()
    for e in kind("experiment"):
        if e["data"].get("status") == "selected" and done_eps.get(e["id"], 0) < int(e["data"]["episodes"]):
            try:
                name = re.match(r"[\w.-]+", e["data"]["method"]).group(0)  # method field is often prose
                rung = yaml.safe_load((ROOT / "methods" / f"{name}.yaml").read_text())["rung"]
                running |= {(rung, sid[:2]) for sid in e["data"]["scene_ids"]}
            except (OSError, KeyError, TypeError, IndexError, AttributeError):
                pass

    lines = ["## Lab status", "", f"*{goal['question'].strip()}*", "",
             "| Rung | Capability | T0 | T1 | T2 | T3 | Status |",
             "|:--:|:--|:--:|:--:|:--:|:--:|:--|"]
    from lab.metrics import wilson_ci

    evidence = []
    for step in goal["de_idealization_ladder"]:
        rung = step["rung"]
        vals, passed, failed = [], [], []
        for t in TIERS:
            if (rung, t) not in cell:
                vals.append("🔵 running" if (rung, t) in running else "⚪ –")
                continue
            r, v = cell[(rung, t)]
            lo, hi = wilson_ci(v["success"], v["n"])
            ok = v["rate"] >= 0.8
            (passed if ok else failed).append(t)
            mark = "🔵" if (rung, t) in running else ("🟢" if ok else "🔴")
            vals.append(f"{mark} {v['rate']:.2f}")
            evidence.append(f"R{rung}/{t} {v['rate']:.2f} [{lo:.2f}, {hi:.2f}] n={v['n']} ({r['id']})")
        active = [t for t in TIERS if (rung, t) in running]
        if active:
            state = "🔵 Running: " + ", ".join(active)
        elif not passed and not failed:
            state = "⚪ Not started"
        elif not failed and len(passed) == len(TIERS):
            state = "🟢 Passed"
        elif failed:
            state = "🔴 Below target: " + ", ".join(failed)
        else:
            state = "🟢 Passed so far"
        lines.append(f"| {rung} | {CAPABILITY.get(rung, step['with'])} | " + " | ".join(vals) + f" | {state} |")
    lines += ["", "🟢 passed (>= 0.80)  🔵 running now  🔴 below target  ⚪ not started. Success rate on dev scenes, latest run per cell.", "",
              "<details><summary>95% CIs and runs</summary>", "", "; ".join(evidence), "", "</details>"]

    exps = kind("experiment")
    done = {}
    for r in kind("run"):
        done[r["data"]["experiment_id"]] = done.get(r["data"]["experiment_id"], 0) + r["data"]["episodes"]
    active = [e for e in exps if e["data"].get("status") == "selected" and done.get(e["id"], 0) < int(e["data"]["episodes"])]
    lines += ["", "## Now (selected experiments not yet fully run)", ""]
    for e in active:
        lines.append(f"- **{e['id']}** ({done.get(e['id'], 0)}/{e['data']['episodes']} episodes run): "
                     f"{e['data']['method'][:160]}. Tests {', '.join(e['data']['hypothesis_ids']) or '-'}.")
    if not active:
        lines.append("- nothing queued; the PI is analysing, reviewing or deciding")
    hyps = kind("hypothesis")
    lines += ["", "## Hypotheses", "", "| id | rung | status | statement |", "|---|---|---|---|"]
    for h in hyps:
        st = h["data"].get("status")
        lines.append(f"| {h['id']} | {h['data'].get('rung', '')} | {st} | {h['data']['statement'][:140]} |")

    decisions = kind("decision")
    if decisions:
        d = decisions[-1]
        lines += ["", f"## Latest decision ({d['id']}, cites {', '.join(d['data']['result_ids'])})", "",
                  d["data"]["change"][:600], "", f"**Next:** {d['data'].get('next_experiment', '')[:400]}"]

    try:
        from lab import runner

        b = runner.budget()
        lines += ["", f"**Budget:** {b['episodes_used']} of {b['episodes_total']} episodes used, {b['episodes_left']} left. "
                      f"Record: {len(kind('evidence'))} evidence, {len(hyps)} hypotheses, {len(exps)} experiments, "
                      f"{len(kind('run'))} runs, {len(kind('result'))} results, {len(decisions)} decisions."]
    except Exception:  # noqa: BLE001 - status must never break a record write
        pass

    lines += ["", "## Recent runs (videos inside)", ""]
    for r in runs[-8:][::-1]:
        s = r["data"]["summary"]
        page = Path(r["data"]["results_path"]) / "RUN.md"
        link = f"[{page}]({page})" if (ROOT / page).exists() else str(r["data"]["results_path"])
        vids = sorted((ROOT / r["data"]["results_path"]).glob("*.gif"))[:3]
        vlinks = " ".join(f"[▶ {v.stem}]({v.relative_to(ROOT)})" for v in vids)
        lines.append(f"- {r['id']} {r['data']['method']} (rung {_rung_of(r)}), {', '.join(s.get('by_tier', {}))}: "
                     f"{s['success_rate']:.2f}, failures {json.dumps(s['failure_categories'])} - {link} {vlinks}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


if __name__ == "__main__":
    print(build())
