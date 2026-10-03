"""Build REPORT.md from the research record: the discovery loop as it actually happened, with links between ids."""

from pathlib import Path

from lab import record

ROOT = Path(__file__).resolve().parent.parent


def _fmt_rate(summary):
    lo, hi = summary["success_ci95"]
    return f"{summary['success_rate']:.2f} [{lo:.2f}, {hi:.2f}] (n={summary['episodes']})"


def build(path: Path = ROOT / "REPORT.md") -> str:
    objs = list(reversed(record.query(limit=100000)))
    by_kind: dict[str, list] = {}
    for o in objs:
        by_kind.setdefault(o["kind"], []).append(o)
    lines = ["# Lab record report", "", "Generated from `record/lab.db` (full export: `record/export.json`).", ""]

    lines += ["## Decisions", ""]
    for d in by_kind.get("decision", []):
        x = d["data"]
        lines += [f"- **{d['id']}** ({d['author']}), cites {', '.join(x['result_ids'])}: {x['change']}",
                  f"  - rationale: {x['rationale']}", f"  - next: {x.get('next_experiment', '')}"]

    lines += ["", "## Results", ""]
    for r in by_kind.get("result", []):
        x = r["data"]
        rates = "; ".join(f"{rid}: {_fmt_rate(m)}" for rid, m in x["metrics"].items())
        review = x.get("review") or {}
        lines += [f"- **{r['id']}** runs {rates}", f"  - interpretation: {x['interpretation']}",
                  f"  - failures: {x.get('failure_analysis', '')}",
                  f"  - review: {review.get('verdict', 'pending')} by {review.get('by', '-')}: {review.get('reasons', '')}"]
        if x.get("surprise"):
            lines.append(f"  - surprise: {x['surprise']}")

    lines += ["", "## Runs", "", "| run | experiment | method | commit | episodes | success [95% CI] | eps/hour | idealizations |",
              "|---|---|---|---|---|---|---|---|"]
    for r in by_kind.get("run", []):
        x = r["data"]
        s = x["summary"]
        lines.append(f"| {r['id']}{' (FINAL held-out)' if x.get('final_eval') else ''} | {x['experiment_id']} | {x['method']} "
                     f"| {x['commit']} | {x['episodes']} | {_fmt_rate(s)} | {s.get('episodes_per_hour')} | "
                     f"{', '.join(x.get('idealizations', []))} |")

    lines += ["", "## Experiments (candidates, selected, rejected)", ""]
    for e in by_kind.get("experiment", []):
        x = e["data"]
        lines.append(f"- **{e['id']}** [{x.get('status')}] tests {', '.join(x['hypothesis_ids'])} with {x['method']} on "
                     f"{len(x['scene_ids'])} scenes, {x['episodes']} episodes. Expected learning: {x['expected_learning']}. "
                     f"Cost: {x['cost']}. {x.get('selection_rationale', '')}")

    lines += ["", "## Hypotheses (agent-generated, labeled by author)", ""]
    for h in by_kind.get("hypothesis", []):
        x = h["data"]
        lines.append(f"- **{h['id']}** [{x.get('status')}] by {h['author']}: {x['statement']} "
                     f"(predicted: {x['predicted_effect']}; evidence: {', '.join(x.get('evidence_ids', [])) or 'none'})")

    lines += ["", "## Evidence", ""]
    for e in by_kind.get("evidence", []):
        x = e["data"]
        lines.append(f"- **{e['id']}** {x['source']}: {x['claim']} (relevance: {x['relevance']})")

    path.write_text("\n".join(lines) + "\n")
    return str(path)


if __name__ == "__main__":
    record.export()
    print(build())
