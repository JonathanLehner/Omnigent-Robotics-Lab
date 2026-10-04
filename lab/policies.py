"""Custom Omnigent policies for the lab (stdlib only: they run inside Omnigent's server/runner process).

The same boundaries are also enforced in the tools themselves (runner gates), so a harness whose native tool
calls bypass Omnigent's policy hooks still cannot run held-out scenes or change the frozen eval silently.
"""

import json

WRITE_HINTS = ("write", "edit", "shell", "bash", "patch", "apply", "exec", "multiedit", "notebook")
HELDOUT_HINTS = ("-ho-", "scenes/heldout", "heldout/")


def _call(event):
    if event.get("type") != "tool_call":
        return None, None, ""
    data = event.get("data") or {}
    name = str(data.get("name") or event.get("target") or "")
    args = data.get("arguments") or {}
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            args = {"raw": args}
    return name, args, json.dumps(args, default=str).lower()


def protect_heldout_set(event):
    """DENY any tool call touching held-out scenes, except the final evaluation batch, which ASKs a human."""
    name, args, blob = _call(event)
    if name is None or not any(h in blob for h in HELDOUT_HINTS) and not args.get("final_eval"):
        return None
    if name.endswith("run_sim_batch") and args.get("final_eval"):
        return {"result": "ASK", "reason": "Final evaluation on the locked held-out scenes. Approve only once the "
                                           "method is frozen; this is the number reported to judges."}
    return {"result": "DENY", "reason": "Held-out scenes are locked until the human-approved final evaluation."}


def approve_large_batch(max_episodes: int = 120):
    """Factory: experiments planned above max_episodes need human sign-off before their first batch.

    Counted per experiment, not per call, so splitting an experiment into small batches does not bypass it
    (the runner also refuses episodes beyond the experiment's planned total)."""

    def evaluate(event):
        name, args, _ = _call(event)
        if not name or not name.endswith("run_sim_batch"):
            return None
        n = len(args.get("scene_ids") or []) * int(args.get("episodes_per_scene", 3))
        planned, started = n, False
        try:
            from lab import record  # stdlib-only module

            exp = record.get(str(args.get("experiment_id")))
            planned = max(n, int((exp or {}).get("data", {}).get("episodes", 0)))
            started = bool(record.query("run", contains=f'"experiment_id": "{args.get("experiment_id")}"', limit=1))
        except Exception:  # noqa: BLE001 - fall back to per-call size
            pass
        if n > max_episodes or (planned > max_episodes and not started):
            return {"result": "ASK", "reason": f"Experiment {args.get('experiment_id')} plans {planned} episodes "
                                               f"(this batch {n}), above {max_episodes}; approve the compute?"}
        return {"result": "ALLOW"}

    return evaluate


def freeze_eval_code(paths: tuple = ("lab/metrics.py", "frozen_eval.sha256", "goals/", "lab/policies.py")):
    """Factory: writes or shell commands that mention success criteria / eval / policy files need human sign-off."""

    def evaluate(event):
        name, _, blob = _call(event)
        if not name or not any(h in name.lower() for h in WRITE_HINTS):
            return None
        hit = next((p for p in paths if p.lower() in blob), None)
        if hit:
            return {"result": "ASK", "reason": f"{name} touches frozen evaluation/policy file {hit!r}; approve?"}
        return None

    return evaluate


def read_only_workspace(event):
    """For agents that get a workspace only so the web UI can show files and videos: deny writes and shell."""
    name, _, _ = _call(event)
    if name and any(h in name.lower() for h in WRITE_HINTS) and not name.lower().endswith("read"):
        return {"result": "DENY", "reason": "This agent's workspace is read-only (for viewing runs and reports)."}
    return None


def no_record_bypass(event):
    """For agents with a shell: experiments and runs go through the planner/runner tools, not direct Python calls."""
    name, _, blob = _call(event)
    if name and any(h in name.lower() for h in WRITE_HINTS) and any(
            k in blob for k in ("run_sim_batch", "record.add", "propose_experiment", "select_experiment", "lab.runner")):
        return {"result": "DENY", "reason": "Experiments and runs are created only by experiment_planner / "
                                            "experiment_runner through their tools. Use pipeline.run_episode for smoke tests."}
    return None


def pi_cites_results(event):
    """record_decision must cite at least one result id (the tool re-checks that each result was reviewed)."""
    name, args, _ = _call(event)
    if name and name.endswith("record_decision") and not args.get("result_ids"):
        return {"result": "DENY", "reason": "A decision must cite the reviewed result ids it rests on."}
    return None


POLICY_REGISTRY = [
    {"handler": "lab.policies.protect_heldout_set", "kind": "callable", "name": "Protect held-out scenes",
     "description": "Deny held-out scene access; ask before the final evaluation."},
    {"handler": "lab.policies.approve_large_batch", "kind": "factory", "name": "Approve large sim batches",
     "description": "Ask before batches above a size.",
     "params_schema": {"type": "object", "properties": {"max_episodes": {"type": "integer"}}}},
    {"handler": "lab.policies.freeze_eval_code", "kind": "factory", "name": "Freeze evaluation code",
     "description": "Ask before edits to success criteria, eval or policy files."},
    {"handler": "lab.policies.pi_cites_results", "kind": "callable", "name": "Decisions cite results",
     "description": "Deny decisions without result ids."},
]


if __name__ == "__main__":
    ev = lambda n, a: {"type": "tool_call", "target": n, "data": {"name": n, "arguments": a}}  # noqa: E731
    assert protect_heldout_set(ev("lab__get_scene", {"scene_id": "T2-ho-01"}))["result"] == "DENY"
    assert protect_heldout_set(ev("lab__run_sim_batch", {"scene_ids": ["T2-ho-01"], "final_eval": True}))["result"] == "ASK"
    assert protect_heldout_set(ev("lab__get_scene", {"scene_id": "T2-dev-01"})) is None
    assert protect_heldout_set(ev("Read", {"file_path": "/x/scenes/heldout/T1-ho-02.json"}))["result"] == "DENY"
    big = approve_large_batch(100)
    assert big(ev("mcp__lab__run_sim_batch", {"experiment_id": "X-none", "scene_ids": ["a"] * 40, "episodes_per_scene": 3}))["result"] == "ASK"
    assert big(ev("mcp__lab__run_sim_batch", {"experiment_id": "X-none", "scene_ids": ["a"] * 4, "episodes_per_scene": 3}))["result"] == "ALLOW"
    fz = freeze_eval_code()
    assert fz(ev("Edit", {"file_path": "lab/metrics.py"}))["result"] == "ASK"
    assert fz(ev("Edit", {"file_path": "lab/pipeline.py"})) is None
    assert fz(ev("Read", {"file_path": "lab/metrics.py"})) is None
    assert pi_cites_results(ev("lab__record_decision", {"result_ids": []}))["result"] == "DENY"
    assert read_only_workspace(ev("sys_os_shell", {"command": "ls"}))["result"] == "DENY"
    assert read_only_workspace(ev("sys_os_read", {"path": "runs/x/RUN.md"})) is None
    assert read_only_workspace(ev("Bash", {"command": "rm x"}))["result"] == "DENY"
    assert read_only_workspace(ev("lab__run_sim_batch", {})) is None
    assert no_record_bypass(ev("exec_command", {"cmd": "uv run python -c 'from lab import runner; runner.run_sim_batch(...)'"}))["result"] == "DENY"
    assert no_record_bypass(ev("exec_command", {"cmd": "uv run python -c 'from lab import pipeline'"})) is None
    print("policies ok")
