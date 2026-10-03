# Omnigent Robotics Lab

An agentic, simulation-first robotics research lab built on [Omnigent](https://github.com/omnigent-ai/omnigent)
(Hack-Nation Challenge 03). A PI agent and eight specialist agents take a robotics research goal, design simulated
experiments, implement a method from installed equipment, test it and decide what to investigate next. Every
handoff is a record object, so each decision can be traced back to its evidence.

**Demo question:** How can a Spot robot plan and assemble a target structure, given only a picture of it, from
handled building blocks, without task-specific training?

Everything runs locally on a MacBook (Apple M4, CPU only): MuJoCo, Sumo's policy-in-the-loop MPC, and the agents
through Omnigent with the Claude Code and Codex logins already on the machine. No API keys.

## Quick start

```bash
./lab.sh setup        # uv env, Omnigent + lab policies, 40 benchmark scenes, frozen eval hash
./lab.sh run          # pursues the goal file autonomously, cycle after cycle (watch at http://localhost:6767)
./lab.sh run -p "..."  # same lab, your own instruction instead
./lab.sh report       # record/export.json + REPORT.md
```

Prerequisites: `uv`, `tmux`, Node 22+, `claude` and `codex` CLIs logged in. Optional: Sumo in `~/src/sumo`
(`pixi install && pixi run build`) for walking, real grasps and MPC placement (rungs 2-4).

## How it is organized

| Layer | Runs | Where |
|---|---|---|
| Relic whole-body policy | 50 Hz | Sumo (pixi env), via `run_sumo_task` |
| Sumo sampling MPC | 20 Hz | `lab/sumo_adapter/run_task.py` |
| Assembly method (what the lab develops) | once per build | `lab/pipeline.py`, `lab/stages/`, `methods/*.yaml`, `prompts/` |
| **Omnigent research lab** | once per experiment | `agents/assembly_lab/` |

Goal-specific parts are plug-ins: `goals/spot_assembly.yaml` (question, tiers, ladder, budget),
`equipment.yaml` (registry the agents read), `methods/v0.yaml` (working start), `scenes/` (benchmark).

### Agents (`agents/assembly_lab/`)

| Agent | Decision it owns | Harness | Lab tools (allow-listed per agent) |
|---|---|---|---|
| PI (orchestrator) | goal, budget, final decisions citing reviewed results | claude-sdk | query_record, record_decision, budget_status, ... |
| literature_scout | what is known | claude-sdk | arxiv_search, openalex_search, add_evidence |
| scene_designer | what the robot is tested on | claude-sdk (Claude Code) + shell | make_dev_scene, check_stability, list/get_scene |
| method_designer | which approach, what to change next | claude-sdk + files | add_hypothesis, eval_prompt, call_model, validate_method |
| experiment_planner | which test next (>= 2 candidates, scored) | claude-sdk | propose_experiment, select_experiment, budget_status |
| robotics_engineer | how the method is implemented | codex + shell | validate_method, run_sumo_task, check_stability |
| experiment_runner | running it correctly | claude-sdk | run_sim_batch |
| analyst | what results mean | claude-sdk | compute_metrics, render_rollout, log_result, set_hypothesis_status |
| reviewer | whether a claim is justified | codex (different model family, no file access) | review_result, compute_metrics |

The tools are one MCP server (`lab/server.py`, 28 tools). Each agent starts its own stdio instance with a `tools:`
allow-list, so "only the runner can run batches" and "only the PI records decisions" are enforced by config.

### Research record (`lab/record.py`)

SQLite with typed, linked objects: evidence (E), scene, hypothesis (H, status open/supported/refuted/reopened,
labeled by author), experiment (X, candidate/selected/rejected), run (R, commit + seeds + results path), result
(RS, metrics copied from runs, reviewer verdict), decision (D, must cite reviewed results). Dangling ids are
rejected. `./lab.sh report` renders the loop as it happened.

### Policies and gates (human approval boundary)

| Boundary | Omnigent policy | Also enforced in the tool layer |
|---|---|---|
| Held-out scenes locked | `lab.policies.protect_heldout_set` (DENY; ASK for the final evaluation) | `run_sim_batch` gate, `get_scene` |
| Large batches | `lab.policies.approve_large_batch` (ASK above 150 episodes) | episode budget (`record/budget.json`) |
| Frozen eval code | `lab.policies.freeze_eval_code` (ASK on edits to metrics/goals/policies) | `run_sim_batch` refuses if `lab/metrics.py` hash changed |
| Spend / runaway loops | built-in `cost_budget`, `max_tool_calls_per_session` | |
| Decisions cite results | `lab.policies.pi_cites_results` | `record_decision` requires reviewer verdicts |
| Two candidate tests | | `select_experiment` requires rejected alternatives; runs need a selected experiment |

Before any budget is spent, a batch passes gates: method config loads, prompts/schemas parse, frozen-eval hash
matches, held-out lock, and a one-episode smoke test.

## The method and the de-idealization ladder

Method v0 (rung 0) uses oracles everywhere and works from the start: oracle structure spec, oracle build order,
robot placed at the site (base welded), idealized weld grasp, scripted arm motion (damped-least-squares IK on the
Spot arm), no arm-block contacts. Each rung replaces one oracle; a rung counts only when the method still meets
the success target on that tier. Success criteria are frozen in `lab/metrics.py`: each block within 3 cm and
10 deg of its target, 5 s after the last release; success rates are reported with 95% Wilson intervals.

| Rung | Replaces | With | Equipment ready |
|---|---|---|---|
| 1 | oracle build order | `support_sort` (symbolic + settle test) or `llm` | yes |
| 2 | robot placed at site | walking with Relic via Sumo | adapter + Sumo built |
| 3 | weld grasp | real gripper grasp on the handle | Sumo / MuJoCo contact |
| 4 | scripted placement | MPC with an agent-written cost | adapter (`task_module`) |
| 5 | oracle structure spec | `perceive: vlm` (codex or claude) on the target picture | yes, `eval_prompt` |

## Measured so far (setup checks, before the agents start)

- Sumo builds and runs on macOS arm64; one Spot MPC episode (24 rollouts, Relic policy in the loop) runs about
  2.6x slower than real time. Upstream `sumo.run_mpc` drops the Spot rollout backend; the adapter fixes it.
- v0 on all 24 dev scenes, 2 seeds each: success 0.98 (95% CI 0.89-1.00), median placement error 2.1 cm
  (gravity sag of the arm under load, close to the 3 cm tolerance), 48 episodes in 10.6 s on 8 workers.
- Perception prompt v1 with codex: correct block count, colors and types on the T2 bridge and 6-block pyramid
  tested so far, 7-12 s per uncached call.

## Honest limitations

- Simulation only. Rung 0 includes a welded base and a weld grasp; every result lists the idealizations in use.
- Target pictures are clean renders from a fixed viewpoint; a real camera image is a harder perception problem.
- `claude-sdk` agents load the user's `~/.claude` plugins and hooks; run the lab with a clean Claude config for a
  controlled comparison.
- Omnigent's native tmux-based Codex/Claude terminals did not start reliably here; the lab uses the headless
  `claude-sdk` and `codex` harnesses instead.
- Validation before real hardware: real Spot grasp contact, perception from real images, sim-to-real gap.

## Layout

```
agents/assembly_lab/           PI bundle (config.yaml, AGENTS.md) + agents/<role>/config.yaml
agents/single_agent_baseline/  control: one agent, same tools and budget, separate record
lab/  record.py server.py policies.py runner.py pipeline.py sim.py scenes.py metrics.py method_models.py report.py
lab/sumo_adapter/run_task.py   runs inside Sumo's pixi env
methods/ prompts/ goals/ scenes/{dev,heldout}/ equipment.yaml record/ runs/
```
