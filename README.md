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
./lab.sh setup         # uv env, Omnigent + lab policies, benchmark scenes, frozen eval hash
./lab.sh run           # pursue goals/spot_assembly.yaml autonomously; restarts continue the same chat
./lab.sh run --new     # same, but in a fresh Omnigent session
./lab.sh run -p "..."  # same lab, your own instruction instead
./lab.sh report        # record/export.json + REPORT.md
```

Prerequisites: `uv`, `tmux`, Node 22+, `claude` and `codex` CLIs logged in. Optional: Sumo in `~/src/sumo`
(`pixi install && pixi run build`) for walking and MPC placement (rungs 2 and 4).

## How it works, end to end

1. **You give it a goal file.** `goals/spot_assembly.yaml` holds the question, robots, scene tiers, the
   de-idealization ladder, the success target (0.8 per tier) and the episode budget. That is the only input.
2. **`./lab.sh run` starts an Omnigent session** with the PI agent (`agents/assembly_lab/`). The session is titled
   with the research goal. The PI reads the goal and the research record and resumes wherever the record left off.
3. **Each discovery cycle** is Question -> Evidence -> Hypothesis -> Experiment -> Result -> Decision. The PI
   delegates every step to a specialist, in parallel where independent:
   literature_scout (cited evidence) -> method_designer (labeled hypothesis, method config, prompts) ->
   experiment_planner (>= 2 candidate experiments with a matched control, picks one under budget) ->
   robotics_engineer (implements new stages, smoke-tests them) -> experiment_runner (runs the batch) ->
   analyst (metrics with CIs, failure categories, watches failure videos) -> reviewer (accepts or rejects) ->
   PI (records a decision that cites the reviewed result).
4. **Cycles run back to back** until every rung passes, the budget is spent, or the remaining rungs need
   equipment that is not installed. Then the PI exports the report and asks you about the held-out evaluation.
5. **Everything is recorded** in `record/lab.db` (below); a restart loses nothing.

### Agents, models and effort (`agents/assembly_lab/`)

| Agent | Decision it owns | Harness | Model | Effort | Lab tools (allow-listed) |
|---|---|---|---|---|---|
| PI (orchestrator) | goal, budget, final decisions | claude-sdk | Claude Opus 5.5 | medium | lab_status, query_record, record_decision, budget_status |
| method_designer | which approach, what to change next | claude-sdk + files | Claude Opus 5.5 | medium | add_hypothesis, eval_prompt, call_model, validate_method |
| analyst | what results mean | claude-sdk + read-only files | Claude Opus 5.5 | medium | compute_metrics, render_rollout, log_result, set_hypothesis_status |
| experiment_planner | which test next | claude-sdk | Claude Sonnet 5.5 | medium | propose_experiment, select_experiment, budget_status |
| scene_designer | what the robot is tested on | claude-sdk + shell | Claude Sonnet 5.5 | medium | make_dev_scene, check_stability, list/get_scene |
| literature_scout | what is known | claude-sdk | Claude Sonnet 5.5 | low | arxiv_search, openalex_search, add_evidence |
| experiment_runner | running it correctly | claude-sdk + read-only files | Claude Sonnet 5.5 | low | run_sim_batch |
| robotics_engineer | how the method is implemented | codex + shell | GPT-5.6-sol (Codex) | medium | validate_method, run_sumo_task, check_stability |
| reviewer | whether a claim is justified | codex, no file access | GPT-5.6-sol (Codex) | medium | review_result, compute_metrics |

Models are set per agent in its `config.yaml` (`executor.model` / `executor.reasoning_effort`, NOT inside `executor.config`, where Omnigent ignores them); Codex agents take
their model from `~/.codex/config.toml`. The reviewer is deliberately a different model family from the analyst.
Method models used *inside* the pipeline (e.g. parsing the target picture) are separate: `codex` or `claude-*`,
set in `methods/*.yaml`, cached per (model, prompt, image) in `record/model_cache/`.

### Token use

Opus only where the reasoning is hard (PI, method design, analysis), Sonnet for routine roles, medium or low
effort everywhere. Agents pass record ids instead of prose, tool outputs are compact JSON, simulation runs in
local processes (no tokens), and repeated method-model calls hit the cache. Sub-agents report back once when done
(Omnigent inbox) instead of being polled. Caps: `max_tool_calls_per_session` (600) and `cost_budget` on the PI.

### What you see in the web UI (http://127.0.0.1:6767)

- **The chat** (one session per research run, titled with the goal): after every experiment, result and decision
  the PI posts the status table and a line on what it does next. Specialists appear as child sessions.
- **Status table** (`LAB_STATUS.md`, rebuilt on every record write): rungs x tiers with
  🟢 passed, 🔵 running, 🔴 below target, ⚪ not started, plus running experiments, hypotheses, latest decision,
  budget and recent runs.
- **Videos:** every batch saves a video per scene (first seed) and per failed episode, as a GIF and an MP4.
  Frames are a 2x3 tile: main, top, far side and left side views, plus the target picture and its top view, with
  the target drawn as translucent ghost blocks at the build site. Omnigent's chat blocks inline images, so videos
  appear as links (▶) that open in the file viewer; each run also has a `runs/<dir>/RUN.md` page.
- **Watch any episode live:** `uv run mjpython -m lab.view T3-dev-03 v3_grasp 1`.

### Layers

| Layer | Runs | Where |
|---|---|---|
| Relic whole-body policy | 50 Hz | Sumo (pixi env), via `run_sumo_task` |
| Sumo sampling MPC | 20 Hz | `lab/sumo_adapter/run_task.py` |
| Assembly method (what the lab develops) | once per build | `lab/pipeline.py`, `lab/stages/`, `methods/*.yaml`, `prompts/` |
| **Omnigent research lab** | once per experiment | `agents/assembly_lab/` |

Goal-specific parts are plug-ins: `goals/spot_assembly.yaml`, `equipment.yaml` (registry the agents read),
`methods/v0.yaml` (working start), `scenes/` (benchmark). The tools are one MCP server (`lab/server.py`); each agent
starts its own instance with a `tools:` allow-list, so "only the runner runs batches" is enforced by config.

### Scenes and tiers (`lab/scenes.py`)

Designed by the team before the agents start: T0 single block, T1 two-block stack (cube on cube, cube on brick),
T2 three blocks (tower, wall, bridge), T3 L, T and stepped pyramid (3-6 blocks). Each scene has a start layout
checked for reachability, a target picture plus top/side views and a merged multi-view picture, and an oracle spec.
6 dev + 4 held-out scenes per tier; held-out scenes are locked. The scene designer agent adds dev scenes (so far:
shuffled-order and off-grid variants) but cannot touch held-out ones.

### Research record (`lab/record.py`)

SQLite with typed, linked objects: evidence (E), scene, hypothesis (H, status open/supported/refuted/reopened,
labeled by author), experiment (X, candidate/selected/rejected), run (R, commit + seeds + results path), result
(RS, metrics copied from runs, reviewer verdict), decision (D, must cite reviewed results). Dangling ids are
rejected. `./lab.sh report` renders the loop as it happened.

### Policies and gates (human approval boundary)

| Boundary | Omnigent policy | Also enforced in the tool layer |
|---|---|---|
| Held-out scenes locked | `lab.policies.protect_heldout_set` (DENY; ASK for the final evaluation) | `run_sim_batch` gate, `get_scene` |
| Compute | none: batches run autonomously (`approve_large_batch` exists, off by default) | episode budget (`record/budget.json`), per-experiment episode cap |
| Frozen eval code | `lab.policies.freeze_eval_code` (ASK on edits to metrics/goals/policies) | `run_sim_batch` refuses if `lab/metrics.py` hash changed |
| Spend / runaway loops | built-in `cost_budget`, `max_tool_calls_per_session` | |
| Decisions cite results | `lab.policies.pi_cites_results` | `record_decision` requires reviewer verdicts |
| Two candidate tests | | `select_experiment` requires rejected alternatives; runs need a selected experiment |
| Read-only viewers | `lab.policies.read_only_workspace` (PI, runner, analyst) | |
| Engineer can't fake runs | `lab.policies.no_record_bypass` | |

Only two things pause for a human: the final held-out evaluation and edits to frozen evaluation code.
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
- Omnigent's chat blocks inline images (its markdown security settings), so videos and colors are links and markers.
- Omnigent's native tmux-based Codex/Claude terminals did not start reliably here; the lab uses the headless
  `claude-sdk` and `codex` harnesses instead.
- Validation before real hardware: real Spot grasp contact, perception from real images, sim-to-real gap.

## Layout

```
agents/assembly_lab/           PI bundle (config.yaml, AGENTS.md) + agents/<role>/config.yaml
agents/single_agent_baseline/  control: one agent, same tools and budget, separate record
lab/  record.py server.py policies.py runner.py pipeline.py sim.py scenes.py metrics.py method_models.py
      report.py status.py view.py   stages/ (agent-written stages)   sumo_tasks/ (agent-written Sumo tasks)
lab/sumo_adapter/run_task.py   runs inside Sumo's pixi env
methods/ prompts/ goals/ scenes/{dev,heldout}/ equipment.yaml record/ runs/
```
