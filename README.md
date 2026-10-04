# Opal Labs

**An autonomous, simulation-first robotics research lab where AI agents design, run and iterate on experiments end to end.**

The global physical AI market is valued at over $110 billion in 2026 and is projected to surpass $430 billion by 2030 ([Yahoo Finance](https://uk.finance.yahoo.com/news/physical-ai-market-set-surpass-143900304.html)). To keep pace, robotics research needs to move faster. In Opal Labs, a PI agent and eight specialist agents take on a robotics research goal: they design simulated experiments in MuJoCo, implement methods with the installed equipment, test them against controls, and decide what to investigate next. Built on [Omnigent](https://github.com/omnigent-ai/omnigent) for Hack-Nation Challenge 03.

## First research goal

Spot should assemble a target structure from handled building blocks, given only a picture of it and without task-specific training. The lab works up a ladder of capabilities, replacing one idealization per rung, and a rung only counts when it meets the success target (each block within 3 cm and 10° after 5 s; at least 0.8 success per tier).

| Rung | Capability | Status |
|:--:|---|---|
| 0 | Baseline (target spec, build order and robot placement given) | Passed, T0–T3 |
| 1 | Planned build order | Passed, T0–T3 |
| 2 | Walking approach (Sumo whole-body MPC) | Passed, T0–T1 alone; T0–T3 within the combined method |
| 3 | Physical gripper grasp | Open (rigid grasp used meanwhile) |
| 4 | Placement chosen by MPC | Open |
| 5 | Structure read from the picture | Passed, T0–T3 |

Tiers: T0 single block, T1 two-block stack, T2 three blocks (tower, wall, bridge), T3 L, T, staircase and pyramid shapes with up to six blocks.

**Main result (reviewer-accepted; the lab stopped after using its episode budget).** The combined method reads the structure from the picture, plans the build order, walks to the site and places each block with closed-loop correction. On unseen trials with a matched control on the same code version:

| Tiers | Combined method | Control (target given, robot placed) |
|---|---|---|
| T0 + T1, 96 trials | 96/96, median error 0.25 cm | 96/96, median error 2.19 cm |
| T2 + T3, 96 trials | 94/96, median error 0.33 cm | 93/96, median error 1.94 cm |

In one day the lab ran 64 experiments and 104 simulation batches (about 3,000 episodes), filed 22 hypotheses before their experiments, and recorded 18 decisions. The full record is in `REPORT.md` and `record/export.json`; the live status is `LAB_STATUS.md`.

## Agent team

| Agent | Responsibility | Model |
|---|---|---|
| PI | Research direction, delegation, budget, final decisions | Claude Opus 5.5 |
| Literature scout | Relevant research and supporting evidence | Claude Sonnet 5.5 |
| Scene designer | Development scenes and target stability | Claude Sonnet 5.5 |
| Method designer | Testable approaches, hypotheses, method revisions | Claude Opus 5.5 |
| Experiment planner | Competing candidate tests, selection under budget | Claude Sonnet 5.5 |
| Robotics engineer | Implementation and validation of methods | GPT-5.6 (Codex) |
| Experiment runner | Simulation batches and execution artifacts | Claude Sonnet 5.5 |
| Analyst | Metrics, failure analysis from videos, interpretation | Claude Opus 5.5 |
| Reviewer | Independent check of every result before the PI acts | GPT-5.6 (Codex) |

Each agent is an Omnigent configuration in `agents/assembly_lab/` with its own model, reasoning effort and tool allow-list (for example, only the runner can start simulation batches and only the PI records decisions).

## How it works

- **Research loop.** Question → evidence → hypothesis → at least two candidate experiments → run with a matched control → analysis → review → decision, then the next cycle. The PI runs cycles until the ladder is done or the episode budget is spent.
- **Research record.** One SQLite store (`lab/record.py`) of evidence, scenes, hypotheses, experiments, runs, results and decisions, linked by id; dangling citations are rejected.
- **Tools.** One MCP server (`lab/server.py`) gives agents the record, literature search (arXiv, OpenAlex), simulation batches, rollout rendering, metrics and model calls. Tools reload lab code on every call.
- **Reproducibility.** Every batch records its code fingerprint and seeds, passes gates (config, frozen evaluation code, held-out lock, smoke test) before using budget, and stays within the experiment's planned episodes. Results that mix code versions are voided.
- **Policies.** Omnigent policies lock the held-out scenes and the success criteria, keep viewing agents read-only, and stop the engineer from creating runs outside the planner and runner.
- **Videos.** Every batch saves GIF and MP4 videos of the first seed per scene and of every failure: a 2×3 tile of main, top, far-side and left-side views plus the target picture, with the target drawn as translucent blocks at the build site. `runs/<run>/RUN.md` lists them.
- **Method.** The robot's method is a pipeline of perception, build-order planning and build stages (`lab/pipeline.py`, `methods/*.yaml`); the engineer agent adds stages in `lab/stages/` and Sumo tasks in `lab/sumo_tasks/`.

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | Omnigent (Claude Agent SDK and Codex harnesses) |
| Research tools | Python 3.12+, MCP |
| Simulation | MuJoCo 3.6 with the Spot arm model from MuJoCo Menagerie |
| Walking | Sumo sampling MPC (20 Hz) over the Relic whole-body policy (50 Hz) |
| Perception | Vision-language model plus camera-anchored pixel measurement |
| Research record | SQLite |
| Methods and goals | YAML configurations and versioned prompts |
| Evaluation | NumPy, frozen metrics, 95% Wilson confidence intervals |
| Frontend | HTML, CSS, JavaScript, JSON data, hosted on Vercel |

## Quick start

**Prerequisites:** `uv`, `tmux`, Node 22+, `claude` and `codex` CLIs logged in. Optional: [Sumo](https://github.com/rai-opensource/sumo) in `~/src/sumo` (`pixi install && pixi run build`) for walking and MPC placement.

```bash
git clone https://github.com/JonathanLehner/Omnigent-Robotics-Lab.git && cd Omnigent-Robotics-Lab
./lab.sh setup     # uv env, Omnigent + lab policies, benchmark scenes, frozen evaluation
./lab.sh run       # pursue the research goal autonomously; watch at http://127.0.0.1:6767
./lab.sh report    # export record/export.json and REPORT.md
```

`./lab.sh run` continues the existing Omnigent session; `./lab.sh run --new` starts a fresh one. To watch an episode in the MuJoCo viewer: `uv run mjpython -m lab.view T3-dev-03 v0 1`.

## Project structure

| Path | Purpose |
|---|---|
| `lab.sh` | Entry point (setup, run, report) |
| `goals/` | Research goal, success target, tiers, ladder and budget |
| `agents/assembly_lab/` | PI configuration and the eight specialist agents |
| `agents/single_agent_baseline/` | Single-agent control |
| `lab/` | MCP server, simulation pipeline, evaluation, record, status and reports |
| `lab/stages/`, `lab/sumo_tasks/` | Pipeline stages and Sumo tasks written by the agents |
| `lab/sumo_adapter/` | Sumo integration |
| `methods/`, `prompts/` | Method configurations and model prompts |
| `scenes/dev/`, `scenes/heldout/` | Development and held-out benchmarks |
| `equipment.yaml` | Equipment registry available to the agents |
| `record/`, `runs/` | Research record and generated run artifacts |
| `frontend/` | Opal Labs website, benchmark assets and record explorer |

## License

MIT
