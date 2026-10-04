# Omnigent Robotics Lab

**An autonomous, simulation-first robotics research lab where AI agents design, run, and iterate on experiments end-to-end.**

The global physical AI market is valued at over $110 billion in 2026 and is projected to surpass $430 billion by 2030 ([Yahoo Finance](https://uk.finance.yahoo.com/news/physical-ai-market-set-surpass-143900304.html)). To keep pace, robotics research needs to move faster. Omnigent Robotics Lab lets a PI agent and 8 specialist agents take on a robotics research goal: they design simulated experiments in MuJoCo/Sumo, implement methods with the installed equipment, test them, and decide what to investigate next.

## What It Does

- **Autonomous research cycles** — A PI agent and 8 specialists run Question-Evidence-Hypothesis-Experiment-Result-Decision loops
- **De-idealization ladder** — Replace one oracle per rung, from build order to perception
- **Full traceability** — Every decision cites a reviewed result in a SQLite record
- **Simulation benchmark** — 6 dev + 4 held-out scenes per tier, from a single block (T0) to a 6-block pyramid (T3)
- **Policy gates** — Locked held-out scenes, frozen eval code, reviewer sign-off and budget caps
- **Video rollouts** — GIF/MP4 per scene for every batch, in 2x3 tiled views

## Agent Team

| Agent | Responsibility |
|---|---|
| Principal investigator | Research direction, delegation, budget, and final decisions |
| Literature scout | Relevant research and supporting evidence |
| Scene designer | Development scenes and target stability |
| Method designer | Testable approaches and method revisions |
| Experiment planner | Candidate comparison and test selection |
| Robotics engineer | Implementation and robot-method validation |
| Experiment runner | Simulation batches and execution artifacts |
| Analyst | Metrics, failure analysis, and interpretation |
| Reviewer | Independent checks of evidence and conclusions |

## Tech Stack

| Layer | Technology |
|---|---|
| Agent orchestration | Omnigent |
| Research tools | Python 3.12+, MCP |
| Simulation | MuJoCo, Sumo (Relic whole-body policy, 50 Hz) |
| MPC | Sumo sampling MPC (20 Hz, policy-in-the-loop) |
| Research record | SQLite (typed record objects) |
| Methods and goals | YAML configurations and versioned prompts |
| Evaluation | NumPy, fixed metrics, 95% Wilson confidence intervals |
| Frontend | HTML, CSS, JavaScript, SVG, JSON data |
| Frontend hosting | Vercel |

## Quick Start

**Prerequisites:** `uv`, `tmux`, Node 22+, `claude` and `codex` CLIs logged in. Optional: Sumo in `~/src/sumo` for walking and MPC placement (rungs 2 and 4).

```bash
# Clone and enter
git clone https://github.com/JonathanLehner/Omnigent-Robotics-Lab.git && cd Omnigent-Robotics-Lab

# Setup (uv env, Omnigent + lab policies, benchmark scenes)
./lab.sh setup

# Run autonomously on the Spot assembly goal
./lab.sh run      # watch live at http://127.0.0.1:6767

# Export report after completion
./lab.sh report
```

## Project Structure

```
| Path | Purpose |
|---|---|
| `lab.sh` | CLI entry point (setup, run, report) |
| `frontend/` | Opal Labs website, benchmark assets, and record explorer |
| `agents/assembly_lab/` | PI configuration and eight specialist agents |
| `agents/single_agent_baseline/` | Single-agent control |
| `lab/` | MCP server, simulation pipeline, evaluation, record, and reports |
| `lab/stages/`, `lab/sumo_tasks/` | Pipeline stages and Sumo tasks written by the agents |
| `lab/sumo_adapter/` | Optional Sumo integration |
| `goals/` | Research question, success target, tiers, and budget |
| `methods/`, `prompts/` | Assembly methods and model prompts |
| `scenes/dev/`, `scenes/heldout/` | Development and held-out benchmarks |
| `equipment.yaml` | Equipment registry available to the agents |
| `record/`, `runs/` | Research record and generated execution artifacts |
```

## License

MIT
