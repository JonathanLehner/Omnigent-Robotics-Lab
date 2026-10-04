## Lab status

*How can a Spot robot plan and assemble a target structure, given only a picture of it, from handled building blocks, without task-specific training?*

| Rung | Capability | T0 | T1 | T2 | T3 | Status |
|:--:|:--|:--:|:--:|:--:|:--:|:--|
| 0 | Baseline (all oracles) | 🟢 1.00 | 🟢 1.00 | 🟢 0.88 | 🟢 1.00 | 🟢 Passed |
| 1 | Planned build order | 🟢 1.00 | 🟢 1.00 | 🟢 0.94 | 🟢 0.98 | 🟢 Passed |
| 2 | Walking approach | 🔵 0.80 | 🔵 1.00 | ⚪ – | ⚪ – | 🔵 Running: T0, T1 |
| 3 | Real gripper grasp | 🟢 1.00 | 🔴 0.00 | ⚪ – | ⚪ – | 🔴 Below target: T1 |
| 4 | Lab-chosen placement (MPC) | ⚪ – | 🔵 1.00 | 🔵 0.62 | 🔵 0.25 | 🔵 Running: T1, T2, T3 |
| 5 | Structure from picture | 🟢 1.00 | 🔴 0.67 | 🔴 0.75 | 🔴 0.00 | 🔴 Below target: T1, T2, T3 |

🟢 passed (>= 0.80)  🔵 running now  🔴 below target  ⚪ not started. Success rate on dev scenes, latest run per cell.

<details><summary>95% CIs and runs</summary>

R0/T0 1.00 [0.61, 1.00] n=6 (R-023); R0/T1 1.00 [0.76, 1.00] n=12 (R-023); R0/T2 0.88 [0.53, 0.98] n=8 (R-035); R0/T3 1.00 [0.68, 1.00] n=8 (R-035); R1/T0 1.00 [0.93, 1.00] n=48 (R-002); R1/T1 1.00 [0.93, 1.00] n=48 (R-018); R1/T2 0.94 [0.80, 0.98] n=32 (R-018); R1/T3 0.98 [0.89, 1.00] n=48 (R-018); R2/T0 0.80 [0.58, 0.92] n=20 (R-030); R2/T1 1.00 [0.68, 1.00] n=8 (R-046); R3/T0 1.00 [0.61, 1.00] n=6 (R-025); R3/T1 0.00 [0.00, 0.24] n=12 (R-025); R4/T1 1.00 [0.86, 1.00] n=24 (R-033); R4/T2 0.62 [0.31, 0.86] n=8 (R-033); R4/T3 0.25 [0.07, 0.59] n=8 (R-033); R5/T0 1.00 [0.93, 1.00] n=48 (R-013); R5/T1 0.67 [0.39, 0.86] n=12 (R-034); R5/T2 0.75 [0.41, 0.93] n=8 (R-034); R5/T3 0.00 [0.00, 0.32] n=8 (R-034)

</details>

## Now (selected experiments not yet fully run)

- **X-014** (149/306 episodes run): v4_place, seeds 0-7, on all 34 T1/T2/T3 dev scenes (272). Control is v0 at matched scenes and seeds 0-7, reused from R-005 (T1-01..06), R-007 (T2-01..06), R-012. Tests -.
- **X-022** (0/24 episodes run): Rung-4 MPC T0 probe, run SERIALLY after X-019 finishes. Run v4_mpc (Sumo CEM placement; cost = block pose + upright + collision clearance + settle velocity + an. Tests -.
- **X-025** (0/96 episodes run): Full pre-registered design. v3_grasp_lift (release_vertical_lift_m 0.125, 12 steps, no oracle correction, default physics) at seeds 0-7 on T0-dev-01..06 and T1-. Tests H-010.
- **X-035** (0/40 episodes run): v2_walk (w_heading=160, frozen) vs fresh matched v0 control. Tests H-012.

## Hypotheses

| id | rung | status | statement |
|---|---|---|---|
| H-001 | 1 | supported | Replacing the oracle build order with symbolic_order_planner (plan_order=support_sort: support-graph topological sort + settle test per pref |
| H-002 | 5 | refuted | Replacing the oracle structure spec with VLM parsing (perceive=vlm, call_model image->JSON with block type + pose), with oracle order and bu |
| H-003 | 5 | open | Hybrid rung-5 perception (VLM for block count, type, color and support relations, followed by a geometric contact snap: x=0, z=0.05+0.10*lay |
| H-004 | 3 | refuted | Rung 3, v3_grasp (methods/v3_grasp.yaml, stage lab/stages/real_grasp.py, build=scripted_real_grasp): the weld is replaced by a physical Spot |
| H-005 | 0 | refuted | [X-016 | rungs 0 and 3, cross-cutting | filed before any X-016 results were read] Contact creep from MuJoCo's default pyramidal friction con |
| H-006 | 4 | reopened | [X-014 | rung 4 | filed before any X-014 / v4_place results were read] v4_place (methods/v4_place.yaml, build=closed_loop_weld_place, lab/st |
| H-007 | 0 | open | [X-016 | rungs 0 and 3, cross-cutting | AMENDS H-005 part (b); filed before any X-016 results were read; part (a) unchanged] (a) Contact cre |
| H-008 | 5 | refuted | [rung 5 / '4.5' | filed before any v45 results] v45_place_vlm (perceive=vlm with codex + prompts/perceive/v1.md, plan_order=oracle, build=cl |
| H-009 | 2 | refuted | [X-019 | rung 2 | intended as pre-registration: a query_record(kind=run, contains="X-019") just before filing returned no runs. Pre-registra |
| H-010 | 3 | open | Successor to H-004 (refuted by RS-005). Rung 3, new config v3_grasp_lift (methods/v3_grasp_lift.yaml) = v3_grasp unchanged except for the po |
| H-011 | 4 | open | [X-022 | pre-registered treatment, reused historical control] v4_mpc vs v0 control R-003 (R-003 predates this hypothesis; only the v4_mpc ar |
| H-012 | 2 | open | [Rung 2 | pre-registered treatment, reused historical control | follows H-009, which X-019 refuted: 20/24, NOT CLEARED] Filed BEFORE any bat |
| H-013 | 5 | open | [Rung 5 | agent hypothesis by method_designer | SUCCESSOR to H-008 (refuted by RS-012), and SUPERSEDES H-003 | filed BEFORE any run of v46_v |
| H-014 | 2 | open | [Rung 2 | AMENDMENT of H-012 for X-035 | label: pre-registered treatment; partly reused historical control | filed BEFORE any X-035 run.] PR |

## Latest decision (D-009, cites RS-014)

Use v45 (VLM structure + closed-loop placement) for rung-5 on T1 grid-aligned scenes only: 12/12 scenes [0.76-1.00] vs v5_vlm 8/12. Rung 5 is NOT cleared on T2/T3 off-grid: v45 7/16 vs v0 15/16. Most of that gap is perception (the 6.5 cm grid, staircase misread as a stack). H-008 stays refuted on clauses (b) and (c). Idealizations still in use: oracle_block_pose (v4_place), weld grasp, cached T1 VLM outputs, scripted base (no walk). Next rung-5 work moves to perception: v46_vlm_measure (H-013), lateral position measured from pixels only. Any future v45 vs v5_vlm comparison must use matched cac

**Next:** H-013 Gate 0 (free, pixel-only measurement, 20 scenes). If it passes: v46_vlm_measure vs v0 on T1-dev-01..12 + 8 off-grid T2/T3 scenes, seeds 0-1, 40 episodes, on committed code. In parallel, once the human commits the tree and reloads the server: X-035 clean rung-2 test (32 episodes, H-014).

**Budget:** 1387 of 1500 episodes used, 113 left. Record: 9 evidence, 14 hypotheses, 36 experiments, 46 runs, 14 results, 9 decisions.

## Recent runs (videos inside)

- R-046 v2_walk (rung 2), T1: 1.00, failures {} - [runs/1791082144_v2_walk/RUN.md](runs/1791082144_v2_walk/RUN.md) [▶ T1-dev-03_s0](runs/1791082144_v2_walk/T1-dev-03_s0.gif) [▶ T1-dev-04_s0](runs/1791082144_v2_walk/T1-dev-04_s0.gif) [▶ T1-dev-05_s0](runs/1791082144_v2_walk/T1-dev-05_s0.gif)
- R-045 v2_walk (rung 2), T0: 1.00, failures {} - [runs/1791082102_v2_walk/RUN.md](runs/1791082102_v2_walk/RUN.md) [▶ T0-dev-02_s1](runs/1791082102_v2_walk/T0-dev-02_s1.gif)
- R-044 v2_walk (rung 2), T0: 1.00, failures {} - [runs/1791082081_v2_walk/RUN.md](runs/1791082081_v2_walk/RUN.md) [▶ T0-dev-02_s1](runs/1791082081_v2_walk/T0-dev-02_s1.gif)
- R-043 v2_walk (rung 2), T1: 1.00, failures {} - [runs/1791081914_v2_walk/RUN.md](runs/1791081914_v2_walk/RUN.md) [▶ T1-dev-02_s0](runs/1791081914_v2_walk/T1-dev-02_s0.gif)
- R-042 v2_walk (rung 2), T1: 1.00, failures {} - [runs/1791081862_v2_walk/RUN.md](runs/1791081862_v2_walk/RUN.md) [▶ T1-dev-01_s0](runs/1791081862_v2_walk/T1-dev-01_s0.gif)
- R-041 v2_walk (rung 2), T0: 1.00, failures {} - [runs/1791081771_v2_walk/RUN.md](runs/1791081771_v2_walk/RUN.md) [▶ T0-dev-06_s0](runs/1791081771_v2_walk/T0-dev-06_s0.gif)
- R-040 v2_walk (rung 2), T0: 1.00, failures {} - [runs/1791081696_v2_walk/RUN.md](runs/1791081696_v2_walk/RUN.md) [▶ T0-dev-05_s0](runs/1791081696_v2_walk/T0-dev-05_s0.gif)
- R-039 v2_walk (rung 2), T0: 1.00, failures {} - [runs/1791081628_v2_walk/RUN.md](runs/1791081628_v2_walk/RUN.md) [▶ T0-dev-04_s0](runs/1791081628_v2_walk/T0-dev-04_s0.gif)
