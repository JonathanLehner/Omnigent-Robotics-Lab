## Lab status

*How can a Spot robot plan and assemble a target structure, given only a picture of it, from handled building blocks, without task-specific training?*

| Rung | Capability | T0 | T1 | T2 | T3 | Status |
|:--:|:--|:--:|:--:|:--:|:--:|:--|
| 0 | Baseline (all oracles) | 🟢 1.00 | 🟢 1.00 | 🔴 0.75 | 🔴 0.50 | 🔴 Below target: T2, T3 |
| 1 | Planned build order | 🟢 1.00 | 🟢 1.00 | 🟢 0.94 | 🟢 0.98 | 🟢 Passed |
| 2 | Walking approach | 🟢 0.80 | 🟢 1.00 | ⚪ – | ⚪ – | 🟢 Passed so far |
| 3 | Real gripper grasp | 🟢 1.00 | 🔴 0.00 | ⚪ – | ⚪ – | 🔴 Below target: T1 |
| 4 | Lab-chosen placement (MPC) | ⚪ – | 🔵 1.00 | 🔵 0.62 | 🔵 0.25 | 🔵 Running: T1, T2, T3 |
| 5 | Structure from picture | 🟢 1.00 | 🟢 1.00 | 🔴 0.50 | 🟢 0.88 | 🔴 Below target: T2 |

🟢 passed (>= 0.80)  🔵 running now  🔴 below target  ⚪ not started. Success rate on dev scenes, latest run per cell.

<details><summary>95% CIs and runs</summary>

R0/T0 1.00 [0.61, 1.00] n=6 (R-023); R0/T1 1.00 [0.76, 1.00] n=12 (R-023); R0/T2 0.75 [0.41, 0.93] n=8 (R-079); R0/T3 0.50 [0.22, 0.78] n=8 (R-079); R1/T0 1.00 [0.93, 1.00] n=48 (R-002); R1/T1 1.00 [0.93, 1.00] n=48 (R-018); R1/T2 0.94 [0.80, 0.98] n=32 (R-018); R1/T3 0.98 [0.89, 1.00] n=48 (R-018); R2/T0 0.80 [0.58, 0.92] n=20 (R-030); R2/T1 1.00 [0.68, 1.00] n=8 (R-046); R3/T0 1.00 [0.61, 1.00] n=6 (R-025); R3/T1 0.00 [0.00, 0.24] n=12 (R-025); R4/T1 1.00 [0.86, 1.00] n=24 (R-033); R4/T2 0.62 [0.31, 0.86] n=8 (R-033); R4/T3 0.25 [0.07, 0.59] n=8 (R-033); R5/T0 1.00 [0.93, 1.00] n=48 (R-013); R5/T1 1.00 [0.86, 1.00] n=24 (R-077); R5/T2 0.50 [0.22, 0.78] n=8 (R-078); R5/T3 0.88 [0.53, 0.98] n=8 (R-078)

</details>

## Now (selected experiments not yet fully run)

- **X-014** (149/306 episodes run): v4_place, seeds 0-7, on all 34 T1/T2/T3 dev scenes (272). Control is v0 at matched scenes and seeds 0-7, reused from R-005 (T1-01..06), R-007 (T2-01..06), R-012. Tests -.
- **X-022** (0/24 episodes run): Rung-4 MPC T0 probe, run SERIALLY after X-019 finishes. Run v4_mpc (Sumo CEM placement; cost = block pose + upright + collision clearance + settle velocity + an. Tests -.
- **X-025** (0/96 episodes run): Full pre-registered design. v3_grasp_lift (release_vertical_lift_m 0.125, 12 steps, no oracle correction, default physics) at seeds 0-7 on T0-dev-01..06 and T1-. Tests H-010.

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

## Latest decision (D-010, cites RS-015, RS-016)

STOP. The budget is exhausted: 1 of 1500 episodes left, fewer than the cheapest useful experiment. No rung changes status in this cycle.
- Rung 2 (walking): X-035/RS-015 is VOID on provenance. The reviewer rejected it under H-014's VOID-first rule, because the treatment and control ran on mixed commits and code states. H-012 and H-014 are back to open. The 'refutes' field in RS-015 is superseded by the hypotheses' open status.
- Rung 5 off-grid (T2/T3): X-037/RS-016 is VOID. The reviewer rejected it under H-013's fingerprint clause: commit bb3a96c changed vlm_measure.py, the cache behaviour an

**Next:** Proposed, not run; this needs a new episode budget. Each run must be on a single commit, with the fingerprint frozen across all arms.
(1) X-035 rerun: v2_walk vs v0, 40 eps, round-robin as ONE pairs batch, testing H-014.
(2) X-037 rerun: v46 with per-block perceived-y logging, and the white-mask fix generalised to white cubes on bridges. 72 eps, testing H-013.
(3) X-025: v3_grasp_lift, 96 eps, run

**Budget:** 1499 of 1500 episodes used, 1 left. Record: 9 evidence, 14 hypotheses, 39 experiments, 79 runs, 16 results, 10 decisions.

## Recent runs (videos inside)

- R-079 v0 (rung 0), T2, T3: 0.62, failures {"placement_error": 6} - [runs/1791085380_v0/RUN.md](runs/1791085380_v0/RUN.md) [▶ T2-dev-15_s0](runs/1791085380_v0/T2-dev-15_s0.gif) [▶ T2-dev-16_s0](runs/1791085380_v0/T2-dev-16_s0.gif) [▶ T2-dev-17_s0](runs/1791085380_v0/T2-dev-17_s0.gif)
- R-078 v46_vlm_measure (rung 5), T2, T3: 0.69, failures {"placement_error": 4, "perception_mismatch": 1, "perception_missing_block": 1} - [runs/1791085245_v46_vlm_measure/RUN.md](runs/1791085245_v46_vlm_measure/RUN.md) [▶ T2-dev-15_s0](runs/1791085245_v46_vlm_measure/T2-dev-15_s0.gif) [▶ T2-dev-16_s0](runs/1791085245_v46_vlm_measure/T2-dev-16_s0.gif) [▶ T2-dev-16_s1](runs/1791085245_v46_vlm_measure/T2-dev-16_s1.gif)
- R-077 v46_vlm_measure (rung 5), T1, T2, T3: 0.95, failures {"perception_mismatch": 2, "perception_missing_block": 2} - [runs/1791084911_v46_vlm_measure/RUN.md](runs/1791084911_v46_vlm_measure/RUN.md) [▶ T1-dev-01_s0](runs/1791084911_v46_vlm_measure/T1-dev-01_s0.gif) [▶ T1-dev-02_s0](runs/1791084911_v46_vlm_measure/T1-dev-02_s0.gif) [▶ T1-dev-03_s0](runs/1791084911_v46_vlm_measure/T1-dev-03_s0.gif)
- R-076 v0 (rung 0), T0, T1: 1.00, failures {} - [runs/1791084837_v0/RUN.md](runs/1791084837_v0/RUN.md) [▶ T0-dev-01_s0](runs/1791084837_v0/T0-dev-01_s0.gif) [▶ T0-dev-02_s1](runs/1791084837_v0/T0-dev-02_s1.gif) [▶ T0-dev-03_s2](runs/1791084837_v0/T0-dev-03_s2.gif)
- R-075 v0 (rung 0), T0: 1.00, failures {} - [runs/1791084625_v0/RUN.md](runs/1791084625_v0/RUN.md) [▶ T0-dev-04_s15](runs/1791084625_v0/T0-dev-04_s15.gif)
- R-074 v0 (rung 0), T0: 1.00, failures {} - [runs/1791084614_v0/RUN.md](runs/1791084614_v0/RUN.md) [▶ T0-dev-03_s14](runs/1791084614_v0/T0-dev-03_s14.gif)
- R-073 v0 (rung 0), T0: 1.00, failures {} - [runs/1791084604_v0/RUN.md](runs/1791084604_v0/RUN.md) [▶ T0-dev-02_s13](runs/1791084604_v0/T0-dev-02_s13.gif)
- R-072 v0 (rung 0), T0: 1.00, failures {} - [runs/1791084590_v0/RUN.md](runs/1791084590_v0/RUN.md) [▶ T0-dev-01_s12](runs/1791084590_v0/T0-dev-01_s12.gif)
