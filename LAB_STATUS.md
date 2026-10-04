## Lab status

*How can a Spot robot plan and assemble a target structure, given only a picture of it, from handled building blocks, without task-specific training?*

| Rung | Capability | T0 | T1 | T2 | T3 | Status |
|:--:|:--|:--:|:--:|:--:|:--:|:--|
| 0 | Baseline (all oracles) | 🟢 1.00 | 🟢 1.00 | 🟢 0.94 | 🟢 0.92 | 🟢 Passed |
| 1 | Planned build order | 🟢 1.00 | 🟢 1.00 | 🟢 0.94 | 🟢 0.98 | 🟢 Passed |
| 2 | Walking approach | 🟢 0.94 | 🟢 1.00 | ⚪ – | ⚪ – | 🟢 Passed so far |
| 3 | Real gripper grasp | 🔴 0.75 | 🔴 0.00 | ⚪ – | ⚪ – | 🔴 Below target: T0, T1 |
| 4 | Lab-chosen placement (MPC) | 🔴 0.00 | 🔵 1.00 | 🔵 0.62 | 🔵 0.25 | 🔵 Running: T1, T2, T3 |
| 5 | Structure from picture | 🟢 1.00 | 🟢 1.00 | 🟢 1.00 | 🟢 0.88 | 🟢 Passed |

🟢 passed (>= 0.80)  🔵 running now  🔴 below target  ⚪ not started. Success rate on dev scenes, latest run per cell.

<details><summary>95% CIs and runs</summary>

R0/T0 1.00 [0.93, 1.00] n=48 (R-099); R0/T1 1.00 [0.93, 1.00] n=48 (R-099); R0/T2 0.94 [0.83, 0.98] n=48 (R-100); R0/T3 0.92 [0.80, 0.97] n=48 (R-100); R1/T0 1.00 [0.93, 1.00] n=48 (R-002); R1/T1 1.00 [0.93, 1.00] n=48 (R-018); R1/T2 0.94 [0.80, 0.98] n=32 (R-018); R1/T3 0.98 [0.89, 1.00] n=48 (R-018); R2/T0 0.94 [0.72, 0.99] n=16 (R-095); R2/T1 1.00 [0.93, 1.00] n=48 (R-091); R3/T0 0.75 [0.61, 0.85] n=48 (R-088); R3/T1 0.00 [0.00, 0.07] n=48 (R-088); R4/T0 0.00 [0.00, 0.14] n=24 (R-082); R4/T1 1.00 [0.86, 1.00] n=24 (R-033); R4/T2 0.62 [0.31, 0.86] n=8 (R-033); R4/T3 0.25 [0.07, 0.59] n=8 (R-033); R5/T0 1.00 [0.93, 1.00] n=48 (R-098); R5/T1 1.00 [0.93, 1.00] n=48 (R-098); R5/T2 1.00 [0.68, 1.00] n=8 (R-094); R5/T3 0.88 [0.53, 0.98] n=8 (R-094)

</details>

## Now (selected experiments not yet fully run)

- **X-014** (149/306 episodes run): v4_place, seeds 0-7, on all 34 T1/T2/T3 dev scenes (272). Control is v0 at matched scenes and seeds 0-7, reused from R-005 (T1-01..06), R-007 (T2-01..06), R-012. Tests -.
- **X-022** (0/24 episodes run): Rung-4 MPC T0 probe, run SERIALLY after X-019 finishes. Run v4_mpc (Sumo CEM placement; cost = block pose + upright + collision clearance + settle velocity + an. Tests -.
- **X-025** (0/96 episodes run): Full pre-registered design. v3_grasp_lift (release_vertical_lift_m 0.125, 12 steps, no oracle correction, default physics) at seeds 0-7 on T0-dev-01..06 and T1-. Tests H-010.
- **X-044** (0/157 episodes run): X-014 remainder: v4_place with x-shortfall fix, 157 episodes on T1-T3 dev scenes, plus a matched fresh v0 control.. Tests H-006.
- **X-061** (96/288 episodes run): CANDIDATE (c) JOINT three-arm test on ONE frozen commit 76cc1a7 and ONE experiment id: v0 (fresh), v2_walk and v6_combo, all on the same 96 pairs (48 T2 seeds 2. Tests H-020.

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
| H-010 | 3 | refuted | Successor to H-004 (refuted by RS-005). Rung 3, new config v3_grasp_lift (methods/v3_grasp_lift.yaml) = v3_grasp unchanged except for the po |
| H-011 | 4 | open | [X-022 | pre-registered treatment, reused historical control] v4_mpc vs v0 control R-003 (R-003 predates this hypothesis; only the v4_mpc ar |
| H-012 | 2 | refuted | [Rung 2 | pre-registered treatment, reused historical control | follows H-009, which X-019 refuted: 20/24, NOT CLEARED] Filed BEFORE any bat |
| H-013 | 5 | open | [Rung 5 | agent hypothesis by method_designer | SUCCESSOR to H-008 (refuted by RS-012), and SUPERSEDES H-003 | filed BEFORE any run of v46_v |
| H-014 | 2 | refuted | [Rung 2 | AMENDMENT of H-012 for X-035 | label: pre-registered treatment; partly reused historical control | filed BEFORE any X-035 run.] PR |
| H-015 | 5 | open | [Rung 5 | agent hypothesis by method_designer | DATED AMENDMENT to H-013, 2026-10-04 | filed BEFORE any rerun episode. Pre-check: query_reco |
| H-016 | 2 | refuted | [Rung 2 | SUCCESSOR of H-012/H-014 (refuted by RS-017 on clause 2(e) only) | label: agent hypothesis by method_designer | FILED BEFORE ANY R |
| H-017 | 4 | open | [Rung 4 | SUCCESSOR of H-011 (refuted by RS-018, X-043: v4_mpc 0/24 vs fresh v0 24/24) | label: agent hypothesis by method_designer | FILED  |
| H-018 | 4 | open | [Rung 4 | AMENDMENT of H-011 (OPEN: "not cleared; method-as-described untested", per D-012 and the reviewer's REJECT of RS-018) | SUPERSEDES |
| H-019 | 5 | supported | [Rung 5 | agent hypothesis by method_designer | SUCCESSOR of H-013 and H-015. Both stay OPEN; X-045 was VOIDED by D-013 under H-015 A3, and  |
| H-020 | 5 | supported | [Combined rungs 1+2+5 | agent hypothesis by method_designer | for X-049 | FILED BEFORE ANY v6_combo RUN, NO CODE WRITTEN. methods/v6_combo*. |
| H-021 | 2 | open | [Rung 2 on T2/T3 | agent hypothesis by method_designer | for X-061 arm (a) | FILED BEFORE ANY X-061 RUN, NO CODE WRITTEN. Short pre-registra |
| H-022 | 5 | open | [Combined rungs 1+2+5 on T2/T3 | agent hypothesis by method_designer | SUCCESSOR of H-020 | for X-061 arm (b) | FILED BEFORE ANY X-061 RUN,  |

## Latest decision (D-017, cites RS-028)

COMBINED method (rungs 1+2+5: v47_vlm_anchor + support_sort + v2_walk + closed-loop weld place) CLEARED on T0 and T1. H-020 SUPPORTED (RS-028 with analysis RS-027, reviewer ACCEPT). X-058 at 76cc1a7: v6_combo R-098 T0 48/48 [0.926,1.000], T1 48/48 [0.926,1.000]; fresh v0 R-099 96/96 on identical pairs. E-018 audit: V1-V4 pass. Placement -1.8 cm vs v0. No composition-penalty claim (both at ceiling; ~7-point penalty unresolved). R-097/X-055 exploratory and closed (runner cap bug). RUNG 4 PAUSED: H-011 construct-invalid (D-012); H-018 pre-run gate FAILED at 9cf673d (floor drag 2.38 cm > 1.0, forc

**Next:** X-061: v0 / v2_walk / v6_combo on the same 96 T2/T3 pairs (seeds 2000-2047 / 2100-2147, E-019 PASS) at 76cc1a7, 288 episodes. Proposed, not run: H-018 successor (rung 4), v3_grasp_hold (rung 3), walk terminal heading term (rung 2 precision).

**Budget:** 2595 of 3000 episodes used, 405 left. Record: 19 evidence, 22 hypotheses, 62 experiments, 100 runs, 28 results, 17 decisions.

## Recent runs (videos inside)

- R-100 v0 (rung 0), T2, T3: 0.93, failures {"placement_error": 7} - runs/1791107658_v0 [▶ T2-dev-01_s2000](runs/1791107658_v0/T2-dev-01_s2000.gif) [▶ T2-dev-01_s2002](runs/1791107658_v0/T2-dev-01_s2002.gif) [▶ T2-dev-02_s2004](runs/1791107658_v0/T2-dev-02_s2004.gif)
- R-099 v0 (rung 0), T0, T1: 1.00, failures {} - [runs/1791106666_v0/RUN.md](runs/1791106666_v0/RUN.md) [▶ T0-dev-01_s1400](runs/1791106666_v0/T0-dev-01_s1400.gif) [▶ T0-dev-02_s1408](runs/1791106666_v0/T0-dev-02_s1408.gif) [▶ T0-dev-03_s1416](runs/1791106666_v0/T0-dev-03_s1416.gif)
- R-098 v6_combo (rung 5), T0, T1: 1.00, failures {} - [runs/1791104821_v6_combo/RUN.md](runs/1791104821_v6_combo/RUN.md) [▶ T0-dev-01_s1400](runs/1791104821_v6_combo/T0-dev-01_s1400.gif) [▶ T0-dev-02_s1408](runs/1791104821_v6_combo/T0-dev-02_s1408.gif) [▶ T0-dev-03_s1416](runs/1791104821_v6_combo/T0-dev-03_s1416.gif)
- R-097 v6_combo (rung 5), T0, T1: 0.99, failures {"order_infeasible": 1, "perception_missing_block": 1, "order": 1} - [runs/1791101237_v6_combo/RUN.md](runs/1791101237_v6_combo/RUN.md) [▶ T0-dev-01_s400](runs/1791101237_v6_combo/T0-dev-01_s400.gif) [▶ T0-dev-02_s408](runs/1791101237_v6_combo/T0-dev-02_s408.gif) [▶ T0-dev-03_s416](runs/1791101237_v6_combo/T0-dev-03_s416.gif)
- R-096 v0 (rung 0), T2, T3: 0.94, failures {"placement_error": 1} - [runs/1791099094_v0/RUN.md](runs/1791099094_v0/RUN.md) [▶ T2-dev-24_s0](runs/1791099094_v0/T2-dev-24_s0.gif) [▶ T2-dev-25_s0](runs/1791099094_v0/T2-dev-25_s0.gif) [▶ T2-dev-26_s0](runs/1791099094_v0/T2-dev-26_s0.gif)
- R-095 v2_walk_settle (rung 2), T0: 0.94, failures {"placement_error": 1} - [runs/1791098460_v2_walk_settle/RUN.md](runs/1791098460_v2_walk_settle/RUN.md) [▶ T0-dev-01_s0](runs/1791098460_v2_walk_settle/T0-dev-01_s0.gif) [▶ T0-dev-02_s1](runs/1791098460_v2_walk_settle/T0-dev-02_s1.gif) [▶ T0-dev-03_s2](runs/1791098460_v2_walk_settle/T0-dev-03_s2.gif)
- R-094 v47_vlm_anchor (rung 5), T1, T2, T3: 0.97, failures {"placement_error": 1} - [runs/1791097883_v47_vlm_anchor/RUN.md](runs/1791097883_v47_vlm_anchor/RUN.md) [▶ T1-dev-01_s0](runs/1791097883_v47_vlm_anchor/T1-dev-01_s0.gif) [▶ T1-dev-02_s0](runs/1791097883_v47_vlm_anchor/T1-dev-02_s0.gif) [▶ T1-dev-03_s0](runs/1791097883_v47_vlm_anchor/T1-dev-03_s0.gif)
- R-093 v2_walk_xy (rung 2), T0: 1.00, failures {} - [runs/1791097205_v2_walk_xy/RUN.md](runs/1791097205_v2_walk_xy/RUN.md) [▶ T0-dev-01_s0](runs/1791097205_v2_walk_xy/T0-dev-01_s0.gif) [▶ T0-dev-02_s1](runs/1791097205_v2_walk_xy/T0-dev-02_s1.gif) [▶ T0-dev-03_s2](runs/1791097205_v2_walk_xy/T0-dev-03_s2.gif)
