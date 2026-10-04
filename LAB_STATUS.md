## Lab status

*How can a Spot robot plan and assemble a target structure, given only a picture of it, from handled building blocks, without task-specific training?*

| Rung | Capability | T0 | T1 | T2 | T3 | Status |
|:--:|:--|:--:|:--:|:--:|:--:|:--|
| 0 | Baseline (all oracles) | 🟢 1.00 | 🟢 1.00 | 🟢 0.96 | 🟢 0.98 | 🟢 Passed |
| 1 | Planned build order | 🟢 1.00 | 🟢 1.00 | 🟢 0.94 | 🟢 0.98 | 🟢 Passed |
| 2 | Walking approach | 🔵 0.94 | 🟢 1.00 | 🟢 0.92 | 🟢 0.88 | 🔵 Running: T0 |
| 3 | Real gripper grasp | 🔴 0.75 | 🔴 0.00 | ⚪ – | ⚪ – | 🔴 Below target: T0, T1 |
| 4 | Lab-chosen placement (MPC) | 🔴 0.00 | 🔵 1.00 | 🔵 0.62 | 🔵 0.25 | 🔵 Running: T1, T2, T3 |
| 5 | Structure from picture | 🔵 1.00 | 🔵 1.00 | 🔵 0.96 | 🔵 1.00 | 🔵 Running: T0, T1, T2, T3 |

🟢 passed (>= 0.80)  🔵 running now  🔴 below target  ⚪ not started. Success rate on dev scenes, latest run per cell.

<details><summary>95% CIs and runs</summary>

R0/T0 1.00 [0.93, 1.00] n=48 (R-099); R0/T1 1.00 [0.93, 1.00] n=48 (R-099); R0/T2 0.96 [0.86, 0.99] n=48 (R-104); R0/T3 0.98 [0.89, 1.00] n=48 (R-104); R1/T0 1.00 [0.93, 1.00] n=48 (R-002); R1/T1 1.00 [0.93, 1.00] n=48 (R-018); R1/T2 0.94 [0.80, 0.98] n=32 (R-018); R1/T3 0.98 [0.89, 1.00] n=48 (R-018); R2/T0 0.94 [0.72, 0.99] n=16 (R-095); R2/T1 1.00 [0.93, 1.00] n=48 (R-091); R2/T2 0.92 [0.80, 0.97] n=48 (R-101); R2/T3 0.88 [0.75, 0.94] n=48 (R-101); R3/T0 0.75 [0.61, 0.85] n=48 (R-088); R3/T1 0.00 [0.00, 0.07] n=48 (R-088); R4/T0 0.00 [0.00, 0.14] n=24 (R-082); R4/T1 1.00 [0.86, 1.00] n=24 (R-033); R4/T2 0.62 [0.31, 0.86] n=8 (R-033); R4/T3 0.25 [0.07, 0.59] n=8 (R-033); R5/T0 1.00 [0.93, 1.00] n=48 (R-098); R5/T1 1.00 [0.93, 1.00] n=48 (R-098); R5/T2 0.96 [0.86, 0.99] n=48 (R-103); R5/T3 1.00 [0.93, 1.00] n=48 (R-103)

</details>

## Now (selected experiments not yet fully run)

- **X-014** (149/306 episodes run): v4_place, seeds 0-7, on all 34 T1/T2/T3 dev scenes (272). Control is v0 at matched scenes and seeds 0-7, reused from R-005 (T1-01..06), R-007 (T2-01..06), R-012. Tests -.
- **X-022** (0/24 episodes run): Rung-4 MPC T0 probe, run SERIALLY after X-019 finishes. Run v4_mpc (Sumo CEM placement; cost = block pose + upright + collision clearance + settle velocity + an. Tests -.
- **X-025** (0/96 episodes run): Full pre-registered design. v3_grasp_lift (release_vertical_lift_m 0.125, 12 steps, no oracle correction, default physics) at seeds 0-7 on T0-dev-01..06 and T1-. Tests H-010.
- **X-044** (0/157 episodes run): X-014 remainder: v4_place with x-shortfall fix, 157 episodes on T1-T3 dev scenes, plus a matched fresh v0 control.. Tests H-006.
- **X-065** (0/32 episodes run): v2_walk_head vs fresh v2_walk (stage 1 T0 screen). 16 pairs: scene T0-dev-((s mod 6)+1), fresh seeds 9000-9015. Caps: v2_walk_head 16, v2_walk 16, one primary b. Tests H-023.
- **X-067** (0/288 episodes run): Three arms on identical pairs: v3_grasp_hold (changes 1+2), v3_grasp_hold_c1 (change 1 only), fresh v3_grasp control. T0-dev-01..06 x 8 seeds, 7000-7047 (scene . Tests H-024.
- **X-069** (0/288 episodes run): v6_combo_aware vs fresh v6_combo (passive movement logging on, placed_block_telemetry true), identical pairs. T2 12 scenes x 4 seeds from 8000+4j (48 pairs); T3. Tests H-025.
- **X-071** (0/464 episodes run): Two components on one frozen commit. (E1, offline, primary): 20 uncached v47 parses x 58 targeted images (methods/v6_combo_v47b.targets.json, sha256 recorded at. Tests H-026.
- **X-073** (0/192 episodes run): H-026 E3+E4: v6_combo_v47b vs fresh v6_combo on identical pairs. E3 (T2): scenes 01,02,03,06,11,12,19,22,24,25,26,28, seeds 10000+4j..10003+4j (10000-10047), 48. Tests H-026.

## Hypotheses

| id | rung | status | statement |
|---|---|---|---|
| H-001 | 1 | supported | Replacing the oracle build order with symbolic_order_planner (plan_order=support_sort: support-graph topological sort + settle test per pref |
| H-002 | 5 | refuted | Replacing the oracle structure spec with VLM parsing (perceive=vlm, call_model image->JSON with block type + pose), with oracle order and bu |
| H-003 | 5 | open | Hybrid rung-5 perception (VLM for block count, type, color and support relations, followed by a geometric contact snap: x=0, z=0.05+0.10*lay |
| H-004 | 3 | refuted | Rung 3, v3_grasp (methods/v3_grasp.yaml, stage lab/stages/real_grasp.py, build=scripted_real_grasp): the weld is replaced by a physical Spot |
| H-005 | 0 | reopened | [X-016 | rungs 0 and 3, cross-cutting | filed before any X-016 results were read] Contact creep from MuJoCo's default pyramidal friction con |
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
| H-022 | 5 | supported | [Combined rungs 1+2+5 on T2/T3 | agent hypothesis by method_designer | SUCCESSOR of H-020 | for X-061 arm (b) | FILED BEFORE ANY X-061 RUN,  |
| H-023 | 2 | open | SUCCESSOR TO H-016 (refuted, RS-023). Filed before any code or run.

METHOD v2_walk_head = v2_walk_xy (candidate A) exactly as logged in R-0 |
| H-024 | 3 | open | Successor to H-010 (refuted by RS-019). Filed before any code or run.
PREMISE (RS-019): the release damage happens while the FINGERS OPEN, n |
| H-025 | 5 | open | [Placement tail, combined method | agent hypothesis by method_designer | SUCCESSOR LINE of H-022 (D-018 next_experiment 3) | FILED BEFORE AN |
| H-026 | 5 | open | [Rung 5 perception | agent hypothesis by method_designer | FILED BEFORE ANY CODE OR RUN for v47b]

CLAIM: The remaining perception_structure |
| H-027 | 4 | open | [Rung 4 | SUCCESSOR of H-018 (OPEN, untested: pre-run gate failed at 9cf673d, T0-dev-01 s100: G2c D_floor 2.38 cm, release forced at 8.0 s)  |

## Latest decision (D-018, cites RS-031, RS-030)

FINAL DECISION; STOP (budget). The COMBINED method v6_combo (picture in via v47_vlm_anchor + support_sort order + v2_walk + closed-loop weld place) is now CLEARED on all four tiers. H-022 SUPPORTED (RS-031, reviewer ACCEPT narrowly): X-063 at 15686c1, R-103 T2 46/48 [0.860, 0.988], T3 48/48 [0.926, 1.000] vs fresh v0 R-104 46/48 and 47/48 on identical pairs. V1-V4 pass (E-022); no order failure; difference vs v0 unresolved (McNemar p=1.0). The earlier batch R-102 was void under V1 (procedural, git_head-only), so it's exploratory.
Rung 2 standalone on T2/T3: H-021 PARTIAL (RS-029 + RS-030, ACCE

**Next:** PROPOSED (needs new budget):
(1) Rung 4: H-018 successor with a release-band height term, sliding penalty and tighter target region; gate re-run.
(2) Rung 3: v3_grasp_hold, holding the hand pose during finger-open.
(3) Placement tail: a closed-loop correction aware of previously placed blocks (T2-dev-12 s4023-type pushes), plus the regression of block error on bridged base height/roll/pitch.
(4) W

**Budget:** 2979 of 50000 episodes used, 47021 left. Record: 28 evidence, 27 hypotheses, 74 experiments, 104 runs, 31 results, 18 decisions.

## Recent runs (videos inside)

- R-104 v0 (rung 0), T2, T3: 0.97, failures {"placement_error": 3} - [runs/1791115477_v0/RUN.md](runs/1791115477_v0/RUN.md) [▶ T2-dev-01_s4000](runs/1791115477_v0/T2-dev-01_s4000.gif) [▶ T2-dev-02_s4004](runs/1791115477_v0/T2-dev-02_s4004.gif) [▶ T2-dev-03_s4008](runs/1791115477_v0/T2-dev-03_s4008.gif)
- R-103 v6_combo (rung 5), T2, T3: 0.98, failures {"perception_mismatch": 1, "perception_missing_block": 1, "perception_structure": 1, "placement": 1} - [runs/1791114791_v6_combo/RUN.md](runs/1791114791_v6_combo/RUN.md) [▶ T2-dev-01_s4000](runs/1791114791_v6_combo/T2-dev-01_s4000.gif) [▶ T2-dev-02_s4004](runs/1791114791_v6_combo/T2-dev-02_s4004.gif) [▶ T2-dev-03_s4008](runs/1791114791_v6_combo/T2-dev-03_s4008.gif)
- R-102 v6_combo (rung 5), T2, T3: 0.97, failures {"walk": 1, "perception_mismatch": 2, "perception_missing_block": 2, "perception_structure": 2} - [runs/1791108881_v6_combo/RUN.md](runs/1791108881_v6_combo/RUN.md) [▶ T2-dev-01_s2000](runs/1791108881_v6_combo/T2-dev-01_s2000.gif) [▶ T2-dev-02_s2004](runs/1791108881_v6_combo/T2-dev-02_s2004.gif) [▶ T2-dev-03_s2008](runs/1791108881_v6_combo/T2-dev-03_s2008.gif)
- R-101 v2_walk (rung 2), T2, T3: 0.90, failures {"placement_error": 10} - [runs/1791107775_v2_walk/RUN.md](runs/1791107775_v2_walk/RUN.md) [▶ T2-dev-01_s2000](runs/1791107775_v2_walk/T2-dev-01_s2000.gif) [▶ T2-dev-01_s2002](runs/1791107775_v2_walk/T2-dev-01_s2002.gif) [▶ T2-dev-02_s2004](runs/1791107775_v2_walk/T2-dev-02_s2004.gif)
- R-100 v0 (rung 0), T2, T3: 0.93, failures {"placement_error": 7} - [runs/1791107658_v0/RUN.md](runs/1791107658_v0/RUN.md) [▶ T2-dev-01_s2000](runs/1791107658_v0/T2-dev-01_s2000.gif) [▶ T2-dev-01_s2002](runs/1791107658_v0/T2-dev-01_s2002.gif) [▶ T2-dev-02_s2004](runs/1791107658_v0/T2-dev-02_s2004.gif)
- R-099 v0 (rung 0), T0, T1: 1.00, failures {} - [runs/1791106666_v0/RUN.md](runs/1791106666_v0/RUN.md) [▶ T0-dev-01_s1400](runs/1791106666_v0/T0-dev-01_s1400.gif) [▶ T0-dev-02_s1408](runs/1791106666_v0/T0-dev-02_s1408.gif) [▶ T0-dev-03_s1416](runs/1791106666_v0/T0-dev-03_s1416.gif)
- R-098 v6_combo (rung 5), T0, T1: 1.00, failures {} - [runs/1791104821_v6_combo/RUN.md](runs/1791104821_v6_combo/RUN.md) [▶ T0-dev-01_s1400](runs/1791104821_v6_combo/T0-dev-01_s1400.gif) [▶ T0-dev-02_s1408](runs/1791104821_v6_combo/T0-dev-02_s1408.gif) [▶ T0-dev-03_s1416](runs/1791104821_v6_combo/T0-dev-03_s1416.gif)
- R-097 v6_combo (rung 5), T0, T1: 0.99, failures {"order_infeasible": 1, "perception_missing_block": 1, "order": 1} - [runs/1791101237_v6_combo/RUN.md](runs/1791101237_v6_combo/RUN.md) [▶ T0-dev-01_s400](runs/1791101237_v6_combo/T0-dev-01_s400.gif) [▶ T0-dev-02_s408](runs/1791101237_v6_combo/T0-dev-02_s408.gif) [▶ T0-dev-03_s416](runs/1791101237_v6_combo/T0-dev-03_s416.gif)
