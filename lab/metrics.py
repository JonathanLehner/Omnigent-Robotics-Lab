"""Success criteria and metrics. FROZEN: fixed by the team before agents start.

Changing this file changes what "success" means, so run_sim_batch refuses to run when its
hash differs from record/frozen_eval.sha256 (re-freeze needs human sign-off: `lab freeze-eval`).
"""

import math

import numpy as np

POS_TOL_M = 0.03  # each block within 3 cm of its target position
ANG_TOL_DEG = 10.0  # ... and within 10 degrees of its target orientation
SETTLE_S = 5.0  # structure must still meet both 5 s after the last release


def quat_angle_deg(q1, q2) -> float:
    """Smallest rotation between two wxyz quaternions."""
    d = abs(float(np.dot(q1, q2)) / (np.linalg.norm(q1) * np.linalg.norm(q2)))
    return math.degrees(2 * math.acos(min(1.0, d)))


def yaw_quat(yaw_deg: float) -> np.ndarray:
    h = math.radians(yaw_deg) / 2
    return np.array([math.cos(h), 0.0, 0.0, math.sin(h)])


def block_errors(target_pos, target_yaw_deg, final_pos, final_quat) -> tuple[float, float]:
    """Position error (m) and orientation error (deg). Boxes are 180-degree yaw-symmetric."""
    pos_err = float(np.linalg.norm(np.asarray(final_pos) - np.asarray(target_pos)))
    ang_err = min(quat_angle_deg(yaw_quat(target_yaw_deg + k), final_quat) for k in (0.0, 180.0))
    return pos_err, ang_err


def episode_success(per_block: list[dict]) -> bool:
    return bool(per_block) and all(b["pos_err"] <= POS_TOL_M and b["ang_err"] <= ANG_TOL_DEG for b in per_block)


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a success rate."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def summarize(episodes: list[dict]) -> dict:
    """Aggregate episode dicts (from pipeline.run_episode) into batch metrics."""
    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    blocks = [b for e in episodes for b in e["blocks"]]
    failures: dict[str, int] = {}
    for e in episodes:
        for f in e["failures"]:
            failures[f] = failures.get(f, 0) + 1
    by_tier: dict[str, list[int]] = {}
    for e in episodes:
        by_tier.setdefault(e["tier"], [0, 0])
        by_tier[e["tier"]][0] += e["success"]
        by_tier[e["tier"]][1] += 1
    return {
        "episodes": n,
        "success_rate": k / n if n else 0.0,
        "success_ci95": wilson_ci(k, n),
        "block_success_rate": (sum(b["pos_err"] <= POS_TOL_M and b["ang_err"] <= ANG_TOL_DEG for b in blocks) / len(blocks))
        if blocks
        else 0.0,
        "median_pos_err_cm": float(np.median([b["pos_err"] for b in blocks]) * 100) if blocks else None,
        "median_ang_err_deg": float(np.median([b["ang_err"] for b in blocks])) if blocks else None,
        "failure_categories": failures,
        "by_tier": {t: {"success": s, "n": m, "rate": s / m} for t, (s, m) in sorted(by_tier.items())},
        "criteria": {"pos_tol_m": POS_TOL_M, "ang_tol_deg": ANG_TOL_DEG, "settle_s": SETTLE_S},
    }


if __name__ == "__main__":
    assert block_errors([0, 0, 0], 0, [0.01, 0, 0], yaw_quat(180))[1] < 1e-4  # yaw symmetry
    assert abs(block_errors([0, 0, 0], 0, [0, 0, 0], yaw_quat(15))[1] - 15) < 1e-4
    lo, hi = wilson_ci(8, 10)
    assert 0.4 < lo < 0.8 < hi < 1
    print("metrics ok")
