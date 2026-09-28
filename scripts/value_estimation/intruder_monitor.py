# -*- coding: utf-8 -*-
"""
Intruder Monitoring — value-estimation AMS experiments.

The script first computes the dynamic-programming benchmark value V*(S0),
and then runs value-estimation experiments using four estimators:
    - lsa: largest-size average estimator;
    - weighted: weighted-average estimator;
    - max: maximum estimator;
    - ea: equal-allocation estimator.

Main settings:
    - horizon H = 6;
    - sample sizes N = 8, 10, ..., 30;
    - 500 independent replications for each estimator and sample size;
    - the same replication seeds are used across estimators for each setting
      and sample size;
    - settings 'a' and 'b': main experiments in Section 5.1;
    - settings 'c' to 'f': robustness checks in Appendix G.1;
    - settings 'a' to 'd' use 3x3 grids; settings 'e' and 'f' use 4x4 grids.

Implementation notes:
    - The same sample size N is used at every stage.
    - The UCB index is Q_hat + sqrt(2 log n / N_a), matching Algorithm 2.
    - EA is a static equal-allocation benchmark.
    
Output:
    results/intruder_value_metrics.csv

Run:
    python scripts/intruder_monitor.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit, prange


# =========================================================
# Configuration
# =========================================================

H = 6

# Exploration coefficient in the UCB index.
# With AMS_UCB_EXPL_C = 1.0, the index is
#     Q_hat + sqrt(2 log n / N_a),
# matching Algorithm 2 in the paper.
AMS_UCB_EXPL_C = 1.0

# The same sample size N is used at every stage in the recursive AMS calls.
N_VALUES = [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]
VALUE_REPLICATIONS = 500

# Order follows the paper tables: LSA, WA, ME, EA.
VALUE_ESTIMATORS = ["lsa", "weighted", "max", "ea"]
VALUE_ESTIMATOR_AMS_CODE = {
    "weighted": 0,  # WA
    "lsa": 1,       # LSA
    "max": 2,       # ME
    "ea": 3,        # EA
}

SETTINGS_TO_RUN = ["a", "b", "c", "d", "e", "f"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "value_estimation"
OUTPUT_FILE = OUTPUT_DIR / "intruder_monitor.csv"

R_DANGER, R_GUARD, R_DEFAULT = -0.1, 1.0, 0.0

DI = np.array([0, -1, 1, 0, 0], dtype=np.int32)
DJ = np.array([0, 0, 0, 1, -1], dtype=np.int32)


SETTING_CONFIGS = {
    # Main experiments in Section 5.1.
    "a": {
        "grid_h": 3, "grid_w": 3,
        "intruder_start": (0, 0), "camera_start": (2, 2),
        "danger_positions": [(1, 1)],
    },
    "b": {
        "grid_h": 3, "grid_w": 3,
        "intruder_start": (0, 0), "camera_start": (2, 2),
        "danger_positions": [(1, 0), (1, 1), (1, 2)],
    },
    # Robustness checks in Appendix G.1.
    "c": {
        "grid_h": 3, "grid_w": 3,
        "intruder_start": (0, 1), "camera_start": (2, 1),
        "danger_positions": [(1, 1)],
    },
    "d": {
        "grid_h": 3, "grid_w": 3,
        "intruder_start": (0, 1), "camera_start": (2, 0),
        "danger_positions": [(1, 0), (1, 1), (1, 2)],
    },
    "e": {
        "grid_h": 4, "grid_w": 4,
        "intruder_start": (0, 0), "camera_start": (3, 3),
        "danger_positions": [(1, 1), (1, 2), (2, 1), (2, 2)],
    },
    "f": {
        "grid_h": 4, "grid_w": 4,
        "intruder_start": (0, 0), "camera_start": (3, 3),
        "danger_positions": [(1, 0), (1, 1), (1, 2), (1, 3)],
    },
}


def get_setting_config(setting):
    """Return the configuration for a given map setting."""
    if setting not in SETTING_CONFIGS:
        raise ValueError(f"Unknown setting: {setting!r}")
    return SETTING_CONFIGS[setting]


def build_danger_mask(setting):
    """Build a 0/1 mask of danger cells for a given map setting."""
    config = get_setting_config(setting)
    positions = list(config["danger_positions"])
    mask = np.zeros((config["grid_h"], config["grid_w"]), dtype=np.int32)
    for i, j in positions:
        mask[i, j] = 1
    return positions, mask


# =========================================================
# Precompute valid moves
# =========================================================

def build_valid_moves_table(grid_h, grid_w):
    """Precompute valid actions for every grid cell."""
    valid_moves = np.full((grid_h, grid_w, 5), -1, dtype=np.int32)
    n_valid = np.zeros((grid_h, grid_w), dtype=np.int32)
    for i in range(grid_h):
        for j in range(grid_w):
            cnt = 0
            for a in range(5):
                ni = i + DI[a]
                nj = j + DJ[a]
                if 0 <= ni < grid_h and 0 <= nj < grid_w:
                    valid_moves[i, j, cnt] = a
                    cnt += 1
            n_valid[i, j] = cnt
    return valid_moves, n_valid




# =========================================================
# Utility functions
# =========================================================

@njit(cache=False)
def apply_action_valid(pos_i, pos_j, action):
    if action == 0:
        return pos_i, pos_j
    elif action == 1:
        return pos_i - 1, pos_j
    elif action == 2:
        return pos_i + 1, pos_j
    elif action == 3:
        return pos_i, pos_j + 1
    else:
        return pos_i, pos_j - 1


@njit(cache=False)
def compute_reward(ii, ij, ci, cj, danger_mask, r_danger, r_guard, r_default):
    if danger_mask[ii, ij] != 0:
        if ii == ci and ij == cj:
            return r_guard
        return r_danger
    return r_default


@njit(cache=False)
def _get_q(Q0, Q1, Q2, Q3, Q4, idx):
    if idx == 0:
        return Q0
    if idx == 1:
        return Q1
    if idx == 2:
        return Q2
    if idx == 3:
        return Q3
    return Q4


@njit(cache=False)
def _get_na(Na0, Na1, Na2, Na3, Na4, idx):
    if idx == 0:
        return Na0
    if idx == 1:
        return Na1
    if idx == 2:
        return Na2
    if idx == 3:
        return Na3
    return Na4


# =========================================================
# Adaptive multistage sampling recursion
# =========================================================

@njit(cache=False)
def ams_recursive(t, ii, ij, ci, cj, N, estimator_code, H,
                  danger_mask, r_danger, r_guard, r_default,
                  rx, ry, rz, rw,
                  valid_moves, n_valid):
    if t >= H:
        return 0.0, 0, rx, ry, rz, rw

    n_cam_actions = n_valid[ci, cj]
    n_int_actions = n_valid[ii, ij]

    if N < n_cam_actions:
        N = n_cam_actions

    Q0 = 0.0
    Q1 = 0.0
    Q2 = 0.0
    Q3 = 0.0
    Q4 = 0.0
    Na0 = 0
    Na1 = 0
    Na2 = 0
    Na3 = 0
    Na4 = 0

    x = rx
    y = ry
    z = rz
    w = rw

    total_samples = 0

    # Phase 1: sample each camera action once.
    for ca_idx in range(n_cam_actions):
        ca = valid_moves[ci, cj, ca_idx]
        nci, ncj = apply_action_valid(ci, cj, ca)

        t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
        x = y
        y = z
        z = w
        w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)

        ia_idx = int(w % np.uint32(n_int_actions))
        ia = valid_moves[ii, ij, ia_idx]
        nii, nij = apply_action_valid(ii, ij, ia)

        r = compute_reward(nii, nij, nci, ncj, danger_mask, r_danger, r_guard, r_default)

        if t + 1 < H:
            v_next, _, x, y, z, w = ams_recursive(
                t + 1, nii, nij, nci, ncj, N, estimator_code, H,
                danger_mask, r_danger, r_guard, r_default,
                x, y, z, w, valid_moves, n_valid
            )
        else:
            v_next = 0.0

        sample_val = r + v_next

        if ca_idx == 0:
            Q0 = sample_val
            Na0 = 1
        elif ca_idx == 1:
            Q1 = sample_val
            Na1 = 1
        elif ca_idx == 2:
            Q2 = sample_val
            Na2 = 1
        elif ca_idx == 3:
            Q3 = sample_val
            Na3 = 1
        else:
            Q4 = sample_val
            Na4 = 1

        total_samples += 1

    # Phase 2: allocate remaining samples.
    if estimator_code == 3:  # EA: static equal allocation.
        tgt_base = N // n_cam_actions
        tgt_rem = N % n_cam_actions
        while total_samples < N:
            for ca_idx in range(n_cam_actions):
                if total_samples >= N:
                    break
                need = tgt_base + (1 if ca_idx < tgt_rem else 0)
                na = _get_na(Na0, Na1, Na2, Na3, Na4, ca_idx)
                if na >= need:
                    continue

                ca = valid_moves[ci, cj, ca_idx]
                nci, ncj = apply_action_valid(ci, cj, ca)

                t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
                x = y
                y = z
                z = w
                w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)

                ia_idx = int(w % np.uint32(n_int_actions))
                ia = valid_moves[ii, ij, ia_idx]
                nii, nij = apply_action_valid(ii, ij, ia)

                r = compute_reward(nii, nij, nci, ncj, danger_mask, r_danger, r_guard, r_default)

                if t + 1 < H:
                    v_next, _, x, y, z, w = ams_recursive(
                        t + 1, nii, nij, nci, ncj, N, estimator_code, H,
                        danger_mask, r_danger, r_guard, r_default,
                        x, y, z, w, valid_moves, n_valid
                    )
                else:
                    v_next = 0.0

                sample_val = r + v_next

                if ca_idx == 0:
                    Na0 += 1
                    Q0 += (sample_val - Q0) / float(Na0)
                elif ca_idx == 1:
                    Na1 += 1
                    Q1 += (sample_val - Q1) / float(Na1)
                elif ca_idx == 2:
                    Na2 += 1
                    Q2 += (sample_val - Q2) / float(Na2)
                elif ca_idx == 3:
                    Na3 += 1
                    Q3 += (sample_val - Q3) / float(Na3)
                else:
                    Na4 += 1
                    Q4 += (sample_val - Q4) / float(Na4)

                total_samples += 1
    else:  # UCB sampling for WA, LSA, and ME.
        while total_samples < N:
            log_total = np.log(float(total_samples))
            two_log = 2.0 * log_total

            max_ucb = -1e300
            best_ca_idx = 0
            for ca_idx in range(n_cam_actions):
                qa = _get_q(Q0, Q1, Q2, Q3, Q4, ca_idx)
                na = _get_na(Na0, Na1, Na2, Na3, Na4, ca_idx)
                ucb = qa + AMS_UCB_EXPL_C * np.sqrt(two_log / float(na))
                if ucb > max_ucb:
                    max_ucb = ucb
                    best_ca_idx = ca_idx

            ca = valid_moves[ci, cj, best_ca_idx]
            nci, ncj = apply_action_valid(ci, cj, ca)

            t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
            x = y
            y = z
            z = w
            w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)

            ia_idx = int(w % np.uint32(n_int_actions))
            ia = valid_moves[ii, ij, ia_idx]
            nii, nij = apply_action_valid(ii, ij, ia)

            r = compute_reward(nii, nij, nci, ncj, danger_mask, r_danger, r_guard, r_default)

            if t + 1 < H:
                v_next, _, x, y, z, w = ams_recursive(
                    t + 1, nii, nij, nci, ncj, N, estimator_code, H,
                    danger_mask, r_danger, r_guard, r_default,
                    x, y, z, w, valid_moves, n_valid
                )
            else:
                v_next = 0.0

            sample_val = r + v_next

            if best_ca_idx == 0:
                Na0 += 1
                Q0 += (sample_val - Q0) / float(Na0)
            elif best_ca_idx == 1:
                Na1 += 1
                Q1 += (sample_val - Q1) / float(Na1)
            elif best_ca_idx == 2:
                Na2 += 1
                Q2 += (sample_val - Q2) / float(Na2)
            elif best_ca_idx == 3:
                Na3 += 1
                Q3 += (sample_val - Q3) / float(Na3)
            else:
                Na4 += 1
                Q4 += (sample_val - Q4) / float(Na4)

            total_samples += 1

    # Phase 3: estimator readout.
    if estimator_code == 0:  # weighted / WA
        value = 0.0
        for idx in range(n_cam_actions):
            value += _get_q(Q0, Q1, Q2, Q3, Q4, idx) * float(_get_na(Na0, Na1, Na2, Na3, Na4, idx))
        value /= float(N)

        action_idx = 0
        max_q = Q0
        if n_cam_actions > 1 and Q1 > max_q:
            max_q = Q1
            action_idx = 1
        if n_cam_actions > 2 and Q2 > max_q:
            max_q = Q2
            action_idx = 2
        if n_cam_actions > 3 and Q3 > max_q:
            max_q = Q3
            action_idx = 3
        if n_cam_actions > 4 and Q4 > max_q:
            action_idx = 4

    elif estimator_code == 1:  # LSA
        action_idx = 0
        max_n = Na0
        if n_cam_actions > 1 and Na1 > max_n:
            max_n = Na1
            action_idx = 1
        if n_cam_actions > 2 and Na2 > max_n:
            max_n = Na2
            action_idx = 2
        if n_cam_actions > 3 and Na3 > max_n:
            max_n = Na3
            action_idx = 3
        if n_cam_actions > 4 and Na4 > max_n:
            action_idx = 4
        value = _get_q(Q0, Q1, Q2, Q3, Q4, action_idx)

    else:  # max / ME
        action_idx = 0
        max_q = Q0
        if n_cam_actions > 1 and Q1 > max_q:
            max_q = Q1
            action_idx = 1
        if n_cam_actions > 2 and Q2 > max_q:
            max_q = Q2
            action_idx = 2
        if n_cam_actions > 3 and Q3 > max_q:
            max_q = Q3
            action_idx = 3
        if n_cam_actions > 4 and Q4 > max_q:
            max_q = Q4
            action_idx = 4
        value = max_q

    action = valid_moves[ci, cj, action_idx]
    return float(value), int(action), x, y, z, w


# =========================================================
# Dynamic-programming benchmark value
# =========================================================

def solve_dp(H=6, danger_mask=None, grid_h=None, grid_w=None, intruder_start=None, camera_start=None):
    """Compute the dynamic-programming benchmark value V*(S0)."""
    if danger_mask is None:
        raise ValueError("solve_dp requires danger_mask")

    if grid_h is None or grid_w is None or intruder_start is None or camera_start is None:
        raise ValueError("solve_dp requires grid dimensions and initial positions")

    cache = {}

    def inmap(x, y):
        return 0 <= x < grid_h and 0 <= y < grid_w

    def get_valid_actions(i, j):
        return [a for a in range(5) if inmap(i + DI[a], j + DJ[a])]

    def get_return(ii, ij, ci, cj):
        if danger_mask[ii, ij] != 0:
            if ci == ii and cj == ij:
                return R_GUARD
            return R_DANGER
        return R_DEFAULT

    def work(ii, ij, ci, cj, stage):
        key = (ii, ij, ci, cj, stage)
        if key in cache:
            return cache[key]
        if stage == H:
            cache[key] = 0.0
            return 0.0

        intruder_actions = get_valid_actions(ii, ij)
        camera_actions = get_valid_actions(ci, cj)

        best_val = -1e12
        for ca in camera_actions:
            nci = ci + DI[ca]
            ncj = cj + DJ[ca]
            v = 0.0
            for ia in intruder_actions:
                nii = ii + DI[ia]
                nij = ij + DJ[ia]
                v += work(nii, nij, nci, ncj, stage + 1) + get_return(nii, nij, nci, ncj)
            v /= len(intruder_actions)
            if v > best_val:
                best_val = v

        cache[key] = best_val
        return best_val

    v_star = work(intruder_start[0], intruder_start[1], camera_start[0], camera_start[1], 0)
    return v_star, cache


# =========================================================
# Batched value-estimation replications
# =========================================================

@njit(cache=False, parallel=True)
def run_batch_value(seeds, N, estimator_code, H_val,
                    s0_ii, s0_ij, s0_ci, s0_cj,
                    danger_mask, r_danger, r_guard, r_default,
                    valid_moves, n_valid):
    n_reps = seeds.shape[0]
    values = np.empty(n_reps, dtype=np.float64)

    for r in prange(n_reps):
        seed = seeds[r]
        rx = np.uint32(seed)
        ry = np.uint32(seed + 1)
        rz = np.uint32(seed + 2)
        rw = np.uint32(seed + 3)

        val, _, _, _, _, _ = ams_recursive(
            0, s0_ii, s0_ij, s0_ci, s0_cj, N, estimator_code, H_val,
            danger_mask, r_danger, r_guard, r_default,
            rx, ry, rz, rw, valid_moves, n_valid
        )
        values[r] = val

    return values


# =========================================================
# Reporting utilities
# =========================================================

def _col_w(keys, min_w=10):
    return max(min_w, max(len(str(k)) for k in keys) + 2)


def print_intruder_value_tables(
    setting_char,
    v0,
    n_values,
    value_estimators,
    value_results,
):
    cw_v = _col_w(value_estimators)
    sep = "=" * 88

    def hdr_val():
        return f"{'N':>6s}" + "".join(f"{e:>{cw_v}s}" for e in value_estimators)

    print("\n" + sep, flush=True)
    print(
        f"SETTING {setting_char!r}  |  V*(S0) = {v0:.8f}  |  "
        f"value reps = {VALUE_REPLICATIONS}",
        flush=True,
    )
    print(sep, flush=True)

    print("\n[1] ESTIMATED VALUE", flush=True)
    print(hdr_val(), flush=True)
    print("-" * (6 + cw_v * len(value_estimators)), flush=True)
    for n in n_values:
        row = f"{n:6d}"
        for e in value_estimators:
            d = value_results[e][n]
            row += f"{d['mean_V']:>{cw_v}.6f}"
        print(row, flush=True)

    print("\n[2] BIAS", flush=True)
    print(hdr_val(), flush=True)
    print("-" * (6 + cw_v * len(value_estimators)), flush=True)
    for n in n_values:
        row = f"{n:6d}"
        for e in value_estimators:
            d = value_results[e][n]
            row += f"{d['bias']:>{cw_v}.6f}"
        print(row, flush=True)

    print("\n[3] STDEV", flush=True)
    print(hdr_val(), flush=True)
    print("-" * (6 + cw_v * len(value_estimators)), flush=True)
    for n in n_values:
        row = f"{n:6d}"
        for e in value_estimators:
            d = value_results[e][n]
            row += f"{d['stdev']:>{cw_v}.6f}"
        print(row, flush=True)

    print("\n[4] MSE", flush=True)
    print(hdr_val(), flush=True)
    print("-" * (6 + cw_v * len(value_estimators)), flush=True)
    for n in n_values:
        row = f"{n:6d}"
        for e in value_estimators:
            d = value_results[e][n]
            row += f"{d['mse']:>{cw_v}.6f}"
        print(row, flush=True)

    print(sep + "\n", flush=True)


# =========================================================
# Main experiment driver
# =========================================================

def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_threads = os.cpu_count() or 4
    settings_list = list(SETTINGS_TO_RUN)

    print("=" * 100, flush=True)
    print("Intruder Monitoring AMS — value only", flush=True)
    print("=" * 100, flush=True)
    print(f"SETTINGS_TO_RUN = {settings_list}", flush=True)
    print(
        f"H={H}, N_values={N_VALUES}, VALUE_REPLICATIONS={VALUE_REPLICATIONS}",
        flush=True,
    )
    print(f"Value estimators: {VALUE_ESTIMATORS}", flush=True)
    print(f"CPU threads available: {n_threads}", flush=True)
    print(f"Output file: {OUTPUT_FILE}", flush=True)

    # Warmup: solve DP once and compile Numba functions before timing the main runs.
    warmup_setting = settings_list[0]
    warmup_config = get_setting_config(warmup_setting)
    pos0, dm0 = build_danger_mask(warmup_setting)
    valid_moves0, n_valid0 = build_valid_moves_table(warmup_config["grid_h"], warmup_config["grid_w"])
    intruder_start0 = warmup_config["intruder_start"]
    camera_start0 = warmup_config["camera_start"]
    print(f"\n[Warmup] using setting {warmup_setting!r}, danger_positions={pos0}", flush=True)

    print("[Warmup] solve_dp ...", flush=True)
    _v0, _ = solve_dp(
        H, dm0, warmup_config["grid_h"], warmup_config["grid_w"],
        intruder_start0, camera_start0,
    )
    print(f"[Warmup] V*={_v0:.6f}", flush=True)

    print("[Warmup] compiling Numba ...", flush=True)
    warmup_seeds = np.array([42], dtype=np.int64)
    for est in VALUE_ESTIMATORS:
        ec = VALUE_ESTIMATOR_AMS_CODE[est]
        _ = run_batch_value(
            warmup_seeds, 8, ec, H,
            intruder_start0[0], intruder_start0[1], camera_start0[0], camera_start0[1],
            dm0, R_DANGER, R_GUARD, R_DEFAULT,
            valid_moves0, n_valid0,
        )
    print("[Warmup] done", flush=True)

    by_setting_output = {}

    for setting_char in settings_list:
        config = get_setting_config(setting_char)
        danger_positions, danger_mask = build_danger_mask(setting_char)
        valid_moves, n_valid = build_valid_moves_table(config["grid_h"], config["grid_w"])
        intruder_start = config["intruder_start"]
        camera_start = config["camera_start"]
        print("\n" + "#" * 100)
        print(
            f"# SETTING {setting_char.upper()}  grid={config['grid_h']}x{config['grid_w']}  "
            f"intruder_start={intruder_start}  camera_start={camera_start}  "
            f"danger_positions={danger_positions}"
        )
        print("#" * 100)

        print("\n[DP] Solving...", flush=True)
        v0, _ = solve_dp(
            H, danger_mask, config["grid_h"], config["grid_w"],
            intruder_start, camera_start,
        )
        print(f"S0: DP V*={v0:.6f}", flush=True)

        value_results = {e: {} for e in VALUE_ESTIMATORS}
        sch = ord(setting_char[0])

        print(f"\n--- Value phase ({VALUE_REPLICATIONS} AMS replications per cell) ---", flush=True)
        for est in VALUE_ESTIMATORS:
            ams_ec = int(VALUE_ESTIMATOR_AMS_CODE[est])
            print(f"\n  VALUE  {est} (ams_code={ams_ec})", flush=True)

            for N in N_VALUES:
                t0 = time.time()
                base_seed = 500000 + sch * 200000 + 1000 * int(N)
                seeds = base_seed + np.arange(VALUE_REPLICATIONS, dtype=np.int64)

                values = run_batch_value(
                    seeds, N, ams_ec, H,
                    intruder_start[0], intruder_start[1], camera_start[0], camera_start[1],
                    danger_mask, R_DANGER, R_GUARD, R_DEFAULT,
                    valid_moves, n_valid,
                )

                mean_v = float(np.mean(values))
                bias = float(mean_v - v0)
                stdev = float(np.std(values, ddof=1)) if VALUE_REPLICATIONS > 1 else 0.0
                mse = float(np.mean((values - v0) ** 2))

                value_results[est][int(N)] = {
                    "mean_V": mean_v,
                    "bias": bias,
                    "stdev": stdev,
                    "mse": mse,
                }

                print(
                    f"    N={N:3d}  mean={mean_v:.6f}  bias={bias:+.6f}  "
                    f"stdev={stdev:.6f}  mse={mse:.8f}  ({time.time()-t0:.1f}s)",
                    flush=True,
                )

        print_intruder_value_tables(
            setting_char,
            v0,
            N_VALUES,
            VALUE_ESTIMATORS,
            value_results,
        )

        by_setting_output[setting_char] = {
            "config": {
                "setting": setting_char,
                "grid_h": config["grid_h"],
                "grid_w": config["grid_w"],
                "intruder_start": intruder_start,
                "camera_start": camera_start,
                "danger_positions": danger_positions,
                "H": H,
                "N_values": N_VALUES,
                "value_replications": VALUE_REPLICATIONS,
                "value_estimators": VALUE_ESTIMATORS,
            },
            "ground_truth": {
                "value": float(v0),
            },
            "results": {
                "value": value_results,
            },
        }

        # -------------------------------------------------
        # Save all completed settings immediately.
        # This overwrites the CSV with the complete set of
        # results available so far, avoiding duplicate rows.
        # -------------------------------------------------
        rows = []

        for completed_setting in settings_list:
            if completed_setting not in by_setting_output:
                continue

            block = by_setting_output[completed_setting]
            completed_v0 = float(block["ground_truth"]["value"])
            value_block = block["results"]["value"]

            for est in VALUE_ESTIMATORS:
                for N in N_VALUES:
                    d = value_block[est][N]
                    rows.append({
                        "setting": completed_setting,
                        "true_value": completed_v0,
                        "estimator": est,
                        "N": N,
                        "mean_V": d["mean_V"],
                        "bias": d["bias"],
                        "stdev": d["stdev"],
                        "mse": d["mse"],
                        "value_replications": VALUE_REPLICATIONS,
                        "H": H,
                    })

        with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "setting",
                    "true_value",
                    "estimator",
                    "N",
                    "mean_V",
                    "bias",
                    "stdev",
                    "mse",
                    "value_replications",
                    "H",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(
            f"[Checkpoint] Setting {setting_char.upper()} completed. "
            f"Results saved to {OUTPUT_FILE}",
            flush=True,
        )

    print(
        f"\nAll requested settings completed. Final results saved to {OUTPUT_FILE}",
        flush=True,
    )
    return rows


if __name__ == "__main__":
    main()
