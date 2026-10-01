"""
Inventory Control — AMS value-estimation experiment.

The script first computes the dynamic-programming benchmark value V*(x0),
and then runs value-estimation experiments using four estimators:
    - LSA: largest-size average estimator;
    - WEIGHTED: weighted-average estimator;
    - MAX: maximum/minimum Q-value readout adapted to cost minimization;
    - EA: equal-allocation estimator.

Main settings:
    - horizon H = 3;
    - maximum inventory level M = 20;
    - main experiments use initial inventory x0 = 5;
    - robustness checks use x0 = 10 for setting 1 and x0 = 0 for setting 2;
    - demand D follows the discrete uniform distribution on {0, ..., 9};
    - sample sizes N = 50, 100, ..., 500;
    - 500 independent replications for each estimator and sample size;
    - the same replication seeds are used across estimators for each setting
      and sample size;
    - setting 1 in the paper: K = 0, p = 1;
    - setting 2 in the paper: K = 5, p = 10.
    - robustness 1: K = 0, p = 1, x0 = 10;
    - robustness 2: K = 5, p = 10, x0 = 0.

The simulation uses normalized one-period costs internally, and the reported
values are transformed back to the original cost scale before computing bias,
standard deviation, and MSE.

Output:
    results/value_estimation/inventory_control.csv

Run:
    python scripts/value_estimation/inventory_control.py
"""

import csv
import os
import time
from pathlib import Path

import numpy as np
from numba import njit, prange


# =========================================================
# Configuration
# =========================================================

H = 3
M = 20
H_COST = 1
X0 = 5

# Exploration coefficient in the adaptive sampling index.
# For this cost-minimization example, the allocation rule uses
#     Q_hat - sqrt(2 log n / N_a),
# which is the minimization counterpart of the UCB rule in the paper.
AMS_UCB_EXPL_C = 1.0

# The two inventory settings reported in Section 5.2 of the paper:
#   setting 1: K = 0, p = 1;
#   setting 2: K = 5, p = 10.
# The third component is the action-set type:
#   action_type = 1: feasible order quantities are 0, 1, ..., M - x;
#   action_type = 0: feasible order quantities are restricted to 0 or 10.
# Each configuration is (setting_id, K, p, action_type, x0, experiment_group).
# Settings 1 and 2 reproduce the main experiments in Section 5.2.
# Settings 1R and 2R reproduce the additional-initial-state robustness checks
# in Appendix G.2.
SETTINGS = [
    ("1",  0,  1, 1,  5, "main"),
    ("2",  5, 10, 1,  5, "main"),
    ("1R", 0,  1, 1, 10, "robustness"),
    ("2R", 5, 10, 1,  0, "robustness"),
]

BUDGETS_TYPE1 = np.array(
    [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
    dtype=np.int32,
)

N_REPS = 500
NORMALIZED = True
SETTINGS_TO_RUN = None   # None means run all settings.

ESTIMATORS = ["LSA", "WEIGHTED", "MAX", "EA"]
EST_CODES = {
    "LSA": 1,
    "WEIGHTED": 0,
    "MAX": 2,
    "EA": 3,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "value_estimation"
OUTPUT_FILE = OUTPUT_DIR / "inventory_control.csv"


# =========================================================
# Action sets
# =========================================================

def build_actions(action_type):
    """Build the candidate order-quantity array for a given action-set type."""
    if action_type == 0:
        return np.array([0, 10], dtype=np.int32)
    return np.arange(M + 1, dtype=np.int32)


# =========================================================
# Dynamic-programming benchmark value
# =========================================================

def solve_dp(K, p, action_type):
    """Solve the inventory-control problem by backward induction.

    Returns
    -------
    V : ndarray
        V[t, x] is the optimal expected cost from period t and inventory x.
    pi : ndarray
        pi[t, x] is the optimal order quantity from period t and inventory x.
    """
    actions_list = list(range(M + 1)) if action_type == 1 else [0, 10]
    V = np.zeros((H + 1, M + 1))
    pi = np.zeros((H, M + 1), dtype=np.int32)

    for t in range(H - 1, -1, -1):
        for x in range(M + 1):
            best_cost = 1e18
            best_a = -1
            for a in actions_list:
                if x + a > M:
                    continue
                total = 0.0
                for d in range(10):
                    cost = 0.0
                    if a > 0:
                        cost += K
                    diff = x + a - d
                    if diff > 0:
                        cost += H_COST * diff
                    else:
                        cost -= p * diff
                    nx = max(0, x + a - d)
                    total += cost + V[t + 1, nx]
                total /= 10.0
                if total < best_cost:
                    best_cost = total
                    best_a = a
            V[t, x] = best_cost
            pi[t, x] = best_a

    return V, pi


# =========================================================
# Cost normalization
# =========================================================

def compute_normalize_factor(K, p, action_type):
    """Compute the maximum one-period cost used to normalize simulation costs."""
    actions_list = list(range(M + 1)) if action_type == 1 else [0, 10]
    max_cost = 0.0
    for x in range(M + 1):
        for a in actions_list:
            if x + a > M:
                continue
            for d in range(10):
                cost = 0.0
                if a > 0:
                    cost += K
                diff = x + a - d
                if diff > 0:
                    cost += H_COST * diff
                else:
                    cost -= p * diff
                if cost > max_cost:
                    max_cost = cost
    return float(max_cost)


# =========================================================
# Transition and cost tables
# =========================================================

def build_tables(K_val, p_val, nf):
    """Build immediate-cost and next-state lookup tables."""
    cost_table = np.empty((M + 1, M + 1, 10), dtype=np.float64)
    nstate_table = np.empty((M + 1, M + 1, 10), dtype=np.int32)
    for s in range(M + 1):
        for a in range(M + 1):
            for d in range(10):
                cost = 0.0
                if a > 0:
                    cost += K_val
                diff = s + a - d
                if diff > 0:
                    cost += H_COST * diff
                else:
                    cost -= p_val * diff
                cost_table[s, a, d] = float(cost) / nf
                nstate_table[s, a, d] = max(0, s + a - d)
    return cost_table, nstate_table


# =========================================================
# Adaptive multistage sampling recursion
# =========================================================

@njit(cache=False)
def ams_recursive(stage, state, N, estimator_code,
                  actions, n_actions, cost_table, nstate_table,
                  rx, ry, rz, rw):
    if stage >= H:
        return 0.0, rx, ry, rz, rw

    n_valid = n_actions
    for a_idx in range(n_actions):
        if actions[a_idx] + state > M:
            n_valid = a_idx
            break

    Q = np.empty(n_valid, dtype=np.float64)
    Na = np.zeros(n_valid, dtype=np.int32)
    n = 0
    is_last = (stage + 1 >= H)

    x = rx
    y = ry
    z = rz
    w = rw

    # Phase 1: sample each feasible action once.
    for a_idx in range(n_valid):
        a_val = actions[a_idx]

        t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
        x = y
        y = z
        z = w
        w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)
        d = int(w % np.uint32(10))

        cost_scaled = cost_table[state, a_val, d]
        if is_last:
            v_next = 0.0
        else:
            next_state = nstate_table[state, a_val, d]
            v_next, x, y, z, w = ams_recursive(
                stage + 1, next_state, N, estimator_code,
                actions, n_actions, cost_table, nstate_table,
                x, y, z, w
            )

        Q[a_idx] = cost_scaled + v_next
        Na[a_idx] = 1
        n += 1

    # Phase 2: allocate remaining samples.
    if estimator_code == 3:  # EA: static equal allocation by round-robin.
        cur_a = 0
        while n < N:
            a_val = actions[cur_a]

            t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
            x = y
            y = z
            z = w
            w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)
            d = int(w % np.uint32(10))

            cost_scaled = cost_table[state, a_val, d]
            if is_last:
                v_next = 0.0
            else:
                next_state = nstate_table[state, a_val, d]
                v_next, x, y, z, w = ams_recursive(
                    stage + 1, next_state, N, estimator_code,
                    actions, n_actions, cost_table, nstate_table,
                    x, y, z, w
                )

            vv = cost_scaled + v_next
            Q[cur_a] = (Q[cur_a] * float(Na[cur_a]) + vv) / float(Na[cur_a] + 1)
            Na[cur_a] += 1
            n += 1

            cur_a += 1
            if cur_a >= n_valid:
                cur_a = 0
    else:  # Adaptive LCB allocation for cost minimization.
        while n < N:
            log_n = np.log(float(n))
            two_log = 2.0 * log_n

            min_lcb = 1e18
            best_a_idx = 0
            for a_idx in range(n_valid):
                lcb = Q[a_idx] - AMS_UCB_EXPL_C * np.sqrt(two_log / float(Na[a_idx]))
                if lcb < min_lcb:
                    min_lcb = lcb
                    best_a_idx = a_idx

            a_val = actions[best_a_idx]

            t_val = (x ^ (x << np.uint32(11))) & np.uint32(0xFFFFFFFF)
            x = y
            y = z
            z = w
            w = (w ^ (w >> np.uint32(19)) ^ t_val ^ (t_val >> np.uint32(8))) & np.uint32(0xFFFFFFFF)
            d = int(w % np.uint32(10))

            cost_scaled = cost_table[state, a_val, d]
            if is_last:
                v_next = 0.0
            else:
                next_state = nstate_table[state, a_val, d]
                v_next, x, y, z, w = ams_recursive(
                    stage + 1, next_state, N, estimator_code,
                    actions, n_actions, cost_table, nstate_table,
                    x, y, z, w
                )

            vv = cost_scaled + v_next
            Q[best_a_idx] = (Q[best_a_idx] * float(Na[best_a_idx]) + vv) / float(Na[best_a_idx] + 1)
            Na[best_a_idx] += 1
            n += 1

    # Phase 3: estimator readout.
    if estimator_code == 0:   # WEIGHTED / WA
        total_val = 0.0
        for a_idx in range(n_valid):
            total_val += Q[a_idx] * float(Na[a_idx])
        value = total_val / float(N)

    elif estimator_code == 1:  # LSA
        best_a_idx = 0
        max_na = Na[0]
        for a_idx in range(1, n_valid):
            if Na[a_idx] > max_na:
                max_na = Na[a_idx]
                best_a_idx = a_idx
        value = Q[best_a_idx]

    else:  # MAX / ME, adapted to minimization by selecting the smallest Q value.
        best_val = Q[0]
        for a_idx in range(1, n_valid):
            if Q[a_idx] < best_val:
                best_val = Q[a_idx]
        value = best_val

    return float(value), x, y, z, w


# =========================================================
# Batched value-estimation replications
# =========================================================

@njit(cache=False, parallel=True)
def run_batch_value(seeds, N_budget, estimator_code, x0,
                    actions, n_actions, cost_table, nstate_table):
    n_reps = seeds.shape[0]
    values = np.empty(n_reps, dtype=np.float64)

    for r in prange(n_reps):
        seed = seeds[r]
        rx = np.uint32(seed)
        ry = np.uint32(seed + 1)
        rz = np.uint32(seed + 2)
        rw = np.uint32(seed + 3)

        val, _, _, _, _ = ams_recursive(
            0, x0, N_budget, estimator_code,
            actions, n_actions, cost_table, nstate_table,
            rx, ry, rz, rw
        )
        values[r] = val

    return values


# =========================================================
# Main experiment driver
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    n_threads = os.cpu_count() or 4
    settings_to_run = list(range(len(SETTINGS))) if SETTINGS_TO_RUN is None else SETTINGS_TO_RUN

    print("=" * 100)
    print("Inventory Control — value-only AMS experiments")
    print("=" * 100)
    print(f"H={H}, M={M}, h={H_COST}, reps={N_REPS}")
    print(f"Normalized internal costs: {NORMALIZED}")
    print(f"CPU threads: {n_threads}")
    print(f"Estimators: {ESTIMATORS}")
    print(f"Setting indices: {settings_to_run} -> settings {[SETTINGS[i][0] for i in settings_to_run]}")
    print(f"Output file: {OUTPUT_FILE}")

    # Warmup: compile Numba functions before timing the main runs.
    print("\nWarming up Numba...", end=" ", flush=True)
    t0 = time.time()

    warmup_seeds = np.array([42], dtype=np.int64)
    warmup_a21 = np.arange(M + 1, dtype=np.int32)
    warmup_ct, warmup_nt = build_tables(0, 1, 1.0)

    for _ec in [0, 1, 2, 3]:
        _ = run_batch_value(
            warmup_seeds, 25, _ec, X0,
            warmup_a21, M + 1, warmup_ct, warmup_nt,
        )

    print(f"done ({time.time() - t0:.1f}s)")

    csv_rows = []

    for sid in settings_to_run:
        pid, K_val, p_val, action_type, x0, experiment_group = SETTINGS[sid]
        actions = build_actions(action_type)
        n_actions = len(actions)
        budgets = BUDGETS_TYPE1

        nf = compute_normalize_factor(K_val, p_val, action_type) if NORMALIZED else 1.0
        cost_table, nstate_table = build_tables(K_val, p_val, nf)

        V_dp, pi_dp = solve_dp(K_val, p_val, action_type)
        v_star = V_dp[0, x0]

        actions_desc = "{0, 1, ..., M-x}" if action_type == 1 else "{0, 10} when feasible"

        print(f"\n{'=' * 86}")
        print(f"Setting {pid} [{experiment_group}]: K={K_val}, p={p_val}, x0={x0}, actions={actions_desc}")
        print(f"  V* = {v_star:.6f}, a*(x0={x0}) = {pi_dp[0, x0]}")
        print(f"  normalize_factor = {nf}")
        print(f"  Budgets: {list(budgets)}")
        print(f"{'=' * 86}")

        for est_name in ESTIMATORS:
            est_code = EST_CODES[est_name]
            print(f"\n  VALUE  {est_name} (est_code={est_code})", flush=True)

            for N_budget in budgets:
                N_budget = int(N_budget)
                start = time.time()

                base_seed = 100000 + sid * 100000 + 1000 * N_budget
                seeds = base_seed + np.arange(N_REPS, dtype=np.int64)

                values = run_batch_value(
                    seeds, N_budget, est_code, x0,
                    actions, n_actions, cost_table, nstate_table,
                )

                if NORMALIZED:
                    values = values * nf

                mean_v = float(np.mean(values))
                bias = float(mean_v - v_star)
                stdev = float(np.std(values, ddof=1)) if N_REPS > 1 else 0.0
                mse = float(np.mean((values - v_star) ** 2))
                elapsed = time.time() - start

                print(
                    f"    N={N_budget:4d}  mean={mean_v:.6f}  bias={bias:+.6f}  "
                    f"stdev={stdev:.6f}  mse={mse:.8f}  ({elapsed:.1f}s)",
                    flush=True,
                )

                csv_rows.append({
                    "setting": pid,
                    "experiment_group": experiment_group,
                    "x0": x0,
                    "K": K_val,
                    "p": p_val,
                    "action_type": action_type,
                    "true_value": float(v_star),
                    "estimator": est_name,
                    "N": N_budget,
                    "mean_V": mean_v,
                    "bias": bias,
                    "stdev": stdev,
                    "mse": mse,
                    "value_replications": N_REPS,
                    "H": H,
                })

    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "setting",
                "experiment_group",
                "x0",
                "K",
                "p",
                "action_type",
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
        writer.writerows(csv_rows)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print("Done.")
    return csv_rows


if __name__ == "__main__":
    main()
