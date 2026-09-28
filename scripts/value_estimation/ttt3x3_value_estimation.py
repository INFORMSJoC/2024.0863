"""
3x3 Tic-Tac-Toe — value-estimation AMS experiments.

This script reproduces the 3x3 Tic-Tac-Toe value-estimation experiment.

The script first computes the exact benchmark value V*(s0) by expectimax,
where Player 1 follows a uniform random policy and Player 2 optimizes its
moves. It then runs AMS value-estimation experiments using three estimators:
    - lsa: largest-size average estimator;
    - max: maximum estimator;
    - weighted: weighted-average estimator.

Main settings:
    - board size: 3x3;
    - initial board: Player 1 has placed X at position 0;
    - Player 1 policy: uniform random over feasible moves;
    - Player 2: AMS planning;
    - sample sizes N = 20, 30, 40, 50, 60, 70, 80;
    - 500 independent replications for each estimator and sample size.

Output:
    results/ttt3x3_value_summary.csv

Run:
    python scripts/ttt3x3_value.py
"""

import csv
import math
import os
import time
import concurrent.futures
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.typed import Dict
from tqdm import tqdm


# =========================================================
# Configuration
# =========================================================

BOARD_SIZE = 9

# This depth is sufficient to cover the remaining game from the initial board.
DEPTH = 4

SAMPLE_SIZES = [20, 30, 40, 50, 60, 70, 80]
VALUE_REPLICATIONS = 500

VALUE_ESTIMATORS = ["lsa", "max", "weighted"]
_EST_CODE = {
    "max": 0,
    "lsa": 1,
    "weighted": 2,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "value_estimation"
SUMMARY_FILE = OUTPUT_DIR / "ttt3x3.csv"

WIN_LINES = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
], dtype=np.int32)


# =========================================================
# Numba core
# =========================================================

@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


@njit(cache=True)
def check_winner_nb(board):
    """Return 1 if P1 wins, 2 if P2 wins, 0 if draw, and -1 if ongoing."""
    for k in range(8):
        a = WIN_LINES[k, 0]
        b = WIN_LINES[k, 1]
        c = WIN_LINES[k, 2]
        v = board[a]
        if v != 0 and v == board[b] and v == board[c]:
            return int(v)
    for i in range(BOARD_SIZE):
        if board[i] == 0:
            return -1
    return 0


@njit(cache=True)
def get_actions_nb(board, buf):
    n = 0
    for i in range(BOARD_SIZE):
        if board[i] == 0:
            buf[n] = i
            n += 1
    return n


@njit(cache=True)
def p1_move_nb(board, buf):
    n = get_actions_nb(board, buf)
    return buf[np.random.randint(0, n)]


@njit(cache=True)
def p1_move_probs_nb(board, act_buf, prob_buf):
    n = get_actions_nb(board, act_buf)
    for i in range(n):
        prob_buf[i] = 1.0 / float(n)
    return n


@njit(cache=True)
def do_move_nb(board, action):
    """
    Place P2 at the selected action. If the game is ongoing, Player 1
    responds uniformly at random.

    Returns
    -------
    reward : float
        Terminal reward if the game ends after P2 or P1 moves; otherwise 0.0.
    p1_action : int
        Player 1's response action when the game remains ongoing; otherwise -1.
    """
    board[action] = 2
    res = check_winner_nb(board)
    if res == 2:
        board[action] = 0
        return 1.0, -1
    if res == 0:
        board[action] = 0
        return 0.5, -1

    buf = np.empty(BOARD_SIZE, dtype=np.int32)
    get_actions_nb(board, buf)
    p1_a = p1_move_nb(board, buf)
    board[p1_a] = 1
    res2 = check_winner_nb(board)

    if res2 == 1:
        board[p1_a] = 0
        board[action] = 0
        return 0.0, -1
    if res2 == 0:
        board[p1_a] = 0
        board[action] = 0
        return 0.5, -1

    return 0.0, p1_a


@njit(cache=True)
def estimate_nb(board, samples, estimator_code, depth):
    """
    Recursive AMS estimation for non-root states.

    estimator_code:
        0 = max
        1 = lsa
        2 = weighted
    """
    if depth <= 0:
        return 0.5

    res = check_winner_nb(board)
    if res != -1:
        if res == 2:
            return 1.0
        elif res == 0:
            return 0.5
        else:
            return 0.0

    act_buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = get_actions_nb(board, act_buf)

    if n_actions == 1:
        a = act_buf[0]
        reward, p1_a = do_move_nb(board, a)
        if p1_a >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_a] = 0
            board[a] = 0
        return reward

    if samples < n_actions:
        samples = n_actions

    Q = np.zeros(n_actions, dtype=np.float64)
    Na = np.zeros(n_actions, dtype=np.int32)
    total_N = 0

    # Phase 1: sample each feasible P2 action once.
    for i in range(n_actions):
        a = act_buf[i]
        reward, p1_a = do_move_nb(board, a)
        if p1_a >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_a] = 0
            board[a] = 0
        Q[i] = reward
        Na[i] = 1
        total_N += 1

    # Phase 2: allocate remaining samples using UCB.
    while total_N < samples:
        best_ucb = -1.0e18
        best_idx = 0
        for i in range(n_actions):
            q_avg = Q[i] / float(Na[i])
            ucb = q_avg + math.sqrt(2.0 * math.log(float(total_N)) / float(Na[i]))
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        a = act_buf[best_idx]
        reward, p1_a = do_move_nb(board, a)
        if p1_a >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_a] = 0
            board[a] = 0
        Q[best_idx] += reward
        Na[best_idx] += 1
        total_N += 1

    # Phase 3: estimator readout.
    if estimator_code == 0:  # max / ME
        best_q = -1.0e18
        for i in range(n_actions):
            q_avg = Q[i] / float(Na[i])
            if q_avg > best_q:
                best_q = q_avg
        return best_q

    elif estimator_code == 1:  # LSA; ties in sample counts are broken by Q-value.
        max_n = 0
        for i in range(n_actions):
            if Na[i] > max_n:
                max_n = Na[i]

        best_q = -1.0e18
        for i in range(n_actions):
            if Na[i] == max_n:
                q_avg = Q[i] / float(Na[i])
                if q_avg > best_q:
                    best_q = q_avg
        return best_q

    else:  # weighted / WA
        v = 0.0
        for i in range(n_actions):
            v += Q[i]
        return v / float(total_N)


# =========================================================
# Exact expectimax benchmark
# =========================================================

@njit(cache=True)
def board_to_key(board):
    key = np.int64(0)
    for i in range(BOARD_SIZE):
        key = key * 3 + np.int64(board[i])
    return key


@njit(cache=True)
def exact_expectimax_nb(board, cache):
    key = board_to_key(board)
    if key in cache:
        return cache[key]

    res = check_winner_nb(board)
    if res != -1:
        if res == 2:
            val = 1.0
        elif res == 0:
            val = 0.5
        else:
            val = 0.0
        cache[key] = val
        return val

    p2_buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n_p2 = get_actions_nb(board, p2_buf)

    best_val = -1.0
    for i in range(n_p2):
        a = p2_buf[i]
        board[a] = 2

        res2 = check_winner_nb(board)
        if res2 == 2:
            val = 1.0
        elif res2 == 0:
            val = 0.5
        else:
            p1_buf = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buf, p1_probs)

            val = 0.0
            for j in range(n_p1):
                p1_a = p1_buf[j]
                board[p1_a] = 1

                res3 = check_winner_nb(board)
                if res3 == 1:
                    v_child = 0.0
                elif res3 == 0:
                    v_child = 0.5
                else:
                    v_child = exact_expectimax_nb(board, cache)

                val += p1_probs[j] * v_child
                board[p1_a] = 0

        board[a] = 0
        if val > best_val:
            best_val = val

    cache[key] = best_val
    return best_val


@njit(cache=True)
def exact_q_values_nb(board, cache):
    buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n = get_actions_nb(board, buf)
    actions = np.empty(n, dtype=np.int32)
    q_vals = np.empty(n, dtype=np.float64)

    best_a = -1
    best_q = -1.0

    for i in range(n):
        a = buf[i]
        actions[i] = a
        board[a] = 2

        res = check_winner_nb(board)
        if res == 2:
            q_vals[i] = 1.0
        elif res == 0:
            q_vals[i] = 0.5
        else:
            p1_buf = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buf, p1_probs)

            ev = 0.0
            for j in range(n_p1):
                p1_a = p1_buf[j]
                board[p1_a] = 1

                res2 = check_winner_nb(board)
                if res2 == 1:
                    v_child = 0.0
                elif res2 == 0:
                    v_child = 0.5
                else:
                    v_child = exact_expectimax_nb(board, cache)

                ev += p1_probs[j] * v_child
                board[p1_a] = 0

            q_vals[i] = ev

        board[a] = 0
        if q_vals[i] > best_q:
            best_q = q_vals[i]
            best_a = a

    return best_a, best_q, actions, q_vals


# =========================================================
# Python helpers
# =========================================================

def make_initial_board():
    """
    Construct the initial board:

        X  .  .
        .  .  .
        .  .  .

    Player 2 moves next.
    """
    board = np.zeros(BOARD_SIZE, dtype=np.int8)
    board[0] = 1
    return board


# =========================================================
# Root-level AMS estimation
# =========================================================

def estimate_root(board_np, samples, estimator, depth=DEPTH):
    est_code = _EST_CODE[estimator]

    res = check_winner_nb(board_np)
    if res != -1:
        return 1.0 if res == 2 else (0.5 if res == 0 else 0.0)

    act_buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = int(get_actions_nb(board_np, act_buf))
    actions = [int(act_buf[i]) for i in range(n_actions)]

    if n_actions == 1:
        a = actions[0]
        reward, p1_a = do_move_nb(board_np, a)
        if p1_a >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_a] = 0
            board_np[a] = 0
        return float(reward)

    if samples < n_actions:
        samples = n_actions

    Q = np.zeros(n_actions, dtype=np.float64)
    Na = np.zeros(n_actions, dtype=np.int32)
    total_N = 0

    # Phase 1: sample each feasible P2 action once.
    for i in range(n_actions):
        a = actions[i]
        reward, p1_a = do_move_nb(board_np, a)
        if p1_a >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_a] = 0
            board_np[a] = 0
        else:
            reward = float(reward)
        Q[i] = reward
        Na[i] = 1
        total_N += 1

    # Phase 2: allocate remaining samples using UCB.
    while total_N < samples:
        best_ucb = -1.0e18
        best_idx = 0
        for i in range(n_actions):
            q_avg = Q[i] / float(Na[i])
            ucb = q_avg + math.sqrt(2.0 * math.log(float(total_N)) / float(Na[i]))
            if ucb > best_ucb:
                best_ucb = ucb
                best_idx = i

        a = actions[best_idx]
        reward, p1_a = do_move_nb(board_np, a)
        if p1_a >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_a] = 0
            board_np[a] = 0
        else:
            reward = float(reward)
        Q[best_idx] += reward
        Na[best_idx] += 1
        total_N += 1

    Q_avgs = [float(Q[i] / Na[i]) for i in range(n_actions)]
    N_counts = [int(Na[i]) for i in range(n_actions)]

    # Phase 3: estimator readout.
    if estimator == "max":
        selected_idx = Q_avgs.index(max(Q_avgs))
        V = Q_avgs[selected_idx]

    elif estimator == "lsa":
        max_N = max(N_counts)
        candidates = [j for j, nc in enumerate(N_counts) if nc == max_N]
        selected_idx = max(candidates, key=lambda j: Q_avgs[j])
        V = Q_avgs[selected_idx]

    elif estimator == "weighted":
        V = sum(Q_avgs[j] * N_counts[j] / total_N for j in range(n_actions))

    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return float(V)


# =========================================================
# Worker
# =========================================================

def run_single_estimate(args):
    samples, estimator, seed = args
    seed_rng(seed % (2**31))
    board = make_initial_board()
    value = estimate_root(board, samples, estimator, DEPTH)
    return {"value": value}


# =========================================================
# Output formatting
# =========================================================

def format_value_tables(summary_results, sample_sizes, estimators):
    lookup = {(r["estimator"], r["N"]): r for r in summary_results}
    col_w = max(14, max(len(e) for e in estimators) + 2)
    lines = []

    def hdr():
        lines.append(f"{'N':>8s}" + "".join(f"{e:>{col_w}s}" for e in estimators))

    def sep():
        lines.append("-" * (8 + col_w * len(estimators)))

    lines.append("=" * (8 + col_w * len(estimators)))
    lines.append("VALUE SUMMARY")
    lines.append("=" * (8 + col_w * len(estimators)))

    lines.append("")
    lines.append("[1] ESTIMATED VALUE")
    hdr()
    sep()
    for n in sample_sizes:
        row = f"{n:8d}"
        for e in estimators:
            row += f"{lookup[(e, n)]['mean_V']:>{col_w}.6f}"
        lines.append(row)

    lines.append("")
    lines.append("[2] BIAS")
    hdr()
    sep()
    for n in sample_sizes:
        row = f"{n:8d}"
        for e in estimators:
            row += f"{lookup[(e, n)]['bias']:>{col_w}.6f}"
        lines.append(row)

    lines.append("")
    lines.append("[3] STDEV")
    hdr()
    sep()
    for n in sample_sizes:
        row = f"{n:8d}"
        for e in estimators:
            row += f"{lookup[(e, n)]['stdev']:>{col_w}.6f}"
        lines.append(row)

    lines.append("")
    lines.append("[4] MSE")
    hdr()
    sep()
    for n in sample_sizes:
        row = f"{n:8d}"
        for e in estimators:
            row += f"{lookup[(e, n)]['mse']:>{col_w}.6f}"
        lines.append(row)

    return "\n".join(lines)


# =========================================================
# Batch experiment
# =========================================================

def test_value_estimators(sample_sizes, estimators, true_value, rounds, max_workers=None):
    print(f"\nTrue value V* = {true_value:.8f}", flush=True)

    rng = np.random.RandomState(42)
    seeds_by_n = {
        n: rng.randint(0, 2**31 - 1, size=rounds).astype(np.int64)
        for n in sample_sizes
    }
    summary_results = []

    for est in estimators:
        for n in sample_sizes:
            tasks = [(n, est, int(seed)) for seed in seeds_by_n[n]]

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as ex:
                outputs = list(tqdm(
                    ex.map(run_single_estimate, tasks),
                    total=rounds,
                    desc=f"{est}@{n}",
                ))

            values = np.array([x["value"] for x in outputs], dtype=float)

            mean_V = float(values.mean())
            bias = float(mean_V - true_value)
            stdev = float(values.std(ddof=1)) if rounds > 1 else 0.0
            mse = float(((values - true_value) ** 2).mean())

            print(
                f"N={n:3d} | est={est:8s} | "
                f"mean={mean_V:.6f} | bias={bias:+.6f} | "
                f"stdev={stdev:.6f} | mse={mse:.8f}",
                flush=True,
            )

            summary_results.append({
                "estimator": est,
                "N": n,
                "mean_V": mean_V,
                "bias": bias,
                "stdev": stdev,
                "mse": mse,
                "true_value": true_value,
                "value_replications": rounds,
                "depth": DEPTH,
            })

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=summary_results[0].keys())
        writer.writeheader()
        writer.writerows(summary_results)
    print(f"\nSummary saved to {SUMMARY_FILE}", flush=True)

    table_txt = format_value_tables(summary_results, sample_sizes, estimators)
    print("\n" + table_txt, flush=True)

    return summary_results


# =========================================================
# Main experiment driver
# =========================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print("3x3 Tic-Tac-Toe — value-only AMS experiments", flush=True)
    print("=" * 100, flush=True)
    print(f"DEPTH={DEPTH}, SAMPLE_SIZES={SAMPLE_SIZES}, VALUE_REPLICATIONS={VALUE_REPLICATIONS}", flush=True)
    print(f"Estimators: {VALUE_ESTIMATORS}", flush=True)
    print(f"Output file: {SUMMARY_FILE}", flush=True)

    print("\nWarming up Numba...", end=" ", flush=True)
    t0 = time.time()

    warmup_board = make_initial_board()
    seed_rng(0)
    for code in (0, 1, 2):
        estimate_nb(warmup_board.copy(), 10, code, 2)

    warmup_cache = Dict.empty(key_type=types.int64, value_type=types.float64)
    exact_q_values_nb(warmup_board.copy(), warmup_cache)

    print(f"done ({time.time() - t0:.1f}s)", flush=True)

    print("\nComputing exact true value via expectimax...", flush=True)
    initial_board = make_initial_board()
    cache = Dict.empty(key_type=types.int64, value_type=types.float64)
    opt_action, opt_q, all_actions, all_q = exact_q_values_nb(initial_board, cache)
    true_value = float(opt_q)

    print(f"True value V*(s0) = {true_value:.8f}", flush=True)
    print(f"Optimal action = {int(opt_action)} (Q* = {float(opt_q):.6f})", flush=True)
    print("All Q*(s,a):", flush=True)
    for i in range(len(all_actions)):
        marker = " <-- optimal" if all_actions[i] == opt_action else ""
        print(f"  action {int(all_actions[i]):2d}: Q* = {float(all_q[i]):.6f}{marker}", flush=True)

    return test_value_estimators(
        sample_sizes=SAMPLE_SIZES,
        estimators=VALUE_ESTIMATORS,
        true_value=true_value,
        rounds=VALUE_REPLICATIONS,
        max_workers=os.cpu_count(),
    )


if __name__ == "__main__":
    main()
