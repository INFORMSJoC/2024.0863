"""
4x4 Tic-Tac-Toe — value-estimation AMS experiments.

This script reproduces the 4x4 Tic-Tac-Toe value-estimation experiment with
a uniform-random Player 1 policy.

The script first computes the exact benchmark value V*(s0) by expectimax,
where Player 1 follows a uniform random policy and Player 2 optimizes its
moves. It then runs AMS value-estimation experiments using three estimators:
    - lsa: largest-size average estimator;
    - max: maximum estimator;
    - weighted: weighted-average estimator.

Board representation:
    - board is a NumPy int8 array of length 16 in row-major order;
    - 0 = empty, 1 = Player 1 (X), 2 = Player 2 (O).

Initial board:
    .  .  O  .
    .  O  X  .
    .  X  .  .
    X  .  .  .

    (Player 2 moves next.)

Main settings:
    - board size: 4x4;
    - Player 1 policy: uniform random over feasible moves;
    - Player 2: AMS planning;
    - recursion depth DEPTH = 6;
    - sample sizes N = 20, 30, 40, 50, 60;
    - 500 independent replications for each estimator and sample size.

Output:
    results/ttt4x4_random_value_summary.csv

Run:
    python scripts/ttt4x4_value.py
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

BOARD_SIZE = 16
DEPTH = 6

SAMPLE_SIZES = [20, 30, 40, 50, 60]
VALUE_REPLICATIONS = 500

VALUE_ESTIMATORS = ["lsa", "max", "weighted"]
EST_CODES = {
    "max": 0,
    "lsa": 1,
    "weighted": 2,
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "value_estimation"
SUMMARY_FILE = OUTPUT_DIR / "ttt4x4.csv"

WIN_LINES = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
    [12, 13, 14, 15],
    [0, 4, 8, 12],
    [1, 5, 9, 13],
    [2, 6, 10, 14],
    [3, 7, 11, 15],
    [0, 5, 10, 15],
    [3, 6, 9, 12],
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
    for k in range(10):
        a = WIN_LINES[k, 0]
        b = WIN_LINES[k, 1]
        c = WIN_LINES[k, 2]
        d = WIN_LINES[k, 3]
        v = board[a]
        if v != 0 and v == board[b] and v == board[c] and v == board[d]:
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
    p1_action = p1_move_nb(board, buf)

    board[p1_action] = 1
    res2 = check_winner_nb(board)

    if res2 == 1:
        board[p1_action] = 0
        board[action] = 0
        return 0.0, -1
    if res2 == 0:
        board[p1_action] = 0
        board[action] = 0
        return 0.5, -1

    return 0.0, p1_action


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
        # DEPTH is sufficient to cover the remaining game from the specified
        # initial board, so this branch should not be reached in the
        # reported experiment.
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
        action = act_buf[0]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        return reward

    if samples < n_actions:
        samples = n_actions

    Q = np.zeros(n_actions, dtype=np.float64)
    Na = np.zeros(n_actions, dtype=np.int32)
    total_N = 0

    # Phase 1: sample each feasible P2 action once.
    for i in range(n_actions):
        action = act_buf[i]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
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

        action = act_buf[best_idx]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
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

    elif estimator_code == 1:  # LSA; largest count wins, ties go to the first action.
        max_n = 0
        best_idx = 0
        for i in range(n_actions):
            # Strict ">" keeps the earliest action when counts are tied.
            if Na[i] > max_n:
                max_n = Na[i]
                best_idx = i

        return Q[best_idx] / float(Na[best_idx])

    else:  # weighted / WA
        value = 0.0
        for i in range(n_actions):
            value += Q[i]
        return value / float(total_N)


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
            value = 1.0
        elif res == 0:
            value = 0.5
        else:
            value = 0.0
        cache[key] = value
        return value

    p2_buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n_p2 = get_actions_nb(board, p2_buf)

    best_value = -1.0
    for i in range(n_p2):
        action = p2_buf[i]
        board[action] = 2

        res2 = check_winner_nb(board)
        if res2 == 2:
            value = 1.0
        elif res2 == 0:
            value = 0.5
        else:
            p1_buf = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buf, p1_probs)

            value = 0.0
            for j in range(n_p1):
                p1_action = p1_buf[j]
                board[p1_action] = 1

                res3 = check_winner_nb(board)
                if res3 == 1:
                    child_value = 0.0
                elif res3 == 0:
                    child_value = 0.5
                else:
                    child_value = exact_expectimax_nb(board, cache)

                value += p1_probs[j] * child_value
                board[p1_action] = 0

        board[action] = 0
        if value > best_value:
            best_value = value

    cache[key] = best_value
    return best_value


@njit(cache=True)
def exact_q_values_nb(board, cache):
    buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n = get_actions_nb(board, buf)
    actions = np.empty(n, dtype=np.int32)
    q_values = np.empty(n, dtype=np.float64)

    best_action = -1
    best_q = -1.0

    for i in range(n):
        action = buf[i]
        actions[i] = action
        board[action] = 2

        res = check_winner_nb(board)
        if res == 2:
            q_values[i] = 1.0
        elif res == 0:
            q_values[i] = 0.5
        else:
            p1_buf = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buf, p1_probs)

            expected_value = 0.0
            for j in range(n_p1):
                p1_action = p1_buf[j]
                board[p1_action] = 1

                res2 = check_winner_nb(board)
                if res2 == 1:
                    child_value = 0.0
                elif res2 == 0:
                    child_value = 0.5
                else:
                    child_value = exact_expectimax_nb(board, cache)

                expected_value += p1_probs[j] * child_value
                board[p1_action] = 0

            q_values[i] = expected_value

        board[action] = 0
        if q_values[i] > best_q:
            best_q = q_values[i]
            best_action = action

    return best_action, best_q, actions, q_values


# =========================================================
# Python helpers
# =========================================================

def make_initial_board():
    """
    Construct the initial board:

        .  .  O  .
        .  O  X  .
        .  X  .  .
        X  .  .  .

    Player 2 moves next.
    """
    board = np.zeros(BOARD_SIZE, dtype=np.int8)

    board[2] = 2
    board[5] = 2

    board[6] = 1
    board[9] = 1
    board[12] = 1

    return board


# =========================================================
# Root-level AMS estimation
# =========================================================

def estimate_root(board_np, samples, estimator, depth=DEPTH):
    est_code = EST_CODES[estimator]

    res = check_winner_nb(board_np)
    if res != -1:
        return 1.0 if res == 2 else (0.5 if res == 0 else 0.0)

    act_buf = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = int(get_actions_nb(board_np, act_buf))
    actions = [int(act_buf[i]) for i in range(n_actions)]

    if n_actions == 1:
        action = actions[0]
        reward, p1_action = do_move_nb(board_np, action)
        if p1_action >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_action] = 0
            board_np[action] = 0
        return float(reward)

    if samples < n_actions:
        samples = n_actions

    Q = np.zeros(n_actions, dtype=np.float64)
    Na = np.zeros(n_actions, dtype=np.int32)
    total_N = 0

    # Phase 1: sample each feasible P2 action once.
    for i in range(n_actions):
        action = actions[i]
        reward, p1_action = do_move_nb(board_np, action)
        if p1_action >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_action] = 0
            board_np[action] = 0
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

        action = actions[best_idx]
        reward, p1_action = do_move_nb(board_np, action)
        if p1_action >= 0:
            reward = float(estimate_nb(board_np, samples, est_code, depth - 1))
            board_np[p1_action] = 0
            board_np[action] = 0
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
        value = Q_avgs[selected_idx]

    elif estimator == "lsa":
        # Pure counts rule: take the action with the largest sample count and
        # break ties by the first available action. No Q-value comparison.
        max_N = max(N_counts)
        selected_idx = next(j for j, count in enumerate(N_counts) if count == max_N)
        value = Q_avgs[selected_idx]

    elif estimator == "weighted":
        value = sum(Q_avgs[j] * N_counts[j] / total_N for j in range(n_actions))

    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    return float(value)


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
    print("4x4 Tic-Tac-Toe — value-only AMS experiments", flush=True)
    print("=" * 100, flush=True)
    print(f"DEPTH={DEPTH}, SAMPLE_SIZES={SAMPLE_SIZES}, VALUE_REPLICATIONS={VALUE_REPLICATIONS}", flush=True)
    print(f"Estimators: {VALUE_ESTIMATORS}", flush=True)
    print(f"Output file: {SUMMARY_FILE}", flush=True)

    print("\nWarming up Numba...", end=" ", flush=True)
    t0 = time.time()

    warmup_board = make_initial_board()
    seed_rng(0)
    for code in (0, 1, 2):
        estimate_nb(warmup_board.copy(), 20, code, 2)

    # Warm up the expectimax functions on a smaller late-game state.
    small_board = make_initial_board()
    small_board[0] = 1
    small_board[1] = 2
    small_board[4] = 2
    small_board[7] = 1
    small_board[8] = 2

    warmup_cache = Dict.empty(key_type=types.int64, value_type=types.float64)
    exact_q_values_nb(small_board, warmup_cache)

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
