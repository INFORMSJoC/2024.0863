# -*- coding: utf-8 -*-
"""
4x4 Tic-Tac-Toe with uniform-random Player 1 — policy-performance AMS.

For each sample size, one AMS tree is constructed for each underlying tree
estimator (LSA, maximum, and weighted). The LSA and LSA-maxQ policy models
share the same LSA tree and differ only in their policy readout rule.

When a rollout reaches a Player-2-to-move state that has no accumulated AMS
action statistics, that state is treated as a new root and an on-the-fly AMS
search is run using the corresponding estimator and readout rule. The selected
action is cached, so each previously uncovered state is expanded at most once
within a policy evaluation.

Reported metrics:
    - overall agreement rate with the exact optimal action set;
    - conditional agreement rate among visited states;
    - visited ratio;
    - average cumulative reward and rollout Monte Carlo standard error;
    - exact true value and optimal action.

With POLICY_REPLICATIONS = 1, policy_reward_se is conditional on the single
initial AMS tree and the cached on-the-fly policy expansions.

Output:
    results/policy_performance/ttt4x4_random.csv

Run:
    python scripts/policy_performance/ttt4x4_random.py
"""

import csv
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
from numba import njit, types
from numba.typed import Dict


# =========================================================
# Configuration
# =========================================================

BOARD_SIZE = 16
DEPTH = 6

POLICY_REPLICATIONS = 1
POLICY_ROLLOUTS = 10000

SAMPLE_SIZES = [20,30,40,50,60]
POLICY_MODELS = ["lsa", "max", "weighted", "lsa_maxq"]

EST_CODES = {"max": 0, "lsa": 1, "weighted": 2}
READOUT_CODES = {"lsa": 0, "max": 1, "weighted": 2, "lsa_maxq": 3}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "policy_performance"
SUMMARY_FILE = OUTPUT_DIR / "ttt4x4_random.csv"

WIN_LINES = np.array([
    [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15],
    [0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15],
    [0, 5, 10, 15], [3, 6, 9, 12],
], dtype=np.int32)


# =========================================================
# Numba core
# =========================================================

@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def check_winner_nb(board):
    for k in range(10):
        a, b, c, d = WIN_LINES[k]
        value = board[a]
        if value != 0 and value == board[b] and value == board[c] and value == board[d]:
            return int(value)
    for i in range(BOARD_SIZE):
        if board[i] == 0:
            return -1
    return 0


@njit(cache=True)
def get_actions_nb(board, action_buffer):
    """Return feasible actions in increasing board-position order."""
    n_actions = 0
    for position in range(BOARD_SIZE):
        if board[position] == 0:
            action_buffer[n_actions] = position
            n_actions += 1
    return n_actions


@njit(cache=True)
def p1_move_nb(board, action_buffer):
    n_actions = get_actions_nb(board, action_buffer)
    return action_buffer[np.random.randint(0, n_actions)]


@njit(cache=True)
def p1_move_probs_nb(board, action_buffer, probability_buffer):
    n_actions = get_actions_nb(board, action_buffer)
    probability = 1.0 / float(n_actions)
    for i in range(n_actions):
        probability_buffer[i] = probability
    return n_actions


@njit(cache=True)
def do_move_nb(board, action):
    """Apply P2 action and, if ongoing, one uniform-random P1 response."""
    board[action] = 2
    result = check_winner_nb(board)
    if result == 2:
        board[action] = 0
        return 1.0, -1
    if result == 0:
        board[action] = 0
        return 0.5, -1

    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    p1_action = p1_move_nb(board, action_buffer)
    board[p1_action] = 1
    result_after_p1 = check_winner_nb(board)

    if result_after_p1 == 1:
        board[p1_action] = 0
        board[action] = 0
        return 0.0, -1
    if result_after_p1 == 0:
        board[p1_action] = 0
        board[action] = 0
        return 0.5, -1

    return 0.0, p1_action


@njit(cache=True)
def lsa_value_nb(q_sums, sample_counts, n_actions):
    """LSA tree value: largest count wins; count ties use the first action."""
    largest_count = 0
    selected_index = 0
    for i in range(n_actions):
        if sample_counts[i] > largest_count:
            largest_count = sample_counts[i]
            selected_index = i
    return q_sums[selected_index] / float(sample_counts[selected_index])


@njit(cache=True)
def estimate_nb(board, samples, estimator_code, depth):
    result = check_winner_nb(board)
    if result != -1:
        if result == 2:
            return 1.0
        if result == 0:
            return 0.5
        return 0.0
    if depth <= 0:
        return 0.5

    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = get_actions_nb(board, action_buffer)

    if n_actions == 1:
        action = action_buffer[0]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        return reward

    if samples < n_actions:
        samples = n_actions

    q_sums = np.zeros(n_actions, dtype=np.float64)
    sample_counts = np.zeros(n_actions, dtype=np.int32)
    total_samples = 0

    for i in range(n_actions):
        action = action_buffer[i]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        q_sums[i] = reward
        sample_counts[i] = 1
        total_samples += 1

    while total_samples < samples:
        best_ucb = -1.0e18
        selected_index = 0
        for i in range(n_actions):
            q_average = q_sums[i] / float(sample_counts[i])
            ucb = q_average + math.sqrt(
                2.0 * math.log(float(total_samples)) / float(sample_counts[i])
            )
            if ucb > best_ucb:
                best_ucb = ucb
                selected_index = i

        action = action_buffer[selected_index]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        q_sums[selected_index] += reward
        sample_counts[selected_index] += 1
        total_samples += 1

    if estimator_code == 0:
        best_q = -1.0e18
        for i in range(n_actions):
            q_average = q_sums[i] / float(sample_counts[i])
            if q_average > best_q:
                best_q = q_average
        return best_q
    if estimator_code == 1:
        return lsa_value_nb(q_sums, sample_counts, n_actions)

    total_value = 0.0
    for i in range(n_actions):
        total_value += q_sums[i]
    return total_value / float(total_samples)


@njit(cache=True)
def board_to_key(board):
    key = np.int64(0)
    for i in range(BOARD_SIZE):
        key = key * 3 + np.int64(board[i])
    return key


@njit(cache=True)
def estimate_nb_acc(
    board, samples, estimator_code, depth,
    accumulated_counts, accumulated_q_sums, key_to_index,
):
    result = check_winner_nb(board)
    if result != -1:
        if result == 2:
            return 1.0
        if result == 0:
            return 0.5
        return 0.0
    if depth <= 0:
        return 0.5

    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = get_actions_nb(board, action_buffer)
    state_index = key_to_index[board_to_key(board)]

    if n_actions == 1:
        action = action_buffer[0]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb_acc(
                board, samples, estimator_code, depth - 1,
                accumulated_counts, accumulated_q_sums, key_to_index,
            )
            board[p1_action] = 0
            board[action] = 0
        accumulated_counts[state_index, action] += np.int64(1)
        accumulated_q_sums[state_index, action] += reward
        return reward

    if samples < n_actions:
        samples = n_actions

    q_sums = np.zeros(n_actions, dtype=np.float64)
    sample_counts = np.zeros(n_actions, dtype=np.int32)
    total_samples = 0

    for i in range(n_actions):
        action = action_buffer[i]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb_acc(
                board, samples, estimator_code, depth - 1,
                accumulated_counts, accumulated_q_sums, key_to_index,
            )
            board[p1_action] = 0
            board[action] = 0
        q_sums[i] = reward
        sample_counts[i] = 1
        total_samples += 1

    while total_samples < samples:
        best_ucb = -1.0e18
        selected_index = 0
        for i in range(n_actions):
            q_average = q_sums[i] / float(sample_counts[i])
            ucb = q_average + math.sqrt(
                2.0 * math.log(float(total_samples)) / float(sample_counts[i])
            )
            if ucb > best_ucb:
                best_ucb = ucb
                selected_index = i

        action = action_buffer[selected_index]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb_acc(
                board, samples, estimator_code, depth - 1,
                accumulated_counts, accumulated_q_sums, key_to_index,
            )
            board[p1_action] = 0
            board[action] = 0
        q_sums[selected_index] += reward
        sample_counts[selected_index] += 1
        total_samples += 1

    for i in range(n_actions):
        action = action_buffer[i]
        accumulated_counts[state_index, action] += sample_counts[i]
        accumulated_q_sums[state_index, action] += q_sums[i]

    if estimator_code == 0:
        best_q = -1.0e18
        for i in range(n_actions):
            q_average = q_sums[i] / float(sample_counts[i])
            if q_average > best_q:
                best_q = q_average
        return best_q
    if estimator_code == 1:
        return lsa_value_nb(q_sums, sample_counts, n_actions)

    total_value = 0.0
    for i in range(n_actions):
        total_value += q_sums[i]
    return total_value / float(total_samples)


@njit(cache=True)
def estimated_action_from_acc_nb(
    state_index, readout_code, accumulated_counts, accumulated_q_sums,
    action_buffer, n_actions,
):
    has_visited_action = False
    for i in range(n_actions):
        if accumulated_counts[state_index, action_buffer[i]] > 0:
            has_visited_action = True
            break
    if not has_visited_action:
        return -1

    if readout_code == 1 or readout_code == 2 or readout_code == 3:
        best_action = -1
        best_q = -1.0e18
        for i in range(n_actions):
            action = action_buffer[i]
            count = accumulated_counts[state_index, action]
            if count <= 0:
                continue
            q_hat = accumulated_q_sums[state_index, action] / float(count)
            if q_hat > best_q:
                best_q = q_hat
                best_action = action
        return best_action

    largest_count = 0
    for i in range(n_actions):
        count = accumulated_counts[state_index, action_buffer[i]]
        if count > largest_count:
            largest_count = count

    best_action = -1
    best_q = -1.0e18
    for i in range(n_actions):
        action = action_buffer[i]
        count = accumulated_counts[state_index, action]
        if count == largest_count:
            q_hat = accumulated_q_sums[state_index, action] / float(count)
            if q_hat > best_q:
                best_q = q_hat
                best_action = action
    return best_action


@njit(cache=True)
def on_the_fly_action_nb(board, samples, estimator_code, readout_code, depth):
    """Run AMS with board as root and return the policy-model action."""
    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = get_actions_nb(board, action_buffer)
    if n_actions <= 0:
        return -1
    if n_actions == 1:
        return int(action_buffer[0])
    if samples < n_actions:
        samples = n_actions

    q_sums = np.zeros(n_actions, dtype=np.float64)
    sample_counts = np.zeros(n_actions, dtype=np.int32)
    total_samples = 0

    for i in range(n_actions):
        action = action_buffer[i]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        q_sums[i] = reward
        sample_counts[i] = 1
        total_samples += 1

    while total_samples < samples:
        best_ucb = -1.0e18
        selected_index = 0
        for i in range(n_actions):
            q_average = q_sums[i] / float(sample_counts[i])
            ucb = q_average + math.sqrt(
                2.0 * math.log(float(total_samples)) / float(sample_counts[i])
            )
            if ucb > best_ucb:
                best_ucb = ucb
                selected_index = i

        action = action_buffer[selected_index]
        reward, p1_action = do_move_nb(board, action)
        if p1_action >= 0:
            reward = estimate_nb(board, samples, estimator_code, depth - 1)
            board[p1_action] = 0
            board[action] = 0
        q_sums[selected_index] += reward
        sample_counts[selected_index] += 1
        total_samples += 1

    if readout_code == 0:
        largest_count = 0
        for i in range(n_actions):
            if sample_counts[i] > largest_count:
                largest_count = sample_counts[i]
        selected_index = 0
        best_q = -1.0e18
        for i in range(n_actions):
            if sample_counts[i] == largest_count:
                q_average = q_sums[i] / float(sample_counts[i])
                if q_average > best_q:
                    best_q = q_average
                    selected_index = i
        return int(action_buffer[selected_index])

    selected_index = 0
    best_q = q_sums[0] / float(sample_counts[0])
    for i in range(1, n_actions):
        q_average = q_sums[i] / float(sample_counts[i])
        if q_average > best_q:
            best_q = q_average
            selected_index = i
    return int(action_buffer[selected_index])


# =========================================================
# Exact benchmark
# =========================================================

@njit(cache=True)
def exact_expectimax_nb(board, cache):
    key = board_to_key(board)
    if key in cache:
        return cache[key]

    result = check_winner_nb(board)
    if result != -1:
        if result == 2:
            value = 1.0
        elif result == 0:
            value = 0.5
        else:
            value = 0.0
        cache[key] = value
        return value

    p2_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    n_p2 = get_actions_nb(board, p2_buffer)
    best_value = -1.0

    for i in range(n_p2):
        action = p2_buffer[i]
        board[action] = 2
        result_after_p2 = check_winner_nb(board)

        if result_after_p2 == 2:
            value = 1.0
        elif result_after_p2 == 0:
            value = 0.5
        else:
            p1_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buffer, p1_probs)
            value = 0.0
            for j in range(n_p1):
                p1_action = p1_buffer[j]
                board[p1_action] = 1
                result_after_p1 = check_winner_nb(board)
                if result_after_p1 == 1:
                    child_value = 0.0
                elif result_after_p1 == 0:
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
    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    n_actions = get_actions_nb(board, action_buffer)
    actions = np.empty(n_actions, dtype=np.int32)
    q_values = np.empty(n_actions, dtype=np.float64)
    best_action = -1
    best_q = -1.0

    for i in range(n_actions):
        action = action_buffer[i]
        actions[i] = action
        board[action] = 2
        result = check_winner_nb(board)

        if result == 2:
            q_values[i] = 1.0
        elif result == 0:
            q_values[i] = 0.5
        else:
            p1_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
            p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)
            n_p1 = p1_move_probs_nb(board, p1_buffer, p1_probs)
            expected_value = 0.0
            for j in range(n_p1):
                p1_action = p1_buffer[j]
                board[p1_action] = 1
                result_after_p1 = check_winner_nb(board)
                if result_after_p1 == 1:
                    child_value = 0.0
                elif result_after_p1 == 0:
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


def board_to_key_py(board):
    key = 0
    for i in range(BOARD_SIZE):
        key = key * 3 + int(board[i])
    return key


def enumerate_reachable_p2_boards():
    start = make_initial_board()
    queue = deque([start.copy()])
    seen = {board_to_key_py(start)}
    reachable = []
    action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    p1_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
    p1_probs = np.empty(BOARD_SIZE, dtype=np.float64)

    while queue:
        board = queue.popleft()
        if int(check_winner_nb(board)) != -1:
            continue
        reachable.append(board.copy())

        n_actions = int(get_actions_nb(board, action_buffer))
        for i in range(n_actions):
            action = int(action_buffer[i])
            after_p2 = board.copy()
            after_p2[action] = 2
            result_after_p2 = int(check_winner_nb(after_p2))
            if result_after_p2 == 2 or result_after_p2 == 0:
                continue

            n_p1 = int(p1_move_probs_nb(after_p2, p1_buffer, p1_probs))
            for j in range(n_p1):
                p1_action = int(p1_buffer[j])
                next_board = after_p2.copy()
                next_board[p1_action] = 1
                result_after_p1 = int(check_winner_nb(next_board))
                if result_after_p1 == 1 or result_after_p1 == 0:
                    continue
                key = board_to_key_py(next_board)
                if key not in seen:
                    seen.add(key)
                    queue.append(next_board)

    return reachable


def build_key_to_index(reachable):
    return {board_to_key_py(board): i for i, board in enumerate(reachable)}


def run_full_ams_acc(samples, tree_estimator, seed, key_to_index, n_states):
    seed_rng(int(seed) % (2**31))
    numba_map = Dict.empty(key_type=types.int64, value_type=types.int64)
    for key, index in key_to_index.items():
        numba_map[np.int64(key)] = np.int64(index)

    accumulated_counts = np.zeros((n_states, BOARD_SIZE), dtype=np.int64)
    accumulated_q_sums = np.zeros((n_states, BOARD_SIZE), dtype=np.float64)
    estimate_nb_acc(
        make_initial_board(), samples, EST_CODES[tree_estimator], DEPTH,
        accumulated_counts, accumulated_q_sums, numba_map,
    )
    return accumulated_counts, accumulated_q_sums


def agreement_and_visited_rates(
    accumulated_counts, accumulated_q_sums, key_to_index,
    reachable, readout_name, exact_cache,
):
    readout_code = READOUT_CODES[readout_name]
    n_agree = 0
    n_visited = 0
    tolerance = 1.0e-9

    for board in reachable:
        state_index = key_to_index[board_to_key_py(board)]
        action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
        n_actions = int(get_actions_nb(board, action_buffer))
        visited = any(
            accumulated_counts[state_index, int(action_buffer[i])] > 0
            for i in range(n_actions)
        )
        if visited:
            n_visited += 1

        estimated_action = int(estimated_action_from_acc_nb(
            state_index, readout_code, accumulated_counts, accumulated_q_sums,
            action_buffer, n_actions,
        ))
        _, _, exact_actions, exact_q_values = exact_q_values_nb(board.copy(), exact_cache)
        maximum_q = float(np.max(exact_q_values))
        threshold = maximum_q - tolerance * max(1.0, abs(maximum_q))
        optimal_actions = {
            int(exact_actions[i]) for i in range(len(exact_actions))
            if float(exact_q_values[i]) >= threshold
        }
        if visited and estimated_action >= 0 and estimated_action in optimal_actions:
            n_agree += 1

    n_reachable = len(reachable)
    return (
        n_agree / n_reachable if n_reachable else 0.0,
        n_visited / n_reachable if n_reachable else 0.0,
        n_agree / n_visited if n_visited else 0.0,
    )


def policy_rollout_mean_se(
    accumulated_counts, accumulated_q_sums, key_to_index,
    policy_model, samples, seed_base, n_rollouts,
):
    rewards = np.empty(n_rollouts, dtype=np.float64)
    readout_code = READOUT_CODES[policy_model]
    tree_estimator = "lsa" if policy_model == "lsa_maxq" else policy_model
    estimator_code = EST_CODES[tree_estimator]

    on_the_fly_action_cache = {}
    total_on_the_fly_encounters = 0
    unique_on_the_fly_states = 0
    rollouts_with_on_the_fly = 0

    for rollout_index in range(n_rollouts):
        seed_rng((int(seed_base) + 1009 * rollout_index) % (2**31))
        board = make_initial_board()
        used_on_the_fly = False

        while True:
            result = int(check_winner_nb(board))
            if result != -1:
                rewards[rollout_index] = 1.0 if result == 2 else (0.5 if result == 0 else 0.0)
                break

            action_buffer = np.empty(BOARD_SIZE, dtype=np.int32)
            n_actions = int(get_actions_nb(board, action_buffer))
            state_key = board_to_key_py(board)
            state_index = key_to_index.get(state_key, -1)

            if state_index >= 0:
                estimated_action = int(estimated_action_from_acc_nb(
                    state_index, readout_code, accumulated_counts,
                    accumulated_q_sums, action_buffer, n_actions,
                ))
            else:
                estimated_action = -1

            if estimated_action < 0:
                used_on_the_fly = True
                total_on_the_fly_encounters += 1
                if state_key not in on_the_fly_action_cache:
                    unique_on_the_fly_states += 1
                    on_the_fly_seed = (
                        int(seed_base) + 1_000_003 + 10_007 * state_key
                    ) % (2**31)
                    seed_rng(on_the_fly_seed)
                    estimated_action = int(on_the_fly_action_nb(
                        board.copy(), samples, estimator_code, readout_code, DEPTH,
                    ))
                    on_the_fly_action_cache[state_key] = estimated_action
                else:
                    estimated_action = on_the_fly_action_cache[state_key]

            reward, p1_action = do_move_nb(board, estimated_action)
            if p1_action < 0:
                rewards[rollout_index] = float(reward)
                break

        if used_on_the_fly:
            rollouts_with_on_the_fly += 1

    mean_reward = float(np.mean(rewards))
    se_reward = float(np.std(rewards, ddof=1) / np.sqrt(n_rollouts)) if n_rollouts > 1 else 0.0
    return (
        mean_reward, se_reward, total_on_the_fly_encounters,
        unique_on_the_fly_states, rollouts_with_on_the_fly,
    )


def format_policy_tables(results):
    lines = ["POLICY SUMMARY", "=" * 110]
    for row in results:
        lines.append(
            f"N={row['N']:3d} | model={row['policy_model']:9s} | "
            f"agreement={row['agreement_rate']:.6f} | "
            f"conditional={row['conditional_agreement_rate']:.6f} | "
            f"visited={row['visited_ratio']:.6f} | "
            f"reward={row['policy_reward_mean']:.6f} ± {row['policy_reward_se']:.6f} | "
            f"true_V={row['true_value']:.6f} | "
            f"opt_a={row['optimal_action']}"
        )
    return "\n".join(lines) + "\n"


# =========================================================
# Experiment
# =========================================================

def test_policy_models(sample_sizes, policy_models):
    print("Building reachable states and exact benchmark...", flush=True)
    reachable = enumerate_reachable_p2_boards()
    key_to_index = build_key_to_index(reachable)
    n_states = len(reachable)
    exact_cache = Dict.empty(key_type=types.int64, value_type=types.float64)

    optimal_action, optimal_q, _, _ = exact_q_values_nb(make_initial_board(), exact_cache)
    true_value = float(optimal_q)
    print(f"|S_reach|={n_states}, V*(s0)={true_value:.8f}, optimal_action={int(optimal_action)}", flush=True)

    rng = np.random.RandomState(42)
    summary_rows = []
    raw_rows = []

    for sample_size in sample_sizes:
        replication_data = []
        for replication in range(POLICY_REPLICATIONS):
            common_tree_seed = int(rng.randint(0, 2**31 - 1))
            tree_seeds = {
                "lsa": common_tree_seed,
                "max": common_tree_seed,
                "weighted": common_tree_seed,
            }
            trees = {}
            for estimator in ("lsa", "max", "weighted"):
                trees[estimator] = run_full_ams_acc(
                    sample_size, estimator, tree_seeds[estimator], key_to_index, n_states,
                )
            replication_data.append((replication, tree_seeds, trees))

        for policy_model in policy_models:
            tree_estimator = "lsa" if policy_model == "lsa_maxq" else policy_model
            metrics = []

            for replication, tree_seeds, trees in replication_data:
                accumulated_counts, accumulated_q_sums = trees[tree_estimator]
                agreement, visited, conditional = agreement_and_visited_rates(
                    accumulated_counts, accumulated_q_sums, key_to_index,
                    reachable, policy_model, exact_cache,
                )
                tree_seed = tree_seeds[tree_estimator]
                rollout_seed_base = (tree_seed + 333_333) % (2**31)
                reward_mean, reward_se, encounters, unique_states, rollout_count = policy_rollout_mean_se(
                    accumulated_counts, accumulated_q_sums, key_to_index,
                    policy_model, sample_size, rollout_seed_base, POLICY_ROLLOUTS,
                )
                row = {
                    "policy_model": policy_model,
                    "tree_estimator": tree_estimator,
                    "N": sample_size,
                    "replication": replication,
                    "tree_seed": tree_seed,
                    "rollout_seed_base": rollout_seed_base,
                    "agreement_rate": agreement,
                    "conditional_agreement_rate": conditional,
                    "visited_ratio": visited,
                    "policy_reward_mean": reward_mean,
                    "policy_reward_se": reward_se,
                    "policy_rollouts": POLICY_ROLLOUTS,
                    "total_on_the_fly_encounters": encounters,
                    "unique_on_the_fly_states": unique_states,
                    "rollouts_with_on_the_fly": rollout_count,
                    "rollouts_with_on_the_fly_rate": rollout_count / POLICY_ROLLOUTS,
                    "true_value": true_value,
                    "optimal_action": int(optimal_action),
                }
                raw_rows.append(row)
                metrics.append(row)

            reward_means = [x["policy_reward_mean"] for x in metrics]
            summary = {
                "policy_model": policy_model,
                "tree_estimator": tree_estimator,
                "N": sample_size,
                "agreement_rate": float(np.mean([x["agreement_rate"] for x in metrics])),
                "conditional_agreement_rate": float(np.mean([x["conditional_agreement_rate"] for x in metrics])),
                "visited_ratio": float(np.mean([x["visited_ratio"] for x in metrics])),
                "policy_reward_mean": float(np.mean(reward_means)),
                "policy_reward_se": (
                    float(metrics[0]["policy_reward_se"])
                    if POLICY_REPLICATIONS == 1
                    else float(np.std(reward_means, ddof=1) / np.sqrt(POLICY_REPLICATIONS))
                ),
                "policy_rollouts": POLICY_ROLLOUTS,
                "policy_reps": POLICY_REPLICATIONS,
                "total_on_the_fly_encounters": int(round(np.mean([x["total_on_the_fly_encounters"] for x in metrics]))),
                "unique_on_the_fly_states": int(round(np.mean([x["unique_on_the_fly_states"] for x in metrics]))),
                "rollouts_with_on_the_fly": int(round(np.mean([x["rollouts_with_on_the_fly"] for x in metrics]))),
                "rollouts_with_on_the_fly_rate": float(np.mean([x["rollouts_with_on_the_fly_rate"] for x in metrics])),
                "true_value": true_value,
                "optimal_action": int(optimal_action),
            }
            summary_rows.append(summary)
            print(
                f"N={sample_size:3d} | model={policy_model:9s} | "
                f"reward={summary['policy_reward_mean']:.6f} ± {summary['policy_reward_se']:.6f} | "
                f"visited={summary['visited_ratio']:.6f} | "
                f"otf_unique={summary['unique_on_the_fly_states']} | "
                f"otf_rollout_rate={summary['rollouts_with_on_the_fly_rate']:.6f}",
                flush=True,
            )

    with open(SUMMARY_FILE, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    pivot_text = format_policy_tables(summary_rows)
    print("\n" + pivot_text, flush=True)
    print(f"Saved: {SUMMARY_FILE}", flush=True)
    return summary_rows


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, OSError):
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 100, flush=True)
    print("4x4 Tic-Tac-Toe — policy-performance AMS with on-the-fly expansion", flush=True)
    print("=" * 100, flush=True)
    print(f"DEPTH={DEPTH}, SAMPLE_SIZES={SAMPLE_SIZES}", flush=True)
    print(f"POLICY_REPLICATIONS={POLICY_REPLICATIONS}, POLICY_ROLLOUTS={POLICY_ROLLOUTS}", flush=True)
    print(f"POLICY_MODELS={POLICY_MODELS}", flush=True)

    print("Warming up Numba...", end=" ", flush=True)
    start = time.time()
    warmup_board = make_initial_board()
    seed_rng(0)
    for code in (0, 1, 2):
        estimate_nb(warmup_board.copy(), 20, code, 2)
    for readout_code, estimator_code in ((0, 1), (1, 0), (2, 2), (3, 1)):
        seed_rng(10 + readout_code)
        on_the_fly_action_nb(warmup_board.copy(), 20, estimator_code, readout_code, 2)
    print(f"done ({time.time() - start:.1f}s)", flush=True)

    return test_policy_models(SAMPLE_SIZES, POLICY_MODELS)


if __name__ == "__main__":
    main()
