import math
import random
import numpy as np

from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver
from tsp_cuttree import CutTree
from reduced_cost import calculate_reduced_cost

PARAMS_FOR_SMART_BRANCH_AND_BOUND_SMART_TEST = {
    "n": 30,
    "euclidean": True,
    "reduction": 0.2,
    "normal": False,
    "seed": 312,
    "timeout" : 20
}

def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]

def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []


def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    bssf = math.inf
    initial_matrix = np.array(edges)
    initial_tour = [0]
    for i in range(len(edges)):
        if i != 0:
            initial_matrix[0][i] = math.inf
    initial_reduced_matrix , initial_reduction_cost = calculate_reduced_cost(initial_matrix)
    
    stack = [TSPState(initial_reduced_matrix, initial_tour, initial_reduction_cost)]
    stat_lst = []
    while stack:
        if timer.time_out():
            return stat_lst
        state = stack.pop()
        child_states = expand_bb_state(state)
        for child in child_states:
            if len(child.path) == len(edges):
                if child.cost < bssf:
                    bssf = child.cost
                    stat_lst.append(SolutionStats(
                        tour=child.path,
                        score=child.cost,
                        time=timer.time(),
                        max_queue_size=0,
                        n_nodes_expanded=0,
                        n_nodes_pruned=0,
                        n_leaves_covered=0,
                        fraction_leaves_covered=0.0
                    ))
            elif child.cost >= bssf:
                continue
            else:
                stack.append(child)
    return stat_lst

def expand_bb_state(state: TSPState) -> list[TSPState]:
    children = []
    visited = set(state.path)
    for i in range(len(state.cost_matrix)):
        if i in visited:
            continue
        new_path = state.path + [i]
        new_cost_matrix = state.cost_matrix.copy()
        for j in range(len(new_cost_matrix)):
            new_cost_matrix[state.path[-1]][j] = math.inf
            new_cost_matrix[j][i] = math.inf
        if len(new_path) != len(state.cost_matrix):
            new_cost_matrix[i][new_path[0]] = math.inf
        new_reduced_matrix , additional_reduction_cost = calculate_reduced_cost(new_cost_matrix)
        new_cost = state.cost + additional_reduction_cost
        children.append(TSPState(new_cost_matrix, new_path, new_cost))
    return children
        


def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []

class TSPState:
    def __init__(self, cost_matrix: np.ndarray, path: Tour, cost: float):
        self.cost_matrix: np.ndarray = cost_matrix
        self.path: Tour = path
        self.cost: float = cost