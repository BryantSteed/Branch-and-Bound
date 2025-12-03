import math
import random
import numpy as np
from heapq import heappop, heappush, heapify

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

class TSPState:
    def __init__(self, cost_matrix: np.ndarray, path: Tour, cost: float):
        self.cost_matrix: np.ndarray = cost_matrix
        self.path: Tour = path
        self.cost: float = cost

    def __lt__(self, other: 'TSPState') -> bool:
        len_self = len(self.path)
        len_other = len(other.path)
        if len_self == len_other:
            return self.cost < other.cost
        return len_self > len_other


def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    num_nodes = len(edges[0])
    stats = []
    global_best_score = math.inf
    for start_node in range(num_nodes):
        path = [start_node]
        visited = set([start_node])
        current_node = start_node
        if timer.time_out():
            return stats
        process_start_node_greedy(edges, num_nodes, path, visited, current_node)
        global_best_score = add_stat_greedy(edges, timer, stats, global_best_score, path)
    return stats

def process_start_node_greedy(edges, num_nodes, path, visited, current_node):
    while len(visited) < num_nodes:
        valid_nodes = set()
        for node in range(num_nodes):
            if node not in visited and not math.isinf(edges[current_node][node]):
                valid_nodes.add(node)
        if not valid_nodes:
            break
        curr_min = math.inf
        for node in valid_nodes:
            if edges[current_node][node] < curr_min:
                curr_min = edges[current_node][node]
                best_node = node
        path.append(best_node)
        visited.add(best_node)
        current_node = best_node

def add_stat_greedy(edges, timer, stats, global_best_score, path):
    if len(path) != len(edges):
        return global_best_score
    cost = score_tour(path, edges)
    if cost < global_best_score:
        global_best_score = cost
        stat: SolutionStats = SolutionStats(tour=path,
                                               score=cost,
                                               time=timer.time(),
                                               max_queue_size=0,
                                               n_nodes_expanded=0,
                                               n_nodes_pruned=0,
                                               n_leaves_covered=0,
                                               fraction_leaves_covered=0.0)
        stats.append(stat)
    return global_best_score


def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    return []


def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    greedy_solutions = greedy_tour(edges, timer)
    bssf = greedy_solutions[-1].score if greedy_solutions else math.inf
    # bssf = math.inf
    initial_matrix = np.array(edges)
    initial_tour = [0]
    initial_reduced_matrix , initial_reduction_cost = calculate_reduced_cost(initial_matrix)
    stack = [TSPState(initial_reduced_matrix, initial_tour, initial_reduction_cost)]
    stat_lst = []
    while stack:
        if timer.time_out():
            return stat_lst
        state = stack.pop()
        if state.cost >= bssf:
            continue
        child_states = expand_bb_state(state, bssf)
        bssf = process_children_bb(child_states, bssf, stat_lst, stack, timer, edges)
    return stat_lst

def process_children_bb(child_states: list[TSPState], 
                     bssf: float, 
                     stat_lst: list[SolutionStats], 
                     stack: list[TSPState], 
                     timer: Timer, 
                     edges: list[list[float]]):
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
    return bssf

def expand_bb_state(state: TSPState, bssf: float) -> list[TSPState]:
    children = []
    visited = set(state.path)
    for i in range(len(state.cost_matrix)):
        if i in visited or math.isinf(state.cost_matrix[state.path[-1]][i]):
            continue
        new_path = state.path + [i]

        edge_cost = state.cost_matrix[state.path[-1]][i]

        new_cost_matrix = state.cost_matrix.copy()
        for j in range(len(new_cost_matrix)):
            new_cost_matrix[state.path[-1]][j] = math.inf
            new_cost_matrix[j][i] = math.inf
        if len(new_path) != len(state.cost_matrix):
            new_cost_matrix[i][new_path[0]] = math.inf
        
        new_reduced_matrix, new_reduction_cost = calculate_reduced_cost(new_cost_matrix)
        new_cost = state.cost + new_reduction_cost + edge_cost
        children.append(TSPState(new_reduced_matrix, new_path, new_cost))
    # children.sort(key=lambda x: -(x.cost))
    return children

def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    greedy_solutions = greedy_tour(edges, timer)
    bssf = greedy_solutions[-1].score if greedy_solutions else math.inf
    # bssf = math.inf
    initial_matrix = np.array(edges)
    initial_tour = [0]
    initial_reduced_matrix , initial_reduction_cost = calculate_reduced_cost(initial_matrix)
    pq = [TSPState(initial_reduced_matrix, initial_tour, initial_reduction_cost)]
    heapify(pq)
    stat_lst = []
    while pq:
        if timer.time_out():
            return stat_lst
        state = heappop(pq)
        if state.cost >= bssf:
            continue
        child_states = expand_bbsmart_state(state, bssf)
        bssf = process_children_bbsmart(child_states, bssf, stat_lst, pq, timer, edges)
    return stat_lst

def process_children_bbsmart(child_states: list[TSPState], 
                     bssf: float, 
                     stat_lst: list[SolutionStats], 
                     pq: list[TSPState], 
                     timer: Timer, 
                     edges: list[list[float]]):
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
            heappush(pq, child)
    return bssf

def expand_bbsmart_state(state: TSPState, bssf: float) -> list[TSPState]:
    children = []
    visited = set(state.path)
    for i in range(len(state.cost_matrix)):
        if i in visited or math.isinf(state.cost_matrix[state.path[-1]][i]):
            continue
        new_path = state.path + [i]

        edge_cost = state.cost_matrix[state.path[-1]][i]

        new_cost_matrix = state.cost_matrix.copy()
        for j in range(len(new_cost_matrix)):
            new_cost_matrix[state.path[-1]][j] = math.inf
            new_cost_matrix[j][i] = math.inf
        if len(new_path) != len(state.cost_matrix):
            new_cost_matrix[i][new_path[0]] = math.inf
        
        new_reduced_matrix, new_reduction_cost = calculate_reduced_cost(new_cost_matrix)
        new_cost = state.cost + new_reduction_cost + edge_cost
        children.append(TSPState(new_reduced_matrix, new_path, new_cost))
    return children