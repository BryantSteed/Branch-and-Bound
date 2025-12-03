# Project Report - Branch and Bound

## Baseline

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

For the reduced cost matrix, I will represent that matrix as a two dimensional array (or a list of lists). With that, I will perform the column reductions (as specified in the slides) and then the row reductions.

Will then return the lower bound calculated by the algorithm along with the final reduced cost matrix.

She said she used a dictionary. He used a numpy array. We thought the best idea was to put the function in a separate spot.

### Theoretical Analysis - Reduced Cost Matrix

#### Time - **O(n^2)**

```py
def calculate_reduced_cost(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    row_mins = np.min(matrix, axis=1) # O(n^2)
    valid_row = np.isfinite(row_mins) & (row_mins > 0) # O(n)
    row_vals = np.where(valid_row, row_mins, 0.0) # O(n)
    matrix -= row_vals[:, None] # O(n^2)
    reduction_modifier = float(row_vals.sum()) # O(n)

    col_mins = np.min(matrix, axis=0)
    valid_col = np.isfinite(col_mins) & (col_mins > 0)
    col_vals = np.where(valid_col, col_mins, 0.0)
    matrix -= col_vals
    reduction_modifier += float(col_vals.sum())

    return matrix, reduction_modifier
```

This took a lot to get it this performant. I'll only go over the one for row because its the same for columns. Getting the row mins vector is O(n^2) because you iterate over each row. Computing the valid row boolean vector is linear because you take two n length vectors and compare them.

Computing the row vals (which contains the amount to subtract from each row) vector is also linear because it uses an n length bool vector and uses row mins where True and 0.0 where not.

Subtracting the row vals from the matrix is O(n^2) because you are subtracting at potentially every position of the matrix. It broadcasts across all columns: thats how it works. Computing the the sum of modifications is linear because there are n column values.

So, Computing the row mins dominates, making the time complexity **O(n^2)**

#### Space - **O(n^2)**

```py
def calculate_reduced_cost(matrix: np.ndarray) -> Tuple[np.ndarray, float]: # O(n^2)
    row_mins = np.min(matrix, axis=1) # O(n)
    valid_row = np.isfinite(row_mins) & (row_mins > 0) # O(n)
    row_vals = np.where(valid_row, row_mins, 0.0) # O(n)
    matrix -= row_vals[:, None] # O(n^2)    2D Matrix creation from row_vals
    reduction_modifier = float(row_vals.sum()) # O(1)

    col_mins = np.min(matrix, axis=0)
    valid_col = np.isfinite(col_mins) & (col_mins > 0)
    col_vals = np.where(valid_col, col_mins, 0.0)
    matrix -= col_vals
    reduction_modifier += float(col_vals.sum())

    return matrix, reduction_modifier
```

The data structure here that takes up the most space is the matrix itself with O(n^2) space complexity. For our intermediary calculations, we take compute some vectors for valid rows and columns. We also create another 2D array from which we subtract the original Matrix. That matrix takes up O(n^2) space.

Because of the matrix data structure, our space complexity is **O(n^2)**

## Core

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

The idea will be to take a certain partial state as a reduced cost matrix and expand it to the different possible reduced cost matrices. I will make use of my reduced cost matrix algorithm to compute lower bounds as I go. If the lower bound is already worse than my best solution so far, then I will not explore that path.

As I go, I will keep track of which path corresponds to that given partial state. I will likely use a python list as the data structure to store this. I will store the reduced cost matrix itself with this path using two nested lists. The con of using this is that is can grow to be very large and waste space. The pros are that access with be extremely fast. I will estimate the branching factor by making an educated guess on how many paths will be pruned.

We talked about using a stack to traverse the solution space. She stored the partial states in a class that contained the matrix and the path taken to get there.

### Theoretical Analysis - Branch and Bound TSP

#### Time 

```py
def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    greedy_solutions = greedy_tour(edges, timer) # O(n^2)
    bssf = greedy_solutions[-1].score if greedy_solutions else math.inf
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
```

In the previou

```py
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
```
```py
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
        elif math.isinf(new_cost_matrix[i][new_path[0]]):
                continue
        new_reduced_matrix, new_reduction_cost = calculate_reduced_cost(new_cost_matrix)
        new_cost = state.cost + new_reduction_cost + edge_cost
        children.append(TSPState(new_reduced_matrix, new_path, new_cost))
    return children
```

#### Space

*Fill me in*

### Empirical Data

| N   | Seed | Solution | time (ms) |
|-----|------|----------|-----------|
| 5   |      |          |           |
| 10  |      |          |           |
| 15  |      |          |           |
| 20  |      |          |           |
| 30  |      |          |           |
| 50  |      |          |           |

### Comparison of Theoretical and Empirical Results

- Empirical order of growth: 
- Measured constant of proportionality: 

![img](img.png)

*Fill me in*

## Stretch 1 

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

CutTree works by recording each time you prune. It tells you the percentage of the tree that a given algorithm has covered. This is nice because you get an idea as to how efficiently it is searching the solution space.

My plots should contain the coverage and the number of leaves pruned from the tree. This will be very useful because it will tell me how efficient the algorithm is and how much its pruning from the tree.

### Search Space Over Time

![Plot demonstrating search space explored over time]()

*Fill me in*

## Stretch 2

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

This time, instead of using the stack, I will use a priority queue. I will use python's built in Heap Priority queue from the python standard library. We are not allowed to use the lower bound as the key so I will decide on something else. One option would be to use the length of the path in a given partial state. Another option would be to search the larger paths first (this would be a max pq).

They said that they were using some weight between the lower bound and the weight of the path as a weight.

### Selected PQ Key

*Fill me in*

### Branch and Bound versus Smart Branch and Bound

*Fill me in*

## Project Report 

*Fill me in*

