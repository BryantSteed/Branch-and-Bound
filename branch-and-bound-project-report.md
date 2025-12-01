# Project Report - Branch and Bound

## Baseline

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

For the reduced cost matrix, I will represent that matrix as a two dimensional array (or a list of lists). With that, I will perform the column reductions (as specified in the slides) and then the row reductions.

Will then return the lower bound calculated by the algorithm along with the final reduced cost matrix.

She said she used a dictionary. He used a numpy array. We thought the best idea was to put the function in a separate spot.

### Theoretical Analysis - Reduced Cost Matrix

#### Time 

*Fill me in*

#### Space

*Fill me in*

## Core

### Design Experience

Jennifer Stone and Tom Esplin on 12/01/2025.

The idea will be to take a certain partial state as a reduced cost matrix and expand it to the different possible reduced cost matrices. I will make use of my reduced cost matrix algorithm to compute lower bounds as I go. If the lower bound is already worse than my best solution so far, then I will not explore that path.

As I go, I will keep track of which path corresponds to that given partial state. I will likely use a python list as the data structure to store this. I will store the reduced cost matrix itself with this path using two nested lists. The con of using this is that is can grow to be very large and waste space. The pros are that access with be extremely fast. I will estimate the branching factor by making an educated guess on how many paths will be pruned.

We talked about using a stack to traverse the solution space. She stored the partial states in a class that contained the matrix and the path taken to get there.

### Theoretical Analysis - Branch and Bound TSP

#### Time 

*Fill me in*

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

