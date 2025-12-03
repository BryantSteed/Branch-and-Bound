from tsp_solve import branch_and_bound, branch_and_bound_smart
from tsp_core import Timer, generate_network, score_tour
from tsp_test_utils import assert_valid_tours
from copy import deepcopy
import math

import sys

# locations, edges = generate_network(
#     12,
#     euclidean=True,
#     reduction=0.2,
#     normal=False,
#     seed=312,
# )

edges = [[0.0, math.inf, math.inf, 0.219, 0.318, math.inf, 0.387, 0.362, 0.491, 0.422, math.inf, 0.535], [0.8, math.inf, 0.94, math.inf, 1.08, 0.3, 0.764, 1.144, 0.586, 0.687, 0.249, 0.779], [0.564, 0.94, 0.0, 0.377, 0.497, math.inf, 0.93, 0.573, math.inf, 0.94, 0.696, 1.069], [0.219, 0.717, 0.377, 0.0, 0.374, 0.456, 0.555, 0.447, math.inf, 0.563, math.inf, 0.692], [0.318, math.inf, 0.497, 0.374, 0.0, math.inf, 0.669, 0.081, math.inf, 0.723, 0.845, 0.817], [0.595, 0.3, 0.64, 0.456, 0.83, 0.0, 0.702, 0.902, 0.59, 0.65, 0.076, 0.774], [0.387, 0.764, 0.93, 0.555, 0.669, 0.702, 0.0, 0.683, 0.208, 0.087, 0.654, 0.15], [0.362, 1.144, 0.573, math.inf, 0.081, 0.902, 0.683, 0.0, 0.834, 0.744, 0.913, 0.829], [0.491, math.inf, 0.964, 0.593, math.inf, math.inf, 0.208, 0.834, 0.0, 0.12, 0.527, 0.194], [0.422, 0.687, 0.94, 0.563, 0.723, 0.65, math.inf, 0.744, 0.12, 0.0, 0.596, 0.134], [0.588, math.inf, 0.696, 0.476, 0.845, 0.076, 0.654, 0.913, 0.527, 0.596, math.inf, 0.715], [0.535, 0.779, 1.069, math.inf, 0.817, 0.774, 0.15, 0.829, 0.194, 0.134, 0.715, 0.0]]

# print(repr(edges))

if sys.argv[1] == 'bb':
    bb_func = branch_and_bound
elif sys.argv[1] == 'bbsmart':
    bb_func = branch_and_bound_smart
else:
    raise ValueError("First argument must be 'bb' or 'bbsmart'")

timer = Timer(120)
stats = bb_func(deepcopy(edges), timer)
assert not timer.time_out()
assert_valid_tours(edges, stats)
bnb_score = score_tour(stats[-1].tour, edges)