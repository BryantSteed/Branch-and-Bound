# See additional instructions for these tests in the instructions for the project
import numpy as np
import math
from reduced_cost import calculate_reduced_cost
from tsp_solve import branch_and_bound
from tsp_core import Timer


def test_reduced_cost_matrix_1():
    cost_matrix = np.array([
        [math.inf, 7, 3, 12],
        [3, math.inf, 6, 14],
        [5, 8, math.inf, 6],
        [9, 3, 5, math.inf]
    ])

    reduced_matrix, reduction_modifier = calculate_reduced_cost(cost_matrix)
    expected_reduced_matrix = np.array([
        [math.inf, 4, 0, 8],
        [0, math.inf, 3, 10],
        [0, 3, math.inf, 0],
        [6, 0, 2, math.inf]
    ])
    expected_reduction_modifier = 15

    assert np.array_equal(reduced_matrix, expected_reduced_matrix), "Reduced matrix does not match expected output"
    assert reduction_modifier == expected_reduction_modifier, "Reduction modifier does not match expected output"


def test_reduced_cost_matrix_2():
    cost_matrix = np.array([
        [math.inf, 20, 30, 10, 11],
        [15, math.inf, 16, 4, 2],
        [3, 5, math.inf, 2, 4],
        [19, 6, 18, math.inf, 3],
        [16, 4, 7, 16, math.inf]
    ])
    reduced_matrix, reduction_modifier = calculate_reduced_cost(cost_matrix)
    expected_reduced_matrix = np.array([
        [math.inf, 10, 17, 0, 1],
        [12, math.inf, 11, 2, 0],
        [0, 3, math.inf, 0, 2],
        [15, 3, 12, math.inf, 0],
        [11, 0, 0, 12, math.inf]
    ])
    expected_reduction_modifier = 25

    assert np.array_equal(reduced_matrix, expected_reduced_matrix), "Reduced matrix does not match expected output"
    assert reduction_modifier == expected_reduction_modifier, "Reduction modifier does not match expected output"


# Add more tests as necessary...

