# This is my own implementation of reduced cost calculation
import numpy as np
import math
from typing import Tuple

def calculate_reduced_cost(matrix: np.ndarray) -> Tuple[np.ndarray, int]:
    reduction_modifier = 0
    for i in range(matrix.shape[0]):
        row_min = np.min(matrix[i])
        if row_min > 0 and row_min != math.inf:
            matrix[i] -= row_min
            reduction_modifier += row_min

    for j in range(matrix.shape[1]):
        col_min = np.min(matrix[:, j])
        if col_min > 0 and col_min != math.inf:
            matrix[:, j] -= col_min
            reduction_modifier += col_min
    
    return matrix, reduction_modifier