# This is my own implementation of reduced cost calculation
import numpy as np
from typing import Tuple

def calculate_reduced_cost(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    row_mins = np.min(matrix, axis=1)
    valid_row = np.isfinite(row_mins) & (row_mins > 0)
    row_vals = np.where(valid_row, row_mins, 0.0)
    matrix -= row_vals[:, None]
    reduction_modifier = float(row_vals.sum())

    col_mins = np.min(matrix, axis=0)
    valid_col = np.isfinite(col_mins) & (col_mins > 0)
    col_vals = np.where(valid_col, col_mins, 0.0)
    matrix -= col_vals
    reduction_modifier += float(col_vals.sum())

    return matrix, reduction_modifier