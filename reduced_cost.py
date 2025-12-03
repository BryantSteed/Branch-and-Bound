# This is my own implementation of reduced cost calculation
import numpy as np
import math
from typing import Tuple

def calculate_reduced_cost(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Vectorized reduced-cost computation.
    - If inplace=False, operates on a copy.
    - Leaves rows/cols that are all inf unchanged.
    - Only subtracts positive minima (matches original behavior).
    """
    # row reduction
    row_mins = np.min(matrix, axis=1)                     # shape (n_rows,)
    valid_row = np.isfinite(row_mins) & (row_mins > 0)    # only finite and >0
    row_vals = np.where(valid_row, row_mins, 0.0)
    matrix -= row_vals[:, None]
    reduction_modifier = float(row_vals.sum())

    # column reduction
    col_mins = np.min(matrix, axis=0)                     # shape (n_cols,)
    valid_col = np.isfinite(col_mins) & (col_mins > 0)
    col_vals = np.where(valid_col, col_mins, 0.0)
    matrix -= col_vals
    reduction_modifier += float(col_vals.sum())

    return matrix, reduction_modifier