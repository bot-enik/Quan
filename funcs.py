import numpy as np

def nth_upper_diagonal_indices(size, n):
    """
    Returns the n-th upper diagonal indices of a square matrix of given size for direct array indexing.

    :param size: Size of the square matrix
    :param n: The n-th upper diagonal (0 for the main diagonal)
    :return: Two arrays, one for row indices and another for column indices, for direct indexing
    """
    # Calculate the row indices, which are straightforward
    row_indices = np.arange(max(0, n), size)
    # Calculate the column indices, offset by n from the row indices
    col_indices = np.arange(0, size - n)
    
    return col_indices, row_indices