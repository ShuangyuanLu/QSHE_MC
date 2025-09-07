import numpy as np
from scipy.sparse import csc_matrix
from pathlib import Path
import h5py

def sparse_unequal_nonzero_mask(n_A: csc_matrix, n_B: csc_matrix) -> csc_matrix:
    """
    Return a sparse (n,1) CSC column vector with 1.0 at positions where:
        n_A[i] != 0 and n_B[i] != 0 and n_A[i] != n_B[i]
    Inputs:
        n_A, n_B: sparse column vectors (CSC) of shape (n, 1)
    Output:
        sparse column vector (CSC) with entries 1.0 where condition holds
    """
    # Make copies and clean up
    n_A = n_A.copy()
    n_B = n_B.copy()
    n_A.sum_duplicates()
    n_B.sum_duplicates()
    n_A.eliminate_zeros()
    n_B.eliminate_zeros()

    # Extract nonzero row indices and values
    rows_A = n_A.indices
    data_A = n_A.data
    rows_B = n_B.indices
    data_B = n_B.data

    # Intersect indices where both are nonzero
    common = np.intersect1d(rows_A, rows_B, assume_unique=True)

    # Create dict for B
    dict_B = dict(zip(rows_B, data_B))

    # Keep positions where a ≠ b
    rows_diff = [i for i, a in zip(rows_A, data_A)
                 if i in dict_B and a != dict_B[i]]

    # Build sparse mask
    data = np.ones(len(rows_diff), dtype=np.float64)
    shape = n_A.shape
    return csc_matrix((data, (rows_diff, np.zeros_like(rows_diff))), shape=shape)


def load_chain(file_path, dset="state_filled_check_box", slc=None):
    """
    file_path: path to .../chain_i/configs.h5
    dset:     dataset name
    slc:      slice on the time axis (e.g., slice(0, 1000)) or an integer index
    """
    with h5py.File(file_path, "r") as f:
        if dset not in f:
            raise KeyError(f"{dset} not found in {file_path}")
        ds = f[dset]
        # read all or a slice along axis 0
        arr = ds[...] if slc is None else ds[slc]
        # you can also read attributes:
        meta = dict(f.attrs)
    return arr, meta


def count_slices(file_path, dset="state_filled_check_box"):
    with h5py.File(file_path, "r") as f:
        ds = f[dset]                 # metadata only; no bulk read
        return ds.shape[0]           # or: len(ds)












# def sparse_unequal_mask(n_A: csc_matrix, n_B: csc_matrix) -> csc_matrix:
#     """
#     Return a sparse CSC (n,1) vector mask where (n_A != n_B) and both are nonzero.
#     Inputs: n_A, n_B — CSC column vectors of same shape (n, 1)
#     Output: CSC column vector of same shape, with 1.0 at positions where entries differ.
#     """
#     # Defensive copies and cleanup
#     n_A = n_A.copy()
#     n_B = n_B.copy()
#     n_A.sum_duplicates()
#     n_B.sum_duplicates()
#     n_A.eliminate_zeros()
#     n_B.eliminate_zeros()
#
#     # Fast access
#     rows_A = n_A.indices
#     data_A = n_A.data
#     rows_B = n_B.indices
#     data_B = n_B.data
#
#     # Intersect nonzero positions
#     common = np.intersect1d(rows_A, rows_B, assume_unique=True)
#
#     # Build lookup for B
#     dict_B = dict(zip(rows_B, data_B))
#
#     # Compare A[i] vs B[i] for common rows
#     rows_diff = [i for i, a in zip(rows_A, data_A)
#                  if i in dict_B and a != dict_B[i]]
#
#     # Output sparse mask (column vector)
#     data = np.ones(len(rows_diff), dtype=np.float64)
#     shape = n_A.shape
#     return csc_matrix((data, (rows_diff, np.zeros_like(rows_diff))), shape=shape)