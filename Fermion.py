import numpy as np
from scipy.sparse import csr_matrix, eye

def fermion_ops(N):
    """
    Spinless fermions, sites indexed j=0..N-1.
    Returns lists: c  (annihilation), cd (creation), n (number) as CSR matrices.
    Basis is computational bit basis |n_{N-1} ... n_1 n_0>.
    """
    dim = 1 << N
    c_list, cd_list, n_list = [], [], []

    for j in range(N):
        rows_c, cols_c, data_c = [], [], []
        rows_cd, cols_cd, data_cd = [], [], []

        mask_j = 1 << j
        # bitmask for sites "to the left" of j (i.e., lower indices 0..j-1)
        left_mask = mask_j - 1

        for s in range(dim):
            occ = (s >> j) & 1
            # parity = (-1)^{# occupied to the left of j}
            parity = -1 if (bin(s & left_mask).count("1") & 1) else 1

            if occ:  # c_j |s> -> parity |s with bit j cleared>
                s_new = s ^ mask_j
                rows_c.append(s_new); cols_c.append(s); data_c.append(parity)
            else:    # c_j^\dagger |s> -> parity |s with bit j set>
                s_new = s | mask_j
                rows_cd.append(s_new); cols_cd.append(s); data_cd.append(parity)

        c  = csr_matrix((np.array(data_c, dtype=np.float64), (rows_c,  cols_c)), shape=(dim, dim))
        cd = csr_matrix((np.array(data_cd, dtype=np.float64), (rows_cd, cols_cd)), shape=(dim, dim))
        n  = (cd @ c).tocsr()

        c_list.append(c)
        cd_list.append(cd)
        n_list.append(n)

    return c_list, cd_list, n_list

def fermion_dense_ops(M: int, dtype=np.complex128):
    """
    Build dense creation/annihilation/number operators for M spinless fermionic modes.
    Basis ordering: |n_{M-1} ... n_1 n_0>, with n_0 the least-significant bit.
    Jordan–Wigner string counts occupied bits with indices < j (to the right).
    """
    dim = 1 << M
    c  = [np.zeros((dim, dim), dtype=dtype) for _ in range(M)]
    cd = [np.zeros((dim, dim), dtype=dtype) for _ in range(M)]

    for j in range(M):
        mask_j = 1 << j
        left_mask = mask_j - 1  # bits < j
        # Fill columns by basis state |s>
        for s in range(dim):
            occ = (s >> j) & 1
            # parity = (-1)^{popcount(s & left_mask)}
            parity = -1 if ((s & left_mask).bit_count() & 1) else 1
            if occ:
                # c_j |s> = parity |s with bit j cleared>
                s_new = s ^ mask_j
                c[j][s_new, s] = parity
            else:
                # c_j^† |s> = parity |s with bit j set>
                s_new = s | mask_j
                cd[j][s_new, s] = parity

    n = [cd[j] @ c[j] for j in range(M)]
    I = np.eye(dim, dtype=dtype)
    return c, cd, n, I

# --- Example usage ---
# N = 2
# c, cd, n, _ = fermion_dense_ops(N)
# for i in range(N):
#     print(n[i])

