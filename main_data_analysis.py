import numpy as np
from my_functions import load_chain, count_slices
import shelve
from SQHE import SQHE
import time
from SQHE_data_analysis import SQHE_data_analysis
import matplotlib.pyplot as plt


def avg_corr_roll(S):
    """
    C[dx, dy] = sum_{x1,y1} S[x1, y1, (x1+dx)%Lx, (y1+dy)%Ly]
    """
    Lx, Ly, Lx2, Ly2 = S.shape
    C = np.zeros(Lx, dtype=S.dtype)

    for r in range(Lx):
        Sx = np.roll(S, shift=r, axis=2)  # shift x2 by dx
        Sy = np.roll(S, shift=r, axis=3)
        S_x_minus_y = np.roll(S, shift=(r, -r), axis=(2, 3))
        C[r] += np.einsum('ijij->', Sx)
        C[r] += np.einsum('ijij->', Sy)
        C[r] += np.einsum('ijij->', S_x_minus_y)
    C = C / (Lx * Ly) / 3
    return C


np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)
data_file = "data/data_set_8"
with shelve.open(data_file + "/parameters") as db:
    parameters = db["parameters"]

my_model = SQHE(parameters)
n_chains = parameters["num_chains"]

#arr_0_100, _ = load_chain("data/data_set_0/chain_0/configs.h5", slc=slice(0, 100))
#arr_0_100, _ = load_chain("data/data_set_0/chain_0/configs.h5")

start = time.time()
correlation_all = np.zeros(my_model.L, dtype=np.complex128)
n_data_point = 0
N = my_model.N
correlation_ij = np.zeros((2*N, 2*N), dtype=np.complex128)
for i_chain in range(n_chains):
    print(i_chain)
    file_name_i = data_file + "/chain_" + str(i_chain) +"/configs.h5"
    n_slc = count_slices(file_name_i)
    n_data_point += n_slc
    check_boxes, _ = load_chain(file_name_i)

    for i_measurement in range(n_slc):
        if i_measurement % (n_slc // 10) == 0:
            print(i_measurement)
        #my_model.state_filled_check_box, _ = load_chain(file_name_i, slc=i_measurement)
        my_model.state_filled_check_box = check_boxes[i_measurement, :, :]

        check_box = my_model.state_filled_check_box
        spin_z_A_z_B = (check_box[:, 0] - check_box[:, 1]) * (check_box[:, 2] - check_box[:, 3])
        correlation_ij += np.outer(spin_z_A_z_B, spin_z_A_z_B)

        # my_model.recover_data_from_check_box()
        # i_idx = np.flatnonzero((my_model.state_filled_check_box == [1, 0, 1, 0]).all(axis=1))  # indices i
        # j_idx = np.flatnonzero((my_model.state_filled_check_box == [0, 1, 0, 1]).all(axis=1))  # indices j
        # wave_function_ratio = np.zeros((i_idx.size, j_idx.size, my_model.n_layer), dtype=np.complex128)
        # for layer in range(my_model.n_layer):
        #     A = my_model.wave_function_matrix_inverse[:, :, layer]
        #     if layer == 0 or layer == 2:
        #         i_list, j_list = i_idx, j_idx
        #     else:
        #         i_list, j_list = j_idx, i_idx
        #     matches = my_model.state_filled_site_list[:, layer] == i_list[:, np.newaxis]
        #     i_0_list = np.where(matches)[1]
        #     state_A_i = np.einsum("ik,ki->i", my_model.state_list[i_list, :, layer], A[:, i_0_list])
        #     if layer == 0 or layer == 2:
        #         wave_function_ratio[:, :, layer] = np.conj((my_model.state_list[j_list, :, layer] @ A[:, i_0_list]).T)
        #     else:
        #         wave_function_ratio[:, :, layer] = np.conj((my_model.state_list[j_list, :, layer] @ A[:, i_0_list]).T).T
        #
        # correlation_ij[i_idx[:, None], j_idx] += wave_function_ratio[:, :, 0] * wave_function_ratio[:, :, 1] * wave_function_ratio[:, :, 2] * wave_function_ratio[:, :, 3]


correlation_ij = correlation_ij / n_data_point
print(correlation_ij)
correlation_ij = ((correlation_ij[:N, :N] + correlation_ij[N:2*N, N:2*N]) / 2).reshape(my_model.L, my_model.L, my_model.L, my_model.L)
correlation_all = avg_corr_roll(correlation_ij)

print(correlation_all)

end = time.time()
print("time:", end - start)

plt.plot(np.real(correlation_all))
plt.savefig("correlation_all.png")











