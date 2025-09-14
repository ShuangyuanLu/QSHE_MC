import numpy as np
from ED_SQHE_check import ED_SQHE
from my_functions import load_chain, count_slices
import shelve
from SQHE import SQHE
import time
from SQHE_data_analysis import SQHE_data_analysis
import matplotlib.pyplot as plt


np.set_printoptions(linewidth=np.inf)
np.set_printoptions(threshold=np.inf)

parameters = {'t1': 1, 't2': 0.3, 'm': 0, 'phi': np.pi/2, 'p': 0.3, 'L': 40, 'n_mc': 10000, 'n_measure': 10, 'num_chains': 8, 'data_file': "data/data_set_7"}
my_model = ED_SQHE(parameters)
my_model.get_hamiltonian_obc()
my_model.get_eigenstates()
V = my_model.V

L = parameters['L']
N = my_model.N
n_sublat = 2
sublat_position = [[0, 0], [2/3, -1/3]]

x_pos = np.zeros(n_sublat * N)
y_pos = np.zeros(n_sublat * N)
for i_sublat in range(n_sublat):
    x_i = (np.array(range(L))[:, None] * np.ones(L)[None, :]).reshape(N) + sublat_position[i_sublat][0]
    y_i = (np.ones(L)[:, None] * np.array(range(L))[None, :]).reshape(N) + sublat_position[i_sublat][1]
    x_pos[i_sublat * N: (i_sublat + 1) * N] = x_i + y_i * 1/2
    y_pos[i_sublat * N: (i_sublat + 1) * N] = y_i * np.sqrt(3) / 2

start = time.time()

r_list = [i for i in range(1, L // 2)]
psi_list = []
for i_r in range(len(r_list)):
    r = r_list[i_r]
    x1 = (L - r) // 2
    x2 = x1 + r
    y1, y2 = L // 2, L // 2
    n1 = x1 * L + y1
    n2 = x2 * L + y2

    phase_x1 = x_pos - x_pos[n1] - 1j * (y_pos - y_pos[n1])
    phase_x1[n1] = 1
    phase_x1 = phase_x1 / np.abs(phase_x1)

    phase_x2 = x_pos - x_pos[n2] - 1j * (y_pos - y_pos[n2])
    phase_x2[n2] = 1
    phase_x2 = phase_x2 / np.abs(phase_x2)

    psi_1_list = V[n1, :].copy()
    psi_2_list = V[n2, :].copy()
    V1 = phase_x1[:, None] * V
    V2 = phase_x2[:, None] * V
    V1[n1, :] = 0
    V2[n2, :] = 0

    overlap = V2.conj().T @ V1

    # psi = 0
    # for i1 in range(N):
    #     for i2 in range(N):
    #         psi += (-1) ** (i1 + i2) * psi_1_list[i1] * psi_2_list[i2].conj() * np.linalg.det(np.delete(np.delete(overlap, i2, axis=0), i1, axis=1))

    t = 1
    overlap_perturbed = overlap + t * np.outer(psi_2_list.conj(), psi_1_list)

    det = np.linalg.det(overlap)
    det_perturbed = np.linalg.det(overlap_perturbed)
    #print("psi:", det, det_perturbed)
    psi = np.abs((det_perturbed - det) / t)
    psi_list.append(np.log10(psi))

end = time.time()
print("time:", end-start)

print(psi_list)

plt.plot(r_list, psi_list)
plt.savefig("test.png")



