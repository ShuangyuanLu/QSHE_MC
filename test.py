import numpy as np
import matplotlib.pyplot as plt
import shelve



# test = np.array([0.00000000e+00+0.00000000e+00j, 9.05838163e-03+1.60114302e-05j, 1.14642323e-03+2.01380821e-05j, 3.07786122e-04-1.56713103e-05j, 8.94715636e-05+5.18620631e-06j, 4.07687083e-05+7.66627620e-06j, 1.20049033e-05+4.44418878e-06j, 3.58124723e-06-1.81829024e-06j, 1.29648440e-06+6.82122779e-06j, 7.13915244e-06+1.86447643e-07j, 5.07931179e-06-4.31239709e-06j, 1.25217825e-05-3.04126729e-06j, 1.23291683e-05+5.51471722e-06j, 1.27726166e-05-1.88453838e-06j, 3.29951666e-05+1.06540329e-05j, 3.75403711e-05+7.31897763e-06j, 9.90823750e-05+1.43691009e-06j, 3.02533518e-04+1.62113225e-05j, 1.14624018e-03+4.71336799e-06j, 9.05854704e-03+2.64452991e-05j], dtype=np.complex128)
# test = np.array([ 6.60308125e-01+0.j,  7.25620833e-03+0.j,  2.01500000e-04+0.j,  2.76083333e-04+0.j, -1.12437500e-04+0.j, -2.66250000e-05+0.j,  2.15958333e-04+0.j, -5.16666667e-05+0.j, -5.72916667e-05+0.j, -1.12291667e-04+0.j,  2.01583333e-04+0.j, -1.12291667e-04+0.j, -5.72916667e-05+0.j, -5.16666667e-05+0.j,  2.15958333e-04+0.j, -2.66250000e-05+0.j, -1.12437500e-04+0.j,  2.76083333e-04+0.j,  2.01500000e-04+0.j,  7.25620833e-03+0.j])

with shelve.open('data/data_set_8/result') as db:
    psi_correlation = db['psi_correlation']
    sz_correlation = db["sz_correlation"]

n = 8
# # plt.plot(np.arange(1, psi_correlation.shape[0]), np.abs(psi_correlation[1:]), "o-", linewidth=0.8, markersize=2)
# plt.plot(np.arange(1, n), np.log(np.abs(psi_correlation[1:n])), "o-", linewidth=0.8, markersize=2)
#plt.plot(np.arange(1, sz_correlation.shape[0]), np.real(sz_correlation[1:]), "o-", linewidth=0.8, markersize=2)
plt.plot(np.arange(1, n), np.log(np.abs(sz_correlation[1:n])), "o-", linewidth=0.8, markersize=2)
plt.savefig('test.pdf')
