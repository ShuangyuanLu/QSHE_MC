import shelve

import matplotlib.pyplot as plt
import numpy as np


file_name = "data/data_set_10/results"
with shelve.open(file_name) as db:
    corr = db["correlation_sz"]

N = 20
print(corr)

plt.plot(np.arange(1, N), np.abs(corr[1:N]), 'o-', linewidth=0.8, markersize=2)
plt.xlabel(r'$r$')
plt.ylabel(r'$ C_2$')
plt.savefig("correlation_sz.pdf")









# plot correlation psi
# file_name = "data/data_set_10/results"
# with shelve.open(file_name) as db:
#     corr = db["correlation_psi"]
# N = 10
# plt.plot(np.log10(np.arange(1, N)), np.log10(np.abs(corr[1:N])), 'o-', linewidth=0.8, markersize=2)
# plt.xlabel(r'$\log_{10} r$')
# plt.ylabel(r'$\log_{10} C_2$')
# plt.savefig("correlation_psi.pdf")