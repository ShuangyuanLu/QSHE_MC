import numpy as np
import matplotlib.pyplot as plt
import shelve

test_a = np.array([0, 1, 0, 0], dtype=int)
test_b = np.array([0, 0, 1, 0], dtype=int)
test_c = test_a != test_b
test_d = test_a.copy()
test_d[2: 4] = 1-test_a[2:4]

mask = (test_c == True)
positions = np.argwhere(mask)
mask = np.array([False, False, False, False])
indices = np.where(mask)[0]

print(indices)
