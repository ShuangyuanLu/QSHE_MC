import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import shelve
from pathlib import Path
import h5py
import time

class SQHE:
    def __init__(self, parameters):
        self.t1 = parameters['t1']
        self.t2 = parameters['t2']
        self.m = parameters['m']
        self.phi = parameters['phi']
        self.p = parameters['p']
        self.L = parameters['L']
        self.n_mc = parameters['n_mc']
        self.n_measure = parameters['n_measure']
        self.data_file = parameters['data_file']

        self.start_to_measure = 1/5 * self.n_mc
        self.n_recalculate_inverse = 50

        self.n_layer = 4
        self.n_sublat = 2
        self.N = self.L ** 2
        self.sublat_position = [[0, 0], [2/3, -1/3]]

        self.hamiltonian = None
        self.measurement = []

        # variables to store state info
        self.wave_function_matrix = np.zeros((self.N, self.N, self.n_layer), dtype=np.complex128)
        self.wave_function_matrix_inverse = np.zeros((self.N, self.N, self.n_layer), dtype=np.complex128)
        self.state_list_k = np.zeros((self.L, self.L, 2), dtype=np.complex128)
        # site order 1 ... N for sublat 1, 1 ... N for sublat 2
        self.state_list = np.zeros((self.N * self.n_sublat, self.N, self.n_layer), dtype=np.complex128)   # site, momentum, layer

        # variables updated in monte carlo
        self.state_filled_site_list = np.zeros((self.N, self.n_layer), dtype=int)   # only for n_sublat = 2
        self.state_empty_site_list = np.zeros((self.N, self.n_layer), dtype=int)    # only for n_sublat = 2
        self.state_filled_check_box = np.zeros((self.N * self.n_sublat, self.n_layer), dtype=int)
        self.log_wave_function_value = np.ones(self.n_layer)

    def initiate(self):
        self.get_state_list()
        self.state_filled_site_list = np.arange(self.N)[:, None] * np.ones(self.n_layer, dtype=int)[None, :]
        self.state_empty_site_list = np.arange(self.N, self.N * 2)[:, None] * np.ones(self.n_layer, dtype=int)[None, :]
        self.state_filled_check_box[0: self.N, :] = 1

        self.randomize_initial_state()

        for layer in range(self.n_layer):
            self.get_wave_function_matrix(layer)
            _, self.log_wave_function_value[layer] = self.get_log_wave_function(layer)
        for layer in range(self.n_layer):
            self.get_wave_function_matrix_inverse(layer)

    def update(self, i_site_0, layer):
        site_0 = self.state_filled_site_list[i_site_0, layer]
        i_site_1 = np.random.randint(self.N)
        site_1 = self.state_empty_site_list[i_site_1, layer]

        v_T = self.state_list[site_1, :, layer] - self.state_list[site_0, :, layer]
        wave_function_ratio = 1 + v_T @ self.wave_function_matrix_inverse[:, i_site_0, layer]

        # self.wave_function_matrix[i_site_0, :, layer] = self.state_list[site_1, :, layer]
        # _, new_log_wave_function_value = self.get_log_wave_function(layer)
        # wave_function_ratio = np.exp(new_log_wave_function_value - self.log_wave_function_value[layer])

        weight_ratio = (self.weight(site_1, layer, 1) * self.weight(site_0, layer, 0))
        weight_ratio = (weight_ratio * np.abs(wave_function_ratio)) ** 2
        if weight_ratio < 1 and np.random.rand() > weight_ratio:
            #self.wave_function_matrix[i_site_0, :, layer] = self.state_list[site_0, :, layer]
            return 0

        # self.log_wave_function_value[layer] = new_log_wave_function_value
        self.state_filled_site_list[i_site_0, layer] = site_1
        self.state_empty_site_list[i_site_1, layer] = site_0
        self.state_filled_check_box[site_0, layer] = 0
        self.state_filled_check_box[site_1, layer] = 1
        self.wave_function_matrix[i_site_0, :, layer] = self.state_list[site_1, :, layer]

        self.wave_function_matrix_inverse[:, :, layer] -= np.outer(self.wave_function_matrix_inverse[:, i_site_0, layer] / wave_function_ratio, v_T @ self.wave_function_matrix_inverse[:, :, layer])
        return 1

    def update_2_layer(self):
        pass

    def run(self):
        self.initiate()

        self._open_writer()
        try:
            for i_mc in range(self.n_mc):
                self.sweep()
                if i_mc % self.n_recalculate_inverse == 0:
                    for layer in range(self.n_layer):
                        self.get_wave_function_matrix_inverse(layer)

                if i_mc > self.start_to_measure and (i_mc % self.n_measure) == (self.n_measure - 1):
                    self.measure()

                if i_mc % (self.n_mc // 10) == self.n_mc // 10 - 1:
                    print("finished:", int(i_mc / self.n_mc * 100), "%")
        finally:
            self._close_writer()

    def sweep(self):
        for layer in range(self.n_layer):
            for i_site_0 in range(self.N):
                self.update(i_site_0, layer)

    def measure(self):
        # your array to append each time
        arr = self.state_filled_check_box  # shape: (self.N * self.n_sublat, self.n_layer), dtype=int
        dset = self._ensure_dataset("state_filled_check_box", arr)
        dset.resize(dset.shape[0] + 1, axis=0)
        dset[-1, ...] = arr
        self._meas_count += 1
        if self._meas_count % self._flush_every == 0:
            self._h5.flush()

        self.measurement.append(np.count_nonzero((self.state_filled_check_box[:, 0] + self.state_filled_check_box[:, 1]) == (self.state_filled_check_box[:, 2] + self.state_filled_check_box[:, 3])) / (self.N * 2))

    def plot_result(self):
        plt.plot(self.measurement)
        plt.savefig("data.png")
        print("mean:", np.mean(self.measurement))
        print("std:", np.std(self.measurement) / np.sqrt(len(self.measurement)))

    def weight(self, site, layer, add_or_delete):
        # 1 is to add and 0 is to delete
        state_filled = self.state_filled_check_box[site, :].copy()

        # particle number the same
        weight_0 = (state_filled[0] + state_filled[1]) != (state_filled[2] + state_filled[3])
        state_filled[layer] = add_or_delete
        weight_1 = (state_filled[0] + state_filled[1]) != (state_filled[2] + state_filled[3])

        # spin the same
        # weight_0 = (state_filled[0] - state_filled[1]) != (state_filled[2] - state_filled[3])
        # state_filled[layer] = add_or_delete
        # weight_1 = (state_filled[0] - state_filled[1]) != (state_filled[2] - state_filled[3])

        return self.p ** (weight_1.astype(int) - weight_0.astype(int))

    def get_state_list(self):
        #have 0 denomenator problem if enters trivial phase
        for ix in range(self.L):
            for iy in range(ix, self.L):
                k1 = 2 * math.pi * ix / self.L
                k2 = 2 * math.pi * iy / self.L
                energy, state = self.get_state_small_k1((k1, k2))
                self.state_list_k[ix, iy, :] = state

        for ix in range(self.L):
            for iy in range(0, ix):
                k1 = 2 * math.pi * ix / self.L
                k2 = 2 * math.pi * iy / self.L
                if self.m > self.t2 * math.sqrt(3) * 3 * math.sin(self.phi):
                    energy, state = self.get_state_small_k1((k1, k2))
                else:
                    energy, state = self.get_state_small_k2((k1, k2))
                self.state_list_k[ix, iy, :] = state

        for i_sublat in range(self.n_sublat):
            '''
            (0, 0), (0, 1), (0, 2), (0, 3), ... , (1, 0), (1, 1), (1, 2), (1, 3), ... for layer 0, 3
            -(0, 0), -(0, 1), ... for layer 1, 2
            '''
            u_j_0 = self.state_list_k[:, :, i_sublat]
            u_j = u_j_0.reshape(self.N)
            x_i = (np.array(range(self.L))[:, None] * np.ones(self.L)[None, :]).reshape(self.N) + self.sublat_position[i_sublat][0]
            y_i = (np.ones(self.L)[:, None] * np.array(range(self.L))[None, :]).reshape(self.N) + self.sublat_position[i_sublat][1]
            kx_j = (np.array(range(self.L))[:, None] * np.ones(self.L)[None, :]).reshape(self.N) * 2 * np.pi / self.L
            ky_j = (np.ones(self.L)[:, None] * np.array(range(self.L))[None, :]).reshape(self.N) * 2 * np.pi / self.L
            self.state_list[self.N * i_sublat: self.N * (i_sublat + 1), :, 0] = u_j[None, :] * np.exp(1j * (x_i[:, None] * kx_j[None, :] + y_i[:, None] * ky_j[None, :])) / math.sqrt(self.N)

        self.state_list[:, :, 1] = self.state_list[:, :, 0].conj()  # time reversal
        self.state_list[:, :, 2] = self.state_list[:, :, 0].conj()  # bra vector
        self.state_list[:, :, 3] = self.state_list[:, :, 0]         # bra vector + time reversal

    def get_wave_function_matrix(self, layer):
        self.wave_function_matrix[:, :, layer] = self.state_list[self.state_filled_site_list[:, layer], :, layer]

    def get_wave_function_matrix_inverse(self, layer):
        #start = time.time()
        inv_matrix = np.linalg.inv(self.wave_function_matrix[:, :, layer])
        #end = time.time()
        # print("time_inv:", end-start)
        # error = np.sum(np.abs(self.wave_function_matrix_inverse[:, :, layer] - inv_matrix))
        # print("error:", error)
        self.wave_function_matrix_inverse[:, :, layer] = inv_matrix

    def get_log_wave_function(self, layer):
        log_wave_function = np.linalg.slogdet(self.wave_function_matrix[:, :, layer])
        return log_wave_function

    def randomize_initial_state(self):
        for layer in range(self.n_layer):
            for i_site_0 in range(self.N):
                site_0 = self.state_filled_site_list[i_site_0, layer]
                i_site_1 = np.random.randint(self.N)
                site_1 = self.state_empty_site_list[i_site_1, layer]

                weight_ratio = (self.weight(site_1, layer, 1) * self.weight(site_0, layer, 0)) ** 2
                if np.random.rand() < weight_ratio:
                    self.state_filled_site_list[i_site_0, layer] = site_1
                    self.state_empty_site_list[i_site_1, layer] = site_0
                    self.state_filled_check_box[site_0, layer] = 0
                    self.state_filled_check_box[site_1, layer] = 1

    def recover_data_from_check_box(self):
        self.get_state_list()
        rows_grid = np.arange(self.N * self.n_sublat)
        for layer in range(self.n_layer):
            self.state_filled_site_list[:, layer] = rows_grid[self.state_filled_check_box[:, layer] == 1]
            self.state_empty_site_list[:, layer] = rows_grid[self.state_filled_check_box[:, layer] == 0]

        for layer in range(self.n_layer):
            self.get_wave_function_matrix(layer)
            self.get_wave_function_matrix_inverse(layer)

    def get_hamiltonian(self, k):
        k1, k2 = k
        h_xy = self.t1 * (cmath.exp(1j * (-k1 + 2 * k2) / 3) + cmath.exp(1j * (2 * k1 - k2) / 3) + cmath.exp(1j * (- k1 - k2) / 3))
        h_z = self.m + 2 * self.t2 * math.sin(self.phi) * (math.sin(k1) + math.sin(-k2) + math.sin(-k1 + k2))

        hamiltonian_k = np.array([[h_z, h_xy], [h_xy.conjugate(), -h_z]])
        return hamiltonian_k

    def get_state_small_k1(self, k):
        k1, k2 = k
        h_xy = self.t1 * (cmath.exp(1j * (-k1 + 2 * k2) / 3) + cmath.exp(1j * (2 * k1 - k2) / 3) + cmath.exp(1j * (- k1 - k2) / 3))
        h_z = self.m + 2 * self.t2 * math.sin(self.phi) * (math.sin(k1) + math.sin(-k2) + math.sin(-k1 + k2))
        h = math.sqrt(h_z ** 2 + abs(h_xy) ** 2)
        A = math.sqrt(2 * h * (h + h_z))
        state = np.array([-h_xy, h_z + h]) / A
        return h, state

    def get_state_small_k2(self, k):
        k1, k2 = k
        h_xy = self.t1 * (cmath.exp(1j * (-k1 + 2 * k2) / 3) + cmath.exp(1j * (2 * k1 - k2) / 3) + cmath.exp(1j * (- k1 - k2) / 3))
        h_z = self.m + 2 * self.t2 * math.sin(self.phi) * (math.sin(k1) + math.sin(-k2) + math.sin(-k1 + k2))
        h = math.sqrt(h_z ** 2 + abs(h_xy) ** 2)
        A = math.sqrt(2 * h * (h - h_z))

        state = np.array([h_z - h, h_xy.conjugate()]) / A
        return h, state

    def check_plot_band(self):
        band_energy = np.zeros((self.L, self.L))
        for ix in range(self.L):
            for iy in range(ix, self.L):
                k1 = 2 * math.pi * ix / self.L
                k2 = 2 * math.pi * iy / self.L
                energy, state = self.get_state_small_k1((k1, k2))
                band_energy[ix, iy] = energy

        for ix in range(self.L):
            for iy in range(0, ix):
                k1 = 2 * math.pi * ix / self.L
                k2 = 2 * math.pi * iy / self.L
                energy, state = self.get_state_small_k2((k1, k2))
                band_energy[ix, iy] = energy

        kx = np.linspace(0, 2 * np.pi, self.L)
        ky = np.linspace(0, 2 * np.pi, self.L)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(kx, ky, band_energy, cmap='viridis', alpha=0.8)
        ax.view_init(elev=0, azim=90)
        plt.savefig('band.png')

    def check_hamiltonian_state(self):
        for ix in range(self.L):
            for iy in range(self.L):
                if ix >= iy:
        # for (ix, iy) in [(3, 6), (6, 3)]:
                    k1 = 2 * math.pi * ix / self.L
                    k2 = 2 * math.pi * iy / self.L
                    k = (k1, k2)
                    # print("k:", k)
                    _, state = self.get_state_small_k2(k)
                    hamiltonian = self.get_hamiltonian(k)
                    state_acted = hamiltonian @ state
                    # print("state_acted:", state_acted)
                    # print("norm:", np.linalg.norm(state_acted))
                    # print("state_acted:", state_acted / np.linalg.norm(state_acted))
                    overlap = state.conj() @ state_acted / np.linalg.norm(state_acted)
                    print("overlap:", abs(overlap) - 1)

    def _open_writer(self):
        Path(self.data_file).mkdir(parents=True, exist_ok=True)
        self._h5 = h5py.File(Path(self.data_file) / "configs.h5", "a")  # single-writer per chain
        self._dsets = {}
        self._meas_count = 0
        self._flush_every = 1000  # tune this (e.g., 100~5000)

    def _ensure_dataset(self, name, arr):
        """Create dataset lazily on first use; append along time axis."""
        if name not in self._dsets:
            # small heuristic for chunking; safe default
            chunk_rows = max(1, min(4096, max(1, self.n_mc // max(1, self.n_measure))))
            dset = self._h5.require_dataset(
                name,
                shape=(0,) + arr.shape,
                maxshape=(None,) + arr.shape,
                chunks=(chunk_rows,) + arr.shape,
                dtype=arr.dtype,
                compression="lzf",
            )
            self._dsets[name] = dset
        return self._dsets[name]

    def _close_writer(self):
        if getattr(self, "_h5", None) is not None:
            self._h5.flush()
            self._h5.close()
            self._h5 = None
            self._dsets = {}