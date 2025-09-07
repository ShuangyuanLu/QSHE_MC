from SQHE import SQHE
import numpy as np
import scipy
from SQHE import SQHE
from Fermion import fermion_ops, fermion_dense_ops
from scipy.sparse import csr_matrix, csc_matrix, kron
from my_functions import sparse_unequal_nonzero_mask


class ED_SQHE:
    def __init__(self, parameters):
        self.t1 = parameters['t1']
        self.t2 = parameters['t2']
        self.m = parameters['m']
        self.phi = parameters['phi']
        self.p = parameters['p']
        self.L = parameters['L']
        self.N = self.L ** 2

        self.c = None
        self.cd = None
        self.n = None
        self.state = None
        self.bilayer_state = None
        self.density_matrix = None
        self.correlator = None

        self.n_state_1 = None
        self.n_state_2 = None
        self.n_bilayer_state = None

        self.hamiltonian = None
        self.V = None
        self.sqhe = SQHE(parameters)

        self.get_hamiltonian()
        self.get_eigenstates()

        self.constant = 10


    def get_hamiltonian(self):
        self.hamiltonian = np.zeros((self.N * 2, self.N * 2), dtype=np.complex128)
        for i in range(self.L):
            for j in range(self.L):
                n = i * self.L + j
                self.hamiltonian[n, n] += self.m
                self.hamiltonian[n + self.N, n + self.N] += -self.m

                self.hamiltonian[n, n + self.N] += self.t1
                self.hamiltonian[n + self.N, n] += self.t1
                n1 = ((i - 1) % self.L) * self.L + (j + 1) % self.L
                self.hamiltonian[n, n1 + self.N] += self.t1
                self.hamiltonian[n1 + self.N, n] += self.t1
                n2 = ((i - 1) % self.L) * self.L + j
                self.hamiltonian[n, n2 + self.N] += self.t1
                self.hamiltonian[n2 + self.N, n] += self.t1

                n1 = i * self.L + (j + 1) % self.L
                self.hamiltonian[n, n1] += self.t2 * np.exp(1j * self.phi)
                self.hamiltonian[n1, n] += self.t2 * np.exp(-1j * self.phi)
                self.hamiltonian[n + self.N, n1 + self.N] += self.t2 * np.exp(-1j * self.phi)
                self.hamiltonian[n1 + self.N, n + self.N] += self.t2 * np.exp(1j * self.phi)
                n2 = ((i + 1) % self.L) * self.L + j
                self.hamiltonian[n, n2] += self.t2 * np.exp(-1j * self.phi)
                self.hamiltonian[n2, n] += self.t2 * np.exp(1j * self.phi)
                self.hamiltonian[n + self.N, n2 + self.N] += self.t2 * np.exp(1j * self.phi)
                self.hamiltonian[n2 + self.N, n + self.N] += self.t2 * np.exp(-1j * self.phi)
                n3 = ((i - 1) % self.L) * self.L + (j + 1) % self.L
                self.hamiltonian[n, n3] += self.t2 * np.exp(-1j * self.phi)
                self.hamiltonian[n3, n] += self.t2 * np.exp(1j * self.phi)
                self.hamiltonian[n + self.N, n3 + self.N] += self.t2 * np.exp(1j * self.phi)
                self.hamiltonian[n3 + self.N, n + self.N] += self.t2 * np.exp(-1j * self.phi)

    def get_eigenstates(self):
        D, V = scipy.linalg.eigh(self.hamiltonian)
        self.V = V[:, :self.N]
        # print(D)
        # print(np.sum(D[:self.N]))

    def get_many_body_hamiltonian(self):
        self.c, self.cd, self.n, _ = fermion_dense_ops(self.N * 2)

        self.hamiltonian = np.zeros((2 ** (self.N * 2), 2 ** (self.N * 2)), dtype=np.complex128)
        for i in range(self.L):
            for j in range(self.L):
                n = i * self.L + j
                self.hamiltonian += self.m * self.cd[n] @ self.c[n]
                self.hamiltonian += (-self.m) * (self.cd[n + self.N] @ self.c[n + self.N])

                self.hamiltonian += self.t1 * (self.cd[n] @ self.c[n + self.N])
                self.hamiltonian += self.t1 * (self.cd[n + self.N] @ self.c[n])
                n1 = ((i - 1) % self.L) * self.L + (j + 1) % self.L
                self.hamiltonian += self.t1 * (self.cd[n] @ self.c[n1 + self.N])
                self.hamiltonian += self.t1 * (self.cd[n1 + self.N] @ self.c[n])
                n2 = ((i - 1) % self.L) * self.L + j
                self.hamiltonian += self.t1 * (self.cd[n] @ self.c[n2 + self.N])
                self.hamiltonian += self.t1 * (self.cd[n2 + self.N] @ self.c[n])


                n1 = i * self.L + (j + 1) % self.L
                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[n] @ self.c[n1])
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[n1] @ self.c[n])

                n2 = ((i + 1) % self.L) * self.L + j
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[n] @ self.c[n2])
                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[n2] @ self.c[n])

                n3 = ((i - 1) % self.L) * self.L + (j + 1) % self.L
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[n] @ self.c[n3])
                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[n3] @ self.c[n])

                n1B = n1 + self.N
                n2B = n2 + self.N
                n3B = n3 + self.N
                nB = n + self.N
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[nB] @ self.c[n1B])
                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[n1B] @ self.c[nB])

                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[nB] @ self.c[n2B])
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[n2B] @ self.c[nB])

                self.hamiltonian += (self.t2 * np.exp(1j * self.phi)) * (self.cd[nB] @ self.c[n3B])
                self.hamiltonian += (self.t2 * np.exp(-1j * self.phi)) * (self.cd[n3B] @ self.c[nB])

    def get_many_body_ground_state(self):
        # print(self.hamiltonian.shape)
        D, V = scipy.linalg.eigh(self.hamiltonian)
        self.state = V[:, 0]
        # print(D[0])

    def get_bilayer_state(self):
        self.get_many_body_hamiltonian()
        self.get_many_body_ground_state()
        state_1 = self.state.copy()

        self.phi = -self.phi
        self.get_many_body_hamiltonian()
        self.get_many_body_ground_state()
        state_2 = self.state.copy()

        bilayer_state = np.kron(state_1, state_2)

        rtol = 1e-10
        thr = rtol * np.max(np.abs(bilayer_state))
        idx = np.abs(bilayer_state) < thr
        bilayer_state[idx] = 0
        bilayer_state = csc_matrix(bilayer_state[:, None])
        self.bilayer_state = bilayer_state
        self.density_matrix = kron(bilayer_state, bilayer_state.conj(), format='csc')

        # for correlator
        site_0, site_1 = 0, 1
        # \psi \psi_dagger
        state_1_corr = self.c[site_0] @ (self.cd[site_1] @ state_1)
        state_2_corr = self.c[site_1] @ (self.cd[site_0] @ state_2)
        bilayer_state_corr = np.kron(state_1_corr, state_2_corr)
        # Sz
        # bilayer_state_corr = np.kron(self.n[site_0] @ (self.n[site_1] @ state_1), state_2) + np.kron(state_1, self.n[site_0] @ (self.n[site_1] @ state_2)) \
        #                      - np.kron(self.n[site_0] @ state_1, self.n[site_1] @ state_2) - np.kron(self.n[site_1] @ state_1, self.n[site_0] @ state_2)

        thr_corr = rtol * np.max(np.abs(bilayer_state_corr))
        idx_corr = np.abs(bilayer_state_corr) < thr_corr
        bilayer_state_corr[idx_corr] = 0
        bilayer_state_corr = csc_matrix(bilayer_state_corr[:, None])
        self.correlator = kron(bilayer_state_corr, bilayer_state_corr.conj(), format='csc')




        for site in range(self.N * 2):

            self.n_state_1 = np.diag(self.n[site])
            self.n_state_1 = np.round(np.real(self.n_state_1)).astype(int)
            self.n_state_2 = self.n_state_1
            identity = np.ones(self.n_state_1.size, dtype=int)

            self.n_bilayer_state = np.kron(self.n_state_1, identity) + np.kron(identity, self.n_state_2) + self.constant
            self.n_bilayer_state[idx] = 0
            self.n_bilayer_state = csc_matrix(self.n_bilayer_state[:, None])

            identity_2 = np.ones(self.n_state_1.size * self.n_state_2.size, dtype=int)
            identity_2[idx] = 0
            identity_2 = csc_matrix(identity_2[:, None])

            n_A = kron(self.n_bilayer_state, identity_2, format='csc')
            n_B = kron(identity_2, self.n_bilayer_state, format='csc')

            mask = sparse_unequal_nonzero_mask(n_A, n_B)
            self.density_matrix -= self.density_matrix.multiply(mask) * (1-self.p)
            self.correlator -= self.correlator.multiply(mask) * (1-self.p)


        weight = self.density_matrix.multiply(self.density_matrix.conj()).sum()

        corr = self.density_matrix.multiply(self.correlator.conj()).sum()

        print("weight:", weight)
        print("corr:", corr)
        print("corr/weight:", corr/weight)

        # measurement = 0
        # for site in range(self.N * 2):
        #     self.n_state_1 = np.diag(self.n[site])
        #     self.n_state_1 = np.round(np.real(self.n_state_1)).astype(int)
        #     self.n_state_2 = self.n_state_1
        #     identity = np.ones(self.n_state_1.size, dtype=int)
        #
        #     self.n_bilayer_state = np.kron(self.n_state_1, identity) + np.kron(identity, self.n_state_2) + self.constant
        #     self.n_bilayer_state[idx] = 0
        #     self.n_bilayer_state = csc_matrix(self.n_bilayer_state[:, None])
        #
        #     identity_2 = np.ones(self.n_state_1.size * self.n_state_2.size, dtype=int)
        #     identity_2[idx] = 0
        #     identity_2 = csc_matrix(identity_2[:, None])
        #
        #     n_A = kron(self.n_bilayer_state, identity_2, format='csc')
        #     n_B = kron(identity_2, self.n_bilayer_state, format='csc')
        #     mask = sparse_unequal_nonzero_mask(n_A, n_B)
        #     measurement += self.density_matrix.multiply(mask).multiply(self.density_matrix.conj()).sum()
        #
        # measurement /= weight * self.N * 2
        # measurement = 1 - measurement
        # print("weight:", weight)
        # print("measurement:", measurement)


    def compare_eigenstates(self):
        V1 = self.sqhe.state_list[:, :, 0]
        V1_acted = self.hamiltonian @ V1
        # check orthogonality
        print(V1.T.conj() @ V1)

        for i in range(self.N):
            state = V1[:, i]
            state_normalized = V1_acted[:, i] / np.linalg.norm(V1_acted[:, i])
            overlap = state_normalized.T.conj() @ state
            energy = state.T.conj() @ self.hamiltonian @ state
            print(np.round(energy, 5), np.round(np.abs(overlap), 5), np.linalg.norm(V1_acted[:, i]))

    def compare_many_body_wave_function(self):
        site_list = [0, 2, 4, 5]
        single_body_psi = np.linalg.det(self.V[site_list, :])
        many_body_psi = self.state[np.sum(np.pow(2, site_list))]
        print(single_body_psi)
        print(many_body_psi)



















