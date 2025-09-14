import os
for v in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS","MKL_DYNAMIC","OPENBLAS_DYNAMIC"]:
    os.environ.setdefault(v, "1" if "DYNAMIC" not in v else "0")

import numpy as np
from SQHE import SQHE
from ED_SQHE_check import ED_SQHE
import time
from scipy.sparse import csc_matrix, csr_matrix
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import shelve
from pathlib import Path
from threadpoolctl import threadpool_limits


def _run_one_chain(parameters, seed):
    with threadpool_limits(1):   # ← limit BLAS/OpenMP to 1 thread in this worker
        np.random.seed(seed)
        model = SQHE(parameters)
        model.run()
        return model.measurement


def run_parallel(parameters, num_chains=8, master_seed=12345):
    """
    Launch 'num_chains' independent MC chains in parallel.
    Returns: concatenated list of all measurements (same type as your per-chain entries).
    """
    # Reproducible per-chain seeds
    ss = np.random.SeedSequence(master_seed)
    chain_seeds = ss.spawn(num_chains)
    seeds = [int(s.generate_state(1)[0]) for s in chain_seeds]

    # Fan out work
    results = []
    with ProcessPoolExecutor(max_workers=num_chains) as ex:
        futures = [ex.submit(_run_one_chain, {**parameters, "data_file": f'{parameters["data_file"]}/chain_{i}'}, seeds[i]) for i in range(num_chains)]
        for f in as_completed(futures):
            results.append(f.result())  # each is a list

    # Flatten lists
    all_measurements = []
    for lst in results:
        all_measurements.extend(lst)

    return all_measurements


def mc_run_parallel():
    mp.set_start_method("spawn", force=True)  # robust on Windows/WSL

    parameters = {'t1': 1, 't2': 0.3, 'm': 0, 'phi': np.pi/2,
                  'p': 0, 'L': 20, 'n_mc': 40_000, 'n_measure': 10, 'num_chains': 5, 'data_file': "data/data_set_10"}

    Path(parameters['data_file']).mkdir(parents=True, exist_ok=True)
    with shelve.open(parameters['data_file'] + "/parameters") as db:
        db['parameters'] = parameters

    start = time.time()
    measurements = run_parallel(parameters, num_chains=parameters['num_chains'], master_seed=2025)
    end = time.time()
    print("time:", end - start)

    # (Optional) quick stats
    arr = np.asarray(measurements)
    print(f"#samples = {arr.size}, mean = {arr.mean():.6g}, stderr = {arr.std(ddof=1)/np.sqrt(arr.size):.6g}")


def mc_run_single_thread():
    np.set_printoptions(linewidth=np.inf)
    np.random.seed(0)
    parameters = {'t1': 1, 't2': 0.3, 'm': 0, 'phi': np.pi/2, 'p': 0, 'L': 2, 'n_mc': 10000, 'n_measure': 10, 'num_chains': 8, 'data_file': "data/data_set_9"}

    my_model = SQHE(parameters)
    start = time.time()
    my_model.run()
    end = time.time()
    print(f'Run time: {end - start}')


def ed_check_mc():
    parameters = {'t1': 1, 't2': 0.3, 'm': 0, 'phi': np.pi / 2, 'p': 0, 'L': 2, 'n_mc': 10000, 'n_measure': 10, 'num_chains': 8, 'data_file': "data/data_set_9"}
    my_model = ED_SQHE(parameters)
    # my_model.get_many_body_hamiltonian()
    # my_model.get_many_body_ground_state()
    # my_model.compare_many_body_wave_function()
    my_model.get_bilayer_state()


# if __name__ == "__main__":
#     mc_run_parallel()
#     # mc_run_single_thread()
#     # ed_check_mc()

