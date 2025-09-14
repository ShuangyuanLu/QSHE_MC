# Quantum Spin Hall States with Coherence
This Python script calculates a mixed state with a power-law decay of the Rényi-2 correlator.
Starting from a Quantum Spin Hall (QSH) state (two layers of the Haldane model with Chern numbers +1 and -1),
we apply a quantum channel that fixes the particle number. 
The spin creation correlator exhibits power-law decay under these conditions.

We use Fermion Monte Carlo (FMC) simulations. Similar to Variational Monte Carlo, local observables (such as the correlator)
are computed, but only for a fixed free-fermion state (without optimization).

---

## Usage

This project provides three main functions in `main.py`.  
Uncomment the desired line in the `if __name__ == "__main__":` block to run it.

### 1. `mc_run_parallel()`
- Runs the Monte Carlo simulation using parallel processing. 
- The MC electron configurations are stored in data files. 
- We need to set up parameters and for `p=0` case, we use `sweep_for_update_2_layers()` in the method `run()` of class `SQHE`.

```bach
python main.py
```
### 2. `mc_run_single_thread()`
- Runs the Monte Carlo simulation in single-thread mode.

### 3. `ed_check_mc()`
- Performs exact diagonalization (ED) checks on Monte Carlo results.

### 4. Data analysis
- To analyze the MC data (calculating the correlator), set up `data_file` and `correlator_type`, then run

```bach
python main_data_analysis.py
```


