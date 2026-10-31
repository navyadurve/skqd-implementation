# skqd-implementation

Implementation of the Sample-based Krylov Quantum Diagonalization (SKQD) 
algorithm, following IBM Quantum's tutorial applying it to the 
Single-Impurity Anderson Model (SIAM).

## Reference
- Tutorial: [Sample-based Krylov quantum diagonalization of a fermionic lattice model](https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-krylov-quantum-diagonalization)
- Paper: [Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization](https://arxiv.org/abs/2501.09702)

## Environment
Python 3.12, Qiskit SDK, qiskit-ibm-runtime, qiskit-addon-sqd, ffsim, 
PySCF. Full list in `requirements.txt`.

## Files
| File | What it is |
|---|---|
| `kqd_simple.ipynb` | Warm-up: standalone Krylov quantum diagonalization (KQD) on a 10-qubit Heisenberg chain |
| `skqd_siam_small.ipynb` / `.py` | SKQD, small-scale SIAM (8 orbitals, 16 qubits) on a statevector simulator, checked against exact FCI |
| `skqd_siam_hardware.ipynb` / `.py` | SKQD, large-scale SIAM (20 orbitals, 40 qubits) on real IBM quantum hardware, checked against the tutorial's DMRG energy |

## Running the hardware example
1. Save your IBM Quantum credentials once (never commit the API key):
   ```python
   from qiskit_ibm_runtime import QiskitRuntimeService
   QiskitRuntimeService.save_account(
       token="<your-api-key>",
       instance="<your-instance-CRN>",
       set_as_default=True,
       overwrite=True,
   )
   ```
   Credentials are stored in `~/.qiskit/qiskit-ibm.json`, outside this repo.
2. Pick a mode at the top of the notebook (or set `SKQD_MODE`):
   - `dry_run`: no account needed; transpiles for a fake 156-qubit Heron backend and prints circuit depth / two-qubit gate counts
   - `hardware`: submits to the least-busy real QPU (tutorial estimate about 9 s of QPU time)
   - `retrieve`: fetches an already-submitted job by `SKQD_JOB_ID`
   - `saved`: re-runs SQD on the saved `skqd_hw_counts.json`, with no QPU
3. Outputs: `skqd_hw_job.json`, `skqd_hw_counts.json`, `skqd_hw_results.json` and `skqd_hw_results.png`.

## Status
- [x] Environment set up
- [x] KQD warm-up implementation
- [x] SKQD small-scale simulator example (error 4.3e-4 vs exact FCI)
- [x] SKQD hardware script written and dry-run tested (transpilation + SQD post-processing)
- [x] SKQD 20-orbital run on a real IBM QPU (ibm_fez, error 1.09e-2 vs DMRG)
- [ ] Parameter study (shots, Krylov dimension, time step)
