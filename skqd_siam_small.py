# %% [markdown]
# # Sample-based Krylov Quantum Diagonalization (SKQD) — small-scale SIAM example
#
# Follows IBM Quantum's tutorial "Sample-based Krylov quantum diagonalization of a
# fermionic lattice model" (small-scale simulator part):
# https://quantum.cloud.ibm.com/docs/en/tutorials/sample-based-krylov-quantum-diagonalization
#
# Four-step Qiskit pattern:
#   1. Map the problem to circuits   (SIAM Hamiltonian -> momentum basis -> Trotter circuits)
#   2. Optimize for execution        (transpile to a generic backend)
#   3. Execute with primitives       (StatevectorSampler, shots from every circuit concatenated)
#   4. Post-process                  (SQD: configuration recovery + classical diagonalization)

# %% [markdown]
# ## Step 1: Map the problem to quantum circuits

# %%
from typing import Sequence

import ffsim
import matplotlib.pyplot as plt
import numpy as np
import pyscf.fci
import scipy.linalg
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import CircuitInstruction, Qubit
from qiskit.circuit.library import CPhaseGate, XGate, XXPlusYYGate


# %% [markdown]
# ### SIAM Hamiltonian (position basis), and the rotation to the momentum basis

# %%
def siam_hamiltonian(
    norb: int,
    hopping: float,
    onsite: float,
    hybridization: float,
    chemical_potential: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hamiltonian (h1e, h2e) for the single-impurity Anderson model."""
    impurity_orb = 0  # impurity sits on the first site in the position basis

    # One-body part: bath hopping, impurity-bath hybridization, chemical potential
    h1e = np.zeros((norb, norb))
    np.fill_diagonal(h1e[:, 1:], -hopping)
    np.fill_diagonal(h1e[1:, :], -hopping)
    h1e[impurity_orb, impurity_orb + 1] = -hybridization
    h1e[impurity_orb + 1, impurity_orb] = -hybridization
    h1e[impurity_orb, impurity_orb] = chemical_potential

    # Two-body part: on-site interaction U on the impurity only
    h2e = np.zeros((norb, norb, norb, norb))
    h2e[impurity_orb, impurity_orb, impurity_orb, impurity_orb] = onsite

    return h1e, h2e


def momentum_basis(norb: int) -> np.ndarray:
    """Orbital rotation from the position basis to the momentum basis."""
    n_bath = norb - 1

    # Diagonalize the (non-interacting) bath hopping matrix
    hopping_matrix = np.zeros((n_bath, n_bath))
    np.fill_diagonal(hopping_matrix[:, 1:], -1)
    np.fill_diagonal(hopping_matrix[1:, :], -1)
    _, vecs = np.linalg.eigh(hopping_matrix)

    # Extend to include the impurity (unchanged, on the first site)
    orbital_rotation = np.zeros((norb, norb))
    orbital_rotation[0, 0] = 1
    orbital_rotation[1:, 1:] = vecs

    # Move the impurity to the centre of the orbital ordering
    new_index = n_bath // 2
    perm = np.r_[1 : (new_index + 1), 0, (new_index + 1) : norb]
    return orbital_rotation[:, perm]


def rotated(
    h1e: np.ndarray, h2e: np.ndarray, orbital_rotation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the orbital basis of a Hamiltonian."""
    h1e_rotated = np.einsum(
        "ab,Aa,Bb->AB", h1e, orbital_rotation, orbital_rotation.conj(),
        optimize="greedy",
    )
    h2e_rotated = np.einsum(
        "abcd,Aa,Bb,Cc,Dd->ABCD", h2e,
        orbital_rotation, orbital_rotation.conj(),
        orbital_rotation, orbital_rotation.conj(),
        optimize="greedy",
    )
    return h1e_rotated, h2e_rotated


# %%
# Total spatial orbitals (impurity + bath); must be even and >= 8
norb = 8
nelec = (norb // 2, norb // 2)  # half filling
n_bath = norb - 1

# Hamiltonian parameters
hybridization = 1.0
hopping = 1.0
onsite = 10.0
chemical_potential = -0.5 * onsite

h1e, h2e = siam_hamiltonian(
    norb=norb,
    hopping=hopping,
    onsite=onsite,
    hybridization=hybridization,
    chemical_potential=chemical_potential,
)

# Rotate to the momentum basis (where the ground state is sparser)
orbital_rotation = momentum_basis(norb)
h1e_momentum, h2e_momentum = rotated(h1e, h2e, orbital_rotation.T.conj())
impurity_index = n_bath // 2  # impurity position after the rotation

# Exact (full CI) ground-state energy as the reference
reference_energy, _ = pyscf.fci.direct_spin1.kernel(h1e, h2e, norb, nelec)
print(f"Exact (FCI) reference energy: {reference_energy:.5f}")


# %% [markdown]
# ### Krylov circuits: initial state + repeated second-order Trotter steps
#
# |psi_k> ~ [ e^{-i dt/2 H2} e^{-i dt H1} e^{-i dt/2 H2} ]^k |psi_0>
#
# H2 (impurity U term) -> one CPhase gate; H1 (quadratic) -> an orbital rotation.

# %%
def prepare_initial_state(qubits: Sequence[Qubit], norb: int, nocc: int):
    """Superposition of excitations near the Fermi level, per spin species."""
    assert norb >= 8
    x_gate = XGate()
    rot = XXPlusYYGate(0.5 * np.pi, -0.5 * np.pi)
    for i in range(nocc):
        yield CircuitInstruction(x_gate, [qubits[i]])
        yield CircuitInstruction(x_gate, [qubits[norb + i]])
    for i in range(3):
        for j in range(nocc - i - 1, nocc + i, 2):
            yield CircuitInstruction(rot, [qubits[j], qubits[j + 1]])
            yield CircuitInstruction(rot, [qubits[norb + j], qubits[norb + j + 1]])
    yield CircuitInstruction(rot, [qubits[j + 1], qubits[j + 2]])
    yield CircuitInstruction(rot, [qubits[norb + j + 1], qubits[norb + j + 2]])


def trotter_step(
    qubits: Sequence[Qubit],
    time_step: float,
    one_body_evolution: np.ndarray,
    h2e: np.ndarray,
    impurity_index: int,
    norb: int,
):
    """One second-order Trotter step."""
    onsite = h2e[impurity_index, impurity_index, impurity_index, impurity_index]
    # Two-body evolution for half the time step
    yield CircuitInstruction(
        CPhaseGate(-0.5 * time_step * onsite),
        [qubits[impurity_index], qubits[norb + impurity_index]],
    )
    # One-body evolution for the full time step
    yield CircuitInstruction(
        ffsim.qiskit.OrbitalRotationJW(norb, one_body_evolution), qubits
    )
    # Two-body evolution for the other half
    yield CircuitInstruction(
        CPhaseGate(-0.5 * time_step * onsite),
        [qubits[impurity_index], qubits[norb + impurity_index]],
    )


# %%
time_step = 0.2   # Trotter step dt
krylov_dim = 8    # number of Krylov states = number of circuits

qubits = QuantumRegister(2 * norb, name="q")
circuit = QuantumCircuit(qubits)

for instruction in prepare_initial_state(qubits, norb=norb, nocc=norb // 2):
    circuit.append(instruction)
circuit.measure_all()

# circuits[k] prepares |psi_k>; circuits[0] is the initial state alone
circuits = [circuit.copy()]

one_body_evolution = scipy.linalg.expm(-1j * time_step * h1e_momentum)
for i in range(krylov_dim - 1):
    circuit.remove_final_measurements()
    for instruction in trotter_step(
        qubits, time_step, one_body_evolution, h2e_momentum, impurity_index, norb
    ):
        circuit.append(instruction)
    circuit.measure_all()
    circuits.append(circuit.copy())

print(f"Built {len(circuits)} Krylov circuits on {circuits[0].num_qubits} qubits")

# %% [markdown]
# ## Step 2: Optimize for execution

# %%
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import generate_preset_pass_manager

backend = GenericBackendV2(2 * norb, basis_gates=["cp", "xx_plus_yy", "p", "x"])
pass_manager = generate_preset_pass_manager(optimization_level=3, backend=backend)
isa_circuits = pass_manager.run(circuits)

# %% [markdown]
# ## Step 3: Execute with Qiskit primitives
#
# Sample every Krylov circuit, then **concatenate all shots into a single set of
# bitstrings** — this combined set is what SQD post-processes.

# %%
from qiskit.primitives import BitArray, StatevectorSampler
from qiskit.visualization import plot_histogram

# (seed added for reproducibility; the IBM tutorial samples unseeded)
sampler = StatevectorSampler(seed=42)
job = sampler.run(isa_circuits, shots=500)

bit_array = BitArray.concatenate_shots([res.data.meas for res in job.result()])
print(f"Total shots after concatenation: {bit_array.num_shots}")

fig_hist = plot_histogram(bit_array.get_counts(), number_to_keep=20)
fig_hist.savefig("skqd_small_histogram.png", bbox_inches="tight")

# %% [markdown]
# ## Step 4: Post-process with SQD (configuration recovery + diagonalization)

# %%
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian

result_history = []


def callback(results: list[SCIResult]):
    result_history.append(results)
    print(f"Iteration {len(result_history)}")
    for i, r in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {r.energy}")
        print(f"\t\tSubspace dimension: {np.prod(r.sci_state.amplitudes.shape)}")


rng = np.random.default_rng(24)
result = diagonalize_fermionic_hamiltonian(
    h1e_momentum,
    h2e_momentum,
    bit_array,
    samples_per_batch=100,
    norb=norb,
    nelec=nelec,
    num_batches=3,
    max_iterations=5,
    symmetrize_spin=True,
    callback=callback,
    seed=rng,
)

# %% [markdown]
# ## Results

# %%
min_es = [min(r, key=lambda res: res.energy).energy for r in result_history]
min_id, min_e = min(enumerate(min_es), key=lambda x: x[1])

x1 = range(len(result_history))
y2 = np.sum(result.orbital_occupancies, axis=0)
x2 = range(len(y2))

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

axs[0].plot(x1, min_es, label="energy", marker="o")
axs[0].set_xticks(x1)
axs[0].set_xticklabels(x1)
axs[0].axhline(y=reference_energy, color="#BF5700", linestyle="--", label="reference energy")
axs[0].set_title("Approximated Ground State Energy vs SQD Iterations")
axs[0].set_xlabel("Iteration Index", fontdict={"fontsize": 12})
axs[0].set_ylabel("Energy", fontdict={"fontsize": 12})
axs[0].legend()

axs[1].bar(x2, y2, width=0.8)
axs[1].set_xticks(x2)
axs[1].set_xticklabels(x2)
axs[1].set_title("Avg Occupancy per Spatial Orbital")
axs[1].set_xlabel("Orbital Index", fontdict={"fontsize": 12})
axs[1].set_ylabel("Avg Occupancy", fontdict={"fontsize": 12})

print(f"Reference energy: {reference_energy:.5f}")
print(f"SQD energy: {min_e:.5f}")
print(f"Absolute error: {abs(min_e - reference_energy):.5f}")
plt.tight_layout()
plt.savefig("skqd_small_results.png", dpi=150)
plt.show()

# %% [markdown]
# ### Verify the energy from the returned state's reduced density matrices
# SQD's energy is a variational upper bound on the true ground-state energy.

# %%
rdm1 = result.sci_state.rdm(rank=1, spin_summed=True)
rdm2 = result.sci_state.rdm(rank=2, spin_summed=True)
energy = np.sum(h1e_momentum * rdm1) + 0.5 * np.sum(h2e_momentum * rdm2)
print(f"Recomputed energy: {energy:.5f}")
print(f"Upper bound holds (SQD >= exact): {min_e >= reference_energy - 1e-9}")
