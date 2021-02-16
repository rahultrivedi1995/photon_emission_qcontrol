"""Module to compute the single and two-photon projections of a continuous
matrix product state.

This module provides functions to compute the single and two-photon components
of a matrix product state. 
"""
import numpy as np
import tncontract as tn
from typing import Optional


def compute_single_ph_proj(state: tn.onedim.MatrixProductState) -> np.ndarray:
    """Compute the single-photon component of the matrix product state.

    Args:
        state: The state to be considered expressed as a matrix product state
            object.

    Returns:
        The single-photon state as a numpy array.
    """
    # We first extract the tensors corresponding to the zero and single photons
    # in the time-bins.
    tensors_0 = [state.data[n].data[:, :, 0] for n in range(state.nsites)]
    tensors_1 = [state.data[n].data[:, :, 1] for n in range(state.nsites)]
    # Multiply with all the vacuum tensors except for the first one.
    tensor_prod_right = tensors_0[1]
    for t in tensors_0[2:]:
        tensor_prod_right = tensor_prod_right @ t
    # We will also maintain products from the left.
    tensor_prod_left = tensors_0[0]

    # Calculate the single photon projection.
    single_ph_proj = [np.trace(tensors_1[0] @ tensor_prod_right)]
    for tstep in range(1, state.nsites - 1):
       tensor_prod_right = np.linalg.inv(tensors_0[tstep]) @ tensor_prod_right
       single_ph_proj.append(
           np.trace(tensor_prod_left @ tensors_1[tstep] @ tensor_prod_right))
       tensor_prod_left = tensor_prod_left @ tensors_0[tstep]

    single_ph_proj.append(np.trace(tensor_prod_left @ tensors_1[-1]))
    return np.flip(single_ph_proj)


def compute_photon_number_exp(
        state: tn.onedim.MatrixProductState) -> np.ndarray:
    """Computes the expectation number of photons with position.

    Args:
        state: The state being considered as a matrix product state.

    Returns:
        The mean number of photons in the emitted state as a function of time.
    """
    # We assume that the number of photons in each bin is restricted to 0 or 1.
    num_op = tn.Tensor(np.array([[0, 0], [0, 1]], dtype=complex))

    # Compute the expectation value of the number of photons.
    ph_num = []
    for site in range(state.nsites):
        ph_num.append(state.expval(num_op, site))

    return np.flip(ph_num)


def single_ph_state_as_mps(
        psi: np.ndarray,
        bond_dim: Optional[int] = 2,
        bin_dim: Optional[int] = 2) -> tn.onedim.MatrixProductState:
    """Implements a single-photon state as a MPS.

    Args:
        psi: The amplitude of the single photon state. Note that this is assumed
            to be given on a discretized time-bin basis.

    Returns:
        The state of the single photon as a matrix product state.
    """
    # We flip the input state, since in our convention the time axis is
    # reversed.
    psi_flip = np.flip(psi)
    # Also normalize the state.
    psi_flip = psi_flip / np.linalg.norm(psi_flip)
    # Express the state as a matix product state.
    tensors = []
    # The first tensor is a row vector.
    tensor = np.zeros((1, bond_dim, bin_dim), dtype=complex)
    tensor[0, 0:2, 0] = np.array([1, 0])
    tensor[0, 0:2, 1] = np.array([0, psi_flip[0]])
    tensors.append(tn.Tensor(tensor, labels=["left", "right", "phys"]))

    # The second to second to last tensors are matrices.
    for site in range(1, psi.size - 1):
        tensor = np.zeros((bond_dim, bond_dim, bin_dim), dtype=complex)
        tensor[0:2, 0:2, 0] = np.eye(2, dtype=complex)
        tensor[0, 1, 1] = psi_flip[site]
        tensors.append(tn.Tensor(tensor, labels=["left", "right", "phys"]))

    # The last tensor is a column vector.
    tensor = np.zeros((bond_dim, 1, bin_dim), dtype=complex)
    tensor[0:2, 0, 0] = np.array([0, 1])
    tensor[0:2, 0, 1] = np.array([psi_flip[-1], 0])
    tensors.append(tn.Tensor(tensor, labels=["left", "right", "phys"]))

    return tn.onedim.MatrixProductState(tensors)
