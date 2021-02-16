"""Module for implementing emission from quantum systems as MPS."""
import abc
import tncontract as tn
import numpy as np
from typing import List, Optional

import pulse


def _get_transfer_matrix_list(
        left_mps: tn.onedim.MatrixProductState,
        right_mps: tn.onedim.MatrixProductState) -> List[np.ndarray]:
    """Get the transfer matrix for two given matrix product states."""
    if left_mps.nsites != right_mps.nsites:
        raise ValueError("The numebr of sites in the left and right MPS "
                         "should be the same.")
    transfer_matrices = []
    for site in range(left_mps.nsites):
        left_tensor = left_mps.data[site].data
        right_tensor = left_mps.data[site].data
        tensor_prod = [np.kron(left_tensor[:, :, i].conj(),
                               right_tensor[:, :, i])
                       for i in range(left_tensor.shape[-1])]
        transfer_matrices.append(np.sum(tensor_prod, axis=0))

    return transfer_matrices


class OqsSystem(metaclass=abc.ABCMeta):
    """Interface for a Markovian open quantum system emitting a MPS.

    The localized system is modelled with a time-dependent hamiltonian
    `H(p, t)` which is also a function of some control parameters `p` that
    can be varied as a function of time. It is assumed that
                    H(p, t) = H_b(t) + sum_i p_i S
    where `S` is the sensitivity of the system Hamiltonian to the `ith`
    parameter `p`.

    The system is assumed to couple to a bath with operator `L`. The dynamics
    of the system are provided by the coupling operator `L` and the effective
    Hamiltonian
                H_eff(p, t) = H(p, t) - iL.dagger() * L / 2.
    """
    def __init__(self,
                 dt: float,
                 num_tsteps: int,
                 num_params: int,
                 sys_dim: int,
                 bin_dim: int = 2) -> None:
        """Creates a new `OqsSystem` object.

        Args:
            dt: The time-step to use in discretizing the MPS.
            num_tsteps: The number of time-steps to use in setting up the MPS.
            num_params: Number of parameters parametrizing the Hamiltonian. All
                the parameters are assumed to be real.
            sys_dim: The dimensionality of the localized system. We assume this
                to be equal to the bond-dimension.
            bin_dim: The dimensionality of each time bin.
        """
        self._dt = dt
        self._num_tsteps = num_tsteps
        self._num_params = num_params
        self._bin_dim = bin_dim
        self._sys_dim = sys_dim


    @property
    def dt(self) -> float:
        """The time-step used for discretizing the MPS."""
        return self._dt

    @property
    def num_params(self) -> int:
        return self._num_params

    @property
    def num_tsteps(self) -> int:
        """The number of time-steps used for discretizing MPS."""
        return self._num_tsteps

    @property
    def times(self) -> np.ndarray:
        """Returns the times corresponding to the centers of the MPS bins."""
        return (0.5 + np.arange(self._num_tsteps)) * self._dt

    @property
    def bin_dim(self) -> int:
        """The dimensionality of the each waveguide bin."""
        return self._bin_dim

    @property
    def sys_dim(self) -> int:
        """The dimensionality of the system."""
        return self._sys_dim

    @abc.abstractmethod
    def get_bg_hamiltonian(self, t: float) -> np.ndarray:
        """Get the background Hamiltonian at time t."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_pert_operator(self, index: int) -> np.ndarray:
        """Get the perturbation operators for the current system.

        The perturbation is assumed to correspond to a real parameter and is
        consequently Hermitian.

        Args:
            t: The time at which to calculate the perturbation operator.
            ind: Index of the parameter corresponding to which to access the
                perturbation operator.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Get the initial state corresponding to the open quantum system."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_final_state(self) -> np.ndarray:
        """Get the final state corresponding to the open quantum system."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_decay_op(self) -> np.ndarray:
        """Get the decay operator corresponding to the localized system."""
        raise NotImplementedError()

    def get_hamiltonian(
            self, pulses: List[pulse.Pulse], t: float) -> np.ndarray:
        """Calculate the total Hamiltonian.

        Args:
            params: The parameter values at which to compute the Hamiltonian.
            t: The time at which to compute the Hamiltonian.

        Returns:
            The Hamiltonian as a numpy array.
        """
        H = self.get_bg_hamiltonian(t)
        for ind in range(self._num_params):
            H += (pulses[ind](t) * self.get_pert_operator(ind))

        return H

    def get_mps(self,
                pulses: List[pulse.Pulse]) -> tn.onedim.MatrixProductState:
        """Calculate the MPS representation of the current state.

        For a state that exists from time 0 to T, the first lattice of the MPS
        implemented in this function corresponds to the time T and the last
        lattice point of the MPS corresponds to the time 0.

        Args:
            params: The parameters at which to compute the state.

        Returns:
            The state as a matrix product state.
        """
        # We first construct the tensors corresponding to the decay operator.
        # This will be reused repeatedly.
        decay_tensor = np.zeros((self.sys_dim, self.sys_dim, self.bin_dim - 1),
                                dtype=complex)
        decay_tensor[:, :, 0] = self.get_decay_op() * np.sqrt(self.dt)
        for n in range(1, self.bin_dim - 1):
            decay_tensor[:, :, n] = (decay_tensor[:, :, n - 1] @
                                     self.get_decay_op()) * np.sqrt(self.dt) / n

        # Tensors for the MPS for the current state. Right now we ignore the
        # the boundary conditions.
        tensors = []
        # Setup the tensors corresponding to the second to second-to-last
        # time steps.
        for tstep in range(self._num_tsteps - 1, -1, -1):
            time = (tstep + 0.5) * self.dt
            tensor = np.zeros((self.sys_dim, self.sys_dim, self.bin_dim),
                              dtype=complex)
            tensor[:, :, 0] = (np.eye(self.sys_dim, dtype=complex) -
                               1.0j * self.dt * self.get_hamiltonian(
                                    pulses, time))
            tensor[:, :, 1:] = decay_tensor
            tensors.append(tensor)
        # Modify the first vector with a boundary condition corresponding to the
        # the final state.
        final_state = self.get_final_state()
        tensors[0] = np.sum(final_state[:, np.newaxis, np.newaxis] *
                            tensors[0], axis=0)[np.newaxis, :, :]
        # Modify the last vector with a boundary condition corresponding to the
        # initial state.
        init_state = self.get_initial_state()
        tensors[-1] = np.sum(init_state[np.newaxis, :, np.newaxis] *
                             tensors[-1], axis=1)[:, np.newaxis, :]
        return tn.onedim.MatrixProductState(
                    [tn.Tensor(t, labels=["left", "right", "phys"])
                     for t in tensors])

    def get_inner_product(
            self,
            target_mps: tn.onedim.MatrixProductState,
            pulses: List[pulse.Pulse]) -> complex:
        """Compute the inner product of the output photons with given MPS."""
        mps_state = self.get_mps(pulses)
        return tn.onedim.inner_product_mps(target_mps, mps_state)

    def _get_deriv_mps(
            self,
            index: int,
            pulses: List[pulse.Pulse]) -> tn.onedim.MatrixProductState:
        """Constructs a matrix product state by derivatives of tensors."""
        tensors = []
        # We first ignore the boundary conditions and compute the resulting
        # Matrix product state.
        for tstep in range(self._num_tsteps - 1, -1, -1):
            tensor = np.zeros((self.sys_dim, self.sys_dim, self.bin_dim),
                              dtype=complex)
            tensor[:, :, 0] = -1.0j * self.dt * self.get_pert_operator(index)
            tensors.append(tensor)

        # Modify the first vector with a boundary condition corresponding to the
        # final state.
        final_state = self.get_final_state()
        tensors[0] = np.sum(final_state[:, np.newaxis, np.newaxis] *
                            tensors[0], axis=0)[np.newaxis, :, :]
        # Modify the last vector with a boundary condition corresponding to the
        # initial state.
        initial_state = self.get_initial_state()
        tensors[-1] = np.sum(initial_state[np.newaxis, :, np.newaxis] *
                             tensors[-1], axis=1)[:, np.newaxis, :]
        return tn.onedim.MatrixProductState(
                    [tn.Tensor(t, labels=["left", "right", "phys"])
                     for t in tensors])

    def get_inner_prod_gradient(
            self,
            target_state: tn.onedim.MatrixProductState,
            pulses: List[pulse.Pulse]) -> np.ndarray:
        """Compute the gradient of the inner product with adjoint method."""
        # Get the current MPS state.
        mps_state = self.get_mps(pulses)
        # Get the individual transfer matrices.
        transfer_matrices = _get_transfer_matrix_list(target_state, mps_state)
        # For computing the inner product efficiently, we need to setup the
        # cascaded transfer matrices corresponding to the MPS state.
        forward_tmats = [transfer_matrices[0]]
        backward_tmats = [transfer_matrices[-1]]
        for k in range(1, self._num_tsteps - 1):
            forward_tmats.append(
                    forward_tmats[-1] @ transfer_matrices[k])
            backward_tmats.append(
                    transfer_matrices[self._num_tsteps - k - 1] @
                    backward_tmats[-1])

        # Compute the gradients.
        grads = []
        for index in range(self._num_params):
            grad_index = []
            # Calculate the transfer matrices corresponding to the derivative
            # matrix product state.
            deriv_mps_state = self._get_deriv_mps(index, pulses)
            transfer_matrices_deriv = _get_transfer_matrix_list(
                    target_state, deriv_mps_state)

            # Compute the gradient using the adjoint trick. Here we iterate over
            # the site.
            for site in range(self._num_tsteps):
                if site == 0:
                    deriv = transfer_matrices_deriv[0] @ backward_tmats[-1]
                elif site == self._num_tsteps - 1:
                    deriv = forward_tmats[-1] @ transfer_matrices_deriv[-1]
                else:
                    deriv = (forward_tmats[site - 1] @
                             transfer_matrices_deriv[site] @
                             backward_tmats[-1 - site])
                grad_index.append(deriv[0, 0])
            grads.append(np.flip(grad_index))
        return grads



