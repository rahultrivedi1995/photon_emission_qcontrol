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
    def get_init_state(self) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_final_state(self) -> np.ndarray:
        raise NotImplementedError()

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
        decay_tensor[:, :, 0] = self.get_decay_op() * self.dt
        for n in range(1, self.bin_dim - 1):
            decay_tensor[:, :, n] = (decay_tensor[:, :, n - 1] @
                                     self.get_decay_op()) * self.dt / n

        # Tensors for the MPS for the current state.
        tensors = []
        # Setup the tensors corresponding to the second to second-to-last
        # time steps.
        for tstep in range(self._num_tsteps - 1, 0, -1):
            time = (tstep + 0.5) * self.dt
            tensor = np.zeros((self.sys_dim, self.sys_dim, self.bin_dim),
                              dtype=complex)
            tensor[:, :, 0] = (np.eye(self.sys_dim, dtype=complex) -
                               1.0j * self.dt * self.get_hamiltonian(
                                    pulses, time))
            tensor[:, :, 1:] = decay_tensor
            tensors.append(tn.Tensor(tensor,
                                     labels=["left", "right", "phys"]))
        # Setup the last tensor. We note that this is a column vector.
        init_state
        tensor = np.zeros((self.sys_dim, 1, self.bin_dim),
                          dtype=complex)
        tensor[:, 0, 0] = (np.eye(self.sys_dim, dtype=complex) -
                           1.0j * self.dt *
                           self.get_hamiltonian(
                               pulses,
                               0.5 * self.dt))[:, -1]
        tensor[:, 0, 1:] = decay_tensor[:, -1, :]
        tensors.append(tn.Tensor(tensor, labels=["left", "right", "phys"]))

        return tn.onedim.MatrixProductState(tensors)

    def get_inner_prod_gradient(
            self,
            target_state: tn.onedim.MatrixProductState,
            pulses: List[pulse.Pulse],
        ) -> np.ndarray:
        """Calculate the tangent MPS corresponding to derivative.

        Args:
            target_state: The state with which to compute the overlap.
            params: The parameters at which to compute the gradient.
            index: The index with respect to which to compute the gradient.

        Return:
            The gradient of the matrix-product state with respect to the
            parameters.
        """
        mps_state = self.get_mps(pulses)
        grads = []
        for index in range(self._num_params):
            # Extract the perturbation operator corresponding to the current
            # parameter.
            pert_op = self.get_pert_operator(index)
            # Get the MPS state corresponding to the current parameters.
            grads.append([])
            # Handle the gradient due to the first tensor.
            mps_state_copy = mps_state.copy()
            mps_state_copy.data[0].data[0, :, 0] = (
                    -1.0j * self.dt * pert_op[-1, :])
            mps_state_copy.data[0].data[0, :, 1] = 0
            grads[-1].append(tn.onedim.inner_product_mps(
                        target_state, mps_state_copy))
            # Handle the gradient at the intermediate terms.
            for tstep in range(1, self._num_tsteps - 1):
                mps_state_copy = mps_state.copy()
                mps_state_copy.data[tstep].data[:, :, 0] = (
                        -1.0j * self.dt * pert_op)
                mps_state_copy.data[tstep].data[:, :, 1] = 0
                grads[-1].append(tn.onedim.inner_product_mps(
                            target_state, mps_state_copy))
            # Handle the gradient due to the last tensor.
            mps_state_copy = mps_state.copy()
            mps_state_copy.data[-1].data[:, 0, 0] = (
                    -1.0j * self.dt * pert_op[:, -1])
            mps_state_copy.data[-1].data[:, 0, 1] = 0
            grads[-1].append(tn.onedim.inner_product_mps(
                        target_state, mps_state_copy))

        return grads

    def _get_deriv_tensor(
            self,



    def get_inner_prod_gradient_adjoint(
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
        for k in range(1, self._num_tsteps):
            forward_tmats.append(
                    transfer_matrices[k] @ forward_tmats[-1])
            backward_tmats.append(
                    backward_tmats[-1] @
                    transfer_matrices[self._num_tsteps - k - 1])

        # Compute the gradients.
        grads = []
        for index in range(self._num_params):
            grads.append([])
            # Extract the perturbation operator.
            pert_op = self.get_pert_operator(index)
            # Handle the gradient due to the first tensor.
            deriv_tensor = np.zeros(
                    (1, self.sys_dim, self.bin_dim),
                    dtype=complex)
            deriv_tensor[0, :, 0] = -1.0j * dt * pert_op[-1, :]
            deriv_tmat = np.kron(target_state.data[0].data, deriv_tensor)
            grads[-1].append(deriv_tmat



class DrivenModulatedTLS(OqsSystem):
    """Implements a driven modulated two-level system.

    This is a two-level system with a Hamiltonian
      H_s(delta, Omega) = (delta * sigma.dag() * sigma +
                           Omega_x * sigma_x + Omega_y * sigma_y
    where `delta` is the detuning of the two-level system from the laser
    frequency and `Omega_x, Omega_y` are the quadratures of the laser pulse.
    All of `delta`, `Omega_x` and `Omega_y` are considered to be control
    parameters for the two-level system. The coupling operator for the
    two-level system is assumed to be `sqrt(gamma) * sigma`, where `sigma` is
    the decay rate.
    """
    def __init__(
            self,
            dt: float,
            num_tsteps: int,
            gamma: float) -> None:
        """Makes a new `DrivenModulatedTLS` object.

        Args:
            dt: The time-step to use.
            num_tsteps: The number of time steps to use in the simulation.
            gamma: The decay rate of the two-level system into the bath.
        """
        super(DrivenModulatedTLS, self).__init__(dt, num_tsteps, 3, 2)
        self._gamma = gamma

    def get_bg_hamiltonian(self, t: float) -> np.ndarray:
        """The background effective Hamiltonian for the two-level system.

        This is given by `-0.5j * sigma.dag() * sigma`.
        """
        return np.array([[-0.5j * self._gamma, 0], [0, 0]])

    def get_pert_operator(self, ind: int) -> np.ndarray:
        """Refer to the parent class for documentation.

        We assume that the first parameter is `delta`, the second parameter is
        `Omega_x` and the third parameter is `Omega_y`.
        """
        pert_op = None
        if ind == 0:
            pert_op = np.array([[1, 0], [0, 0]])
        elif ind == 1:
            pert_op = np.array([[0, 1], [1, 0]])
        elif ind == 2:
            pert_op = np.array([[0, -1.0j], [1.0j, 0]])
        else:
            raise ValueError("There are only three parameters, recieved "
                             "index = {}".format(ind))
        return pert_op

    def get_decay_op(self) -> np.ndarray:
        return np.sqrt(self._gamma) * np.array([[0, 0], [1, 0]])
