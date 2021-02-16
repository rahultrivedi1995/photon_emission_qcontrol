"""Module to implement various cavity QED systems."""
import numpy as np
from typing import List

import oqs_emission_mps


def _nested_kronecker_prod(
        arrays: List[np.ndarray]) -> np.ndarray:
    """Perform a nested kronecker product of a list of arrays."""
    a = arrays[0]
    for k in range(1, len(arrays)):
        a = np.kron(a, arrays[k])
    return a


class DrivenModulatedTLS(oqs_emission_mps.OqsSystem):
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

    def get_initial_state(self) -> np.ndarray:
        # Just set the initial state of the emitter to be excited.
        return np.array([1, 0])

    def get_final_state(self) -> np.ndarray:
        # Set the initial state of the emitter to be ground state.
        return np.array([0, 1])

    def get_decay_op(self) -> np.ndarray:
        return np.sqrt(self._gamma) * np.array([[0, 0], [1, 0]])


class ModulatedTavisCumming(oqs_emission_mps.OqsSystem):
    """Implements a modulated Tavis Cumming's model."""
    def __init__(
            self,
            dt: float,
            num_tsteps: int,
            emitter_deltas: List[float],
            coup_const: float,
            kappa: float) -> None:
        """Creates a new `ModulatedTavisCumming` object.

        Args:
            dt: The time discretization to use for this problem.
            num_tsteps: The number of time-steps.
            emitter_deltas: The detunings of the emitter frequencies.
            coup_const: The coupling constant of the emitters to the cavity
                mode.
            kappa: The cavity decay rate.
        """
        self._num_emitters = len(emitter_deltas)
        super(ModulatedTavisCumming, self).__init__(
                dt, num_tsteps, 1,
                np.power(2, self._num_emitters) * (self._num_emitters + 1))
        self._emitter_deltas = emitter_deltas
        self._coup_const = coup_const
        self._kappa = kappa

        # Setup some matrices corresponding to the operators.
        self._a = self._cavity_ann_op()
        self._sigmas = [self._emitter_decay_op(index)
                        for index in range(self._num_emitters)]

    def _cavity_ann_op(self) -> np.ndarray:
        """Returns the cavity annihilation operator."""
        # Constructed as the kronecker product of the cavity operator followed
        # by the lowering operators corresponding to the emitters.
        # We will truncate the cavity hilbert space to the number of emitters.
        a = np.zeros((self._num_emitters + 1, self._num_emitters + 1),
                     dtype=complex)
        a[np.arange(self._num_emitters),
          np.arange(1, self._num_emitters + 1)] = np.sqrt(
                  np.arange(1, self._num_emitters + 1))
        list_op = [a] + [np.eye(2, dtype=complex)] * self._num_emitters
        return _nested_kronecker_prod(list_op)

    def _emitter_decay_op(self, index) -> np.ndarray:
        """Returns the excitation operator corresponding to the emitter.

        index: The emitter which we are dealing with. This index starts from
            0.
        """
        list_op = (
                [np.eye(self._num_emitters + 1, dtype=complex)] +
                [np.eye(2, dtype=complex)] * self._num_emitters)
        list_op[index + 1] = np.array([[0, 1], [0, 0]], dtype=complex)
        return _nested_kronecker_prod(list_op)

    def get_bg_hamiltonian(self, t: float) -> np.ndarray:
        """Implements the modulated tavis cumming Hamiltonian."""
        # Total energy of the emitters.
        H_emitters = np.sum([self._emitter_deltas[i] *
                             self._sigmas[i].conj().T @ self._sigmas[i]
                             for i in range(self._num_emitters)], axis=0)
        # Interaction of the emitters with the cavity.
        H_int = np.sum([self._sigmas[i].conj().T @ self._a +
                        self._a.conj().T @ self._sigmas[i]
                        for i in range(self._num_emitters)], axis=0)
        # Decay of the cavity mode.
        H_decay = -0.5j * self._kappa * self._a.conj().T @ self._a
        return H_emitters + H_int + H_decay

    def get_pert_operator(self, ind: int) -> np.ndarray:
        if ind == 0:
            return self._a.conj().T @ self._a
        else:
            raise ValueError("There is only one tunable parameter.")

    def get_initial_state(self) -> np.ndarray:
        init_cav_state = np.array([1] + [0] * self._num_emitters)
        init_em_state = np.array([0, 1])
        return _nested_kronecker_prod(
                [init_cav_state] + [init_em_state] * self._num_emitters)

    def get_final_state(self) -> np.ndarray:
        final_cav_state = np.array([1] + [0] * self._num_emitters)
        final_em_state = np.array([1, 0])
        return _nested_kronecker_prod(
                [final_cav_state] + [final_em_state] * self._num_emitters)

    def get_decay_op(self) -> np.ndarray:
        return np.sqrt(self._kappa) * self._a

