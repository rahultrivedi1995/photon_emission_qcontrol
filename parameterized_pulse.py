"""Module to implement pulse parametrization for performing control."""
import abc
import numpy as np
from typing import List, Tuple, Optional

import pulse


class ParameterizedPulse(pulse.Pulse):
    """Defines an interface for a pulse whose state can be updated."""
    @abc.abstractmethod
    def update(self, vec: np.ndarray) -> None:
        """Update the state of the pulse with the given vector."""
        raise NotImplementedError()

    @abc.abstractmethod
    def state(self) -> np.ndarray:
        """Get the current state of the pulse."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_gradient(self, times: np.ndarray) -> np.ndarray:
        """Get the gradient of the pulse values relative to its state."""
        raise NotImplementedError()


class DirectParameterizedPulse(ParameterizedPulse):
    def __init__(self,
                 max_time: float,
                 num_tsteps: int,
                 bounds: Tuple[float, float],
                 init_state: Optional[np.ndarray] = None) -> None:
        """Creates a new `DirectParametrizedPulse` object.

        This discretizes the pulse into a piecewise constant function.

        Args:
            max_time: The maximum time within which the pulse is defined.
            num_tsteps: The number of time-steps to discretize the pulse into.
            bounds: The upper and lower value that the pulse can take.
            init_state: The intial state of the pulse.
        """
        self._max_time = max_time
        self._num_tsteps = num_tsteps
        self._time_mesh = np.linspace(0, max_time, num_tsteps + 1)
        self._dt = max_time / num_tsteps
        self._bounds = bounds
        if init_state is None:
            init_state = np.random.uniform(
                    bounds[0], bounds[1], num_tsteps + 1)
        self._state = init_state

    def bounds(self) -> List[Tuple[float, float]]:
        return np.array([(self._bounds[0], self._bounds[1])
                         for _ in range(self.state().size)], dtype=float)

    def state(self) -> np.ndarray:
        """Returns the state of the pulse."""
        return self._state

    def update(self, vec: np.ndarray) -> None:
        self._state = vec

    def get_gradient(self, times: np.ndarray) -> np.ndarray:
        # We first digitize the given pulse into the time-bins.
        time_inds = np.arange(times.size)
        bins = np.digitize(times, self._time_mesh)
        # Remove the time indices that corresponds to the pulse falling outside
        # the pulse length. Outside the pulse length, the pulse is assumed to
        # be 0, so the gradient vanishes.
        time_inds_mod = [t for b, t in zip(bins, time_inds)
                         if b != 0 and b != self._num_tsteps + 1]
        bins_mod = [b for b in bins if b != 0 and b != self._num_tsteps + 1]
        # Fill in the gradients.
        grad = np.zeros((self._num_tsteps + 1, times.size), dtype=complex)
        grad[np.array(bins_mod), np.array(time_inds_mod)] = 1.0
        return grad

    def __call__(self, t: float) -> complex:
        bin_t = np.digitize(t, self._time_mesh)
        if bin_t == 0 or bin_t == self._num_tsteps + 1:
            return 0
        else:
            return self._state[bin_t - 1]


