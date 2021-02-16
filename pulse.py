"""Module to implement pulses applied as classical controls."""
import abc
import numpy as np

from typing import List, Tuple, Optional


class Pulse(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, t: float) -> float:
        raise NotImplementedError()


class ConstantPulse(Pulse):
    """Implements a constant amplitude pulse."""
    def __init__(self,
                 amp: complex) -> None:
        """Creates a new `ConstantPulse` object.

        Args:
            amp: The amplitude of the pulse which in general can be a constant
                complex number.
        """
        self._amp = amp

    def __call__(self, t: float) -> complex:
        return self._amp


class GaussianPulse(Pulse):
    """Implements a Gaussian amplitude pulse."""
    def __init__(self,
                 amp: float,
                 t_cen: float,
                 t_width: float) -> None:
        self._amp = amp
        self._t_cen = t_cen
        self._t_width = t_width

    def __call__(self, t: float) -> float:
        return self._amp * np.exp(
                -(t - self._t_cen)**2 / self._t_width**2)


