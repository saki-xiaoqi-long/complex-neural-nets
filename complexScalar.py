"""
Thin wrapper for complex scalar.
Main contribution is to use only real part for backward
"""


class ComplexScalar(object):

    def __init__(self, magn, phase):
        self._magn = magn
        self._phase = phase

    @property
    def magn(self):
        return self._magn

    @property
    def phase(self):
        return self._phase

    def backward(self):
        self._magn.backward()

    def __repr__(self):
        return str(self.magn.item(), self.phase.item())

    def __str__(self):
        return self.__repr__()
