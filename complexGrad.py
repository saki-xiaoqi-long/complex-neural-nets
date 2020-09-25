import torch
import numpy as np
import re

"""
Does nothing except pretty print complex grad info
"""

class ComplexGrad(torch.Tensor):

    def __deepcopy__(self, memo):
        if not self.is_leaf:
            raise RuntimeError("Only Tensors created explicitly by the user "
                               "(graph leaves) support the deepcopy protocol at the moment")
        if id(self) in memo:
            return memo[id(self)]
        with torch.no_grad():
            if self.is_sparse:
                new_tensor = self.clone()

                # hack tensor to cast as complex
                new_tensor.__class__ = ComplexGrad
            else:
                new_storage = self.storage().__deepcopy__(memo)
                new_tensor = self.new()

                # hack tensor to cast as complex
                new_tensor.__class__ = ComplexGrad
                new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
            memo[id(self)] = new_tensor
            new_tensor.requires_grad = self.requires_grad
            return new_tensor

    def __repr__(self):
        size = self.size()
        split_i = size[0] // 2
        magn = self[:split_i]
        phase = self[split_i:]
        size_m = magn.size()

        magn = magn.view(-1)
        phase = phase.view(-1)

        strings = np.asarray([f'({a}{"exp "}{b}j)' for a, b in zip(magn, phase)])
        strings = strings.reshape(*size_r)
        strings = f'tensor({strings.__str__()})'
        strings = re.sub('\n', ',\n       ', strings)
        return strings

    def __str__(self):
        return self.__repr__()
