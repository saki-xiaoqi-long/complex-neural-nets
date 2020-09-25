from complexScalar import ComplexScalar
from complexGrad import ComplexGrad

import inspect
import numpy as np
import torch
import re
import cmath


"""
Complex tensor support for PyTorch.
Uses a regular tensor where the first half are the magnitude and second are the phase.
Supports only some basic operations without breaking the gradients for complex math.
Supported ops:
1. addition 
    - (tensor, scalar). Both complex and real.
2. subtraction 
    - (tensor, scalar). Both complex and real.
3. multiply
    - (tensor, scalar). Both complex and real.
4. mm (matrix multiply)
    - (tensor). Both complex and real.
5. all indexing ops.
6. t (transpose)
>> c = ComplexTensor(10, 20)
>> #  do regular tensor ops now
>> c = c * 4
>> c = c.mm(c.t())
>> print(c.shape, c.size(0))
>> print(c)
>> print(c[0:1, 1:-1])
"""


class ComplexTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        if isinstance(x, np.ndarray) and 'complex' in str(x.dtype):
            # collapse second to last dim
            print(type(x))
            r = x.real
            i = x.imag
            n_magn = np.sqrt(torch.pow(real, 2) + np.pow(imag, 2))
            n_phase = np.atan(i / r)
            x = np.concatenate([n_magn, n_phase], axis=-2)

        # x is the second to last dim in this case
        if type(x) is int and len(args) == 1:
            x = x * 2

        elif len(args) >= 2:
            size_args = list(args)
            size_args[-2] *= 2
            args = tuple(size_args)

        else:
            if isinstance(x, torch.Tensor):
                s = x.size()[-2]
            elif isinstance(x, list):
                s = len(x)
            elif isinstance(x, np.ndarray):
                s = x.shape[-2]
            if not (s % 2 == 0): raise Exception('second to last dim must be even. ComplexTensor is 2 real matrices under the hood')

        # init new t
        new_t = super().__new__(cls, x, *args, **kwargs)
        return new_t

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
                new_tensor.__class__ = ComplexTensor
            else:
                new_storage = self.storage().__deepcopy__(memo)
                new_tensor = self.new()

                # hack tensor to cast as complex
                new_tensor.__class__ = ComplexTensor
                new_tensor.set_(new_storage, self.storage_offset(), self.size(), self.stride())
            memo[id(self)] = new_tensor
            new_tensor.requires_grad = self.requires_grad
            return new_tensor

    @property
    def magn(self):
        size = self.size()
        n = size[-2]
        n = n * 2
        result = self[..., :n//2, :]
        return result

    @property
    def phase(self):
        size = self.size()
        n = size[-2]
        n = n * 2
        result = self[..., n//2:, :]
        return result

    def __graph_copy__(self, magn, phase):
        # return tensor copy but maintain graph connections
        # force the result to be a ComplexTensor
        result = torch.cat([magn, phase], dim=0)
        result.__class__ = ComplexTensor
        return result

    def __graph_copy_scalar__(self, magn, phase):
        # return tensor copy but maintain graph connections
        # force the result to be a ComplexTensor
        result = torch.stack([magn, phase], dim=-2)
        result.__class__ = ComplexScalar
        return result

    def __add__(self, other):
        """
        Handles scalar and tensor addition
        :param other:
        :return:
        """
        magn = self.magn
        phase = self.phase
        real = magn * torch.cos(phase)
        imag = magn * torch.sin(phase)

        # given a real tensor
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real + other

        # given a complex tensor
        elif type(other) is ComplexTensor:
            o_mag = other.magn
            o_ph = other.phase
            o_real = magn * torch.cos(phase)
            o_imag = magn * torch.sin(phase)
            real = real + other_real
            imag = imag + other_imag

        # given a real scalar
        elif np.isreal(other):
            real = real + other

        # given a complex scalar
        else:
            real = real + other.real
            imag = imag + other.imag

        # back to polar form
        n_magn = torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))
        n_phase = torch.atan(imag / real)

        return self.__graph_copy__(n_magn, n_phase)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """
        Handles scalar (real, complex) and tensor (real, complex) subtraction
        :param other:
        :return:
        """
        magn = self.magn
        phase = self.phase
        real = magn * torch.cos(phase)
        imag = magn * torch.sin(phase)

        # given a real tensor
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real - other

        # given a complex tensor
        elif type(other) is ComplexTensor:
            o_mag = other.magn
            o_ph = other.phase
            o_real = magn * torch.cos(phase)
            o_imag = magn * torch.sin(phase)
            real = real - other_real
            imag = imag - other_imag

        # given a real scalar
        elif np.isreal(other):
            real = real - other

        # given a complex scalar
        else:
            real = real - other.real
            imag = imag - other.imag

        # back to polar form
        n_magn = torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))
        n_phase = torch.atan(imag / real)

        return self.__graph_copy__(n_magn, n_phase)

    def __rsub__(self, other):
        return self.__sub__(other)

    def __mul__(self, other):
        """
        Handles scalar (real, complex) and tensor (real, complex) multiplication
        :param other:
        :return:
        """
        magn = self.magn
        phase = self.phase
        
        # given a real tensor
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            magn = magn * other

        # given a complex tensor
        elif type(other) is ComplexTensor:
            magn = magn * other.magn
            phase = phase + other.phase

        # given a real scalar
        elif np.isreal(other):
            magn = magn * other

        # given a complex scalar
        else:
            magn = magn * cmath.abs(other)
            phase = phase + cmath.phase(other)

        return self.__graph_copy__(magn, phase)

    def __truediv__(self, other):
        magn = self.magn
        phase = self.phase

        # given a real tensor
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            magn = magn / other

        # given a complex tensor
        elif type(other) is ComplexTensor:
            magn = magn / other.magn
            phase = phase - other.phase

        # given a real scalar
        elif np.isreal(other):
            magn = magn / other

        # given a complex scalar
        else:
            magn = magn / cmath.abs(other)
            phase = phase - cmath.phase(other)

        return self.__graph_copy__(magn, phase)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self.__mul__(-1)

    def mm(self, other):
        """
        Handles tensor (real, complex) matrix multiply
        :param other:
        :return:
        """
        magn = self.magn
        phase = self.phase
        real = magn * torch.cos(phase)
        imag = magn * torch.sin(phase)

        # given a real tensor
        if isinstance(other, torch.Tensor) and type(other) is not ComplexTensor:
            real = real.mm(other)
            imag = imag.mm(other)

        # given a complex tensor
        elif type(other) is ComplexTensor:
            o_real = other.magn * torch.cos(other.phase)
            o_img = other.magn * torch.sin(other.phase)
            ac = real.mm(o_real)
            bd = imag.mm(o_img)
            ad = real.mm(o_img)
            bc = imag.mm(o_real)
            real = ac - bd
            imag = ad + bc

        # back to polar form
        n_magn = torch.sqrt(torch.pow(real, 2) + torch.pow(imag, 2))
        n_phase = torch.atan(imag / real)

        return self.__graph_copy__(n_magn, n_phase)

    def t(self):
        real = self.magn.t()
        imag = self.phase.t()

        return self.__graph_copy__(magn, phase)

    def sum(self, *args):
        magn_sum = self.magn.sum(*args)
        phase_sum = self.phase.sum(*args)
        return ComplexScalar(magn_sum, phase_sum)

    def mean(self, *args):
        magn_mean = self.magn.mean(*args)
        phase_mean = self.phase.mean(*args)
        return ComplexScalar(magn_mean, phase_mean)

    @property
    def grad(self):
        g = self._grad
        g.__class__ = ComplexGrad

        return g

    def cuda(self):
        magn = self.magn.cuda()
        phase = self.phase.cuda()

        return self.__graph_copy__(magn, phase)

    def __repr__(self):
        magn = self.magn.flatten()
        phase = self.phase.flatten()

        # use numpy to print for us
        strings = np.asarray([complex(a,b) for a, b in zip(real, imag)]).astype(np.complex64)
        strings = strings.__repr__()
        strings = re.sub('array', 'tensor', strings)
        return strings
        

    def __str__(self):
        return self.__repr__()

    def is_complex(self):
        return True

    def size(self, *args):
        size = self.data.size(*args)
        size = list(size)
        size[-2] = size[-2] // 2
        size = torch.Size(size)
        return size

    @property
    def shape(self):
        size = self.data.shape
        size = list(size)
        size[-2] = size[-2] // 2
        size = torch.Size(size)
        return size

    def __getitem__(self, item):

        # when magn or phase is the caller return regular tensor
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        caller = calframe[1][3]

        if caller == 'magn' or caller == 'phase':
            return super(ComplexTensor, self).__getitem__(item)

        # this is a regular index op, select the requested pairs then form a new ComplexTensor
        m = self.magn[item]
        p = self.phase[item]

        return self.__graph_copy__(m, p)
