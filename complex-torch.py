import torch
from complextensor import ComplexTensor


def __graph_copy__(magn, phase):
    # return tensor copy but maintain graph connections
    # force the result to be a ComplexTensor
    result = torch.cat([magn, phase], dim=-2)
    result.__class__ = ComplexTensor
    return result


def __apply_fx_to_parts(items, fx, *args, **kwargs):
    m = [x.magn for x in items]
    m = fx(r, *args, **kwargs)

    p = [x.phase for x in items]
    p = fx(i, *args, **kwargs)

    return __graph_copy__(r, i)


def stack(items, *args, **kwargs):
    return __apply_fx_to_parts(items, torch.stack, *args, **kwargs)


def cat(items, *args, **kwargs):
    return __apply_fx_to_parts(items, torch.cat, *args, **kwargs)
