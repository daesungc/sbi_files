import numpy as np
import sbi.utils
import torch
from npeet import entropy_estimators as ee

entropy = lambda x: ee.entropy(x, base=np.e)


def get_max_entropy(lb, ub, dims=None):
    """
    Returns entropy of the uniform distribution with the same bounds as the given system.

    Parameters
    ----------
    lb : float, List, torch.Tensor
        Lower bound. Can be a scalar or a list/tensor; length must be equal to ub.
    ub : float, List, torch.Tensor
        Upper bound. Can be a scalar or a list/tensor; length must be equal to ub.
    dims : int
        Number of dimensions. Ignored if the bounds are not scalars.

    Returns
    -------
    ndarray
        Entropy of the uniform distribution given as a single-valued array.
    """

    if isinstance(lb, (int, float)):  # There must be a better way to do this
        assert isinstance(
            lb, (int, float)
        ), "Both bounds require integer or float values if one is."
        assert dims != None, "Supply the number of dimensions."
    elif isinstance(lb, list):
        lb = torch.Tensor(lb)
        ub = torch.Tensor(ub) if isinstance(ub, list) else ub
    elif isinstance(lb, torch.Tensor):
        ub = torch.Tensor(ub) if isinstance(ub, list) else ub
    else:
        raise TypeError("Check the bounds")

    dist = (
        sbi.utils.BoxUniform(low=lb * torch.ones(dims), high=ub * torch.ones(dims))
        if dims is not None
        else sbi.utils.BoxUniform(low=lb, high=ub)
    )

    return dist.entropy().numpy()
