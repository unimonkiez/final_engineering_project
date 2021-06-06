from torch import mean, inner, log10, linalg, Tensor


def _vec_l2norm(x: Tensor) -> Tensor:
    return linalg.norm(x, ord=2, dim=[2])


def get_snr(x: Tensor, s: Tensor, remove_dc: bool = True) -> Tensor:
    """
    Compute SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - mean(x)
        s_zm = s - mean(s)
        t = inner(x_zm, s_zm) * s_zm / _vec_l2norm(s_zm) ** 2
        n = x_zm - t
    else:
        t = inner(x, s) * s / _vec_l2norm(s) ** 2
        n = x - t
    output = 20 * log10(_vec_l2norm(t) / _vec_l2norm(n))
    return output.mean()
