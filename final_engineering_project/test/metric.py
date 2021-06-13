from torch import mean, inner, log10, linalg, Tensor


def _vec_l2norm(x: Tensor) -> Tensor:
    return x.pow(2).mean(dim=[1, 2])


def get_snr(input: Tensor, expect: Tensor, remove_dc: bool = True) -> Tensor:
    signal = expect + 1e-9  # Just so won't be zero
    noise = (input - expect) + 1e-9  # Just so won't be zero
    snr = _vec_l2norm(signal) / _vec_l2norm(noise)
    # snr_db = 10 * log10(snr)
    return snr
