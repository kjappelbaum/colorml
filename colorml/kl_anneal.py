import numpy as np


def monotonical_kl_anneal(epoch):
    """
    To use the standard monotonical anneal as in Bowman et al. 2015
    """
    M = 20
    return np.min([np.tanh(1 / M * epoch), 1])


def cycle_kl_anneal(epoch):
    """
    Implementing Fu et al. 2019
    """
    R = 0.5
    M = 4
    T = 200
    TMRATIO = T / M

    tau = (epoch - 1) % (TMRATIO)
    tau /= TMRATIO

    def f(tau):
        return tau / R

    if tau <= R:
        return f(tau)

    else:
        return 1.0

