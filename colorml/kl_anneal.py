import numpy as np


def monotonical_kl_anneal(epoch):
    """
    To use the standard monotonical anneal as in Bowman et al. 2015
    """
    M = 20
    return np.min([np.tanh(1 / M * epoch), 1])
