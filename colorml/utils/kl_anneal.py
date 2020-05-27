# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np


def monotonical_kl_anneal(epoch, M=100):
    """
    To use the standard monotonical anneal as in Bowman et al. 2015
    """
    return np.min([np.tanh(1 / M * epoch), 1.0])


def linear_kl_anneal(epoch, M=100):
    """
    To use the standard monotonical anneal as in Bowman et al. 2015
    """
    return np.min([epoch / M, 1.0])


def cycle_kl_anneal(epoch, M=4):
    """
    Implementing Fu et al. 2019
    """
    R = 0.5
    T = 150
    TMRATIO = T / M

    tau = (epoch - 1) % (TMRATIO)
    tau /= TMRATIO

    def f(tau):
        return tau / R

    if tau <= R:
        return f(tau)

    else:
        return 1.0
