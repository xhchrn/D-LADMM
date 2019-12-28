"""
File:    mu_updater.py

Created: September 19, 2019

Revised: December 2, 2019

Authors: Howard Heaton, Xiaohan Chen

Purpose: Define a set of mu updaters in Safeguarded KM method.
         We implement 5 types of mu updaters, named `Geometric Series`,
         `Arithmetic Average`, `Exponential Moving Average`, `Recent Term`,
         and `Recent Max` respectively.
"""

import numpy as np

class EMAUpdater(object):

    """Docstring for EMAUpdater. """

    def __init__(self, mu, parameter):
        """TODO: to be defined. """
        self.mu = mu
        self.parameter = parameter

    def step(self, Sx_L2O_norm, bool_term):
        update    = self.parameter * Sx_L2O_norm + (1 - self.parameter) * self.mu
        bool_comp = 1.0 - bool_term
        self.mu   = (bool_comp * self.mu + bool_term * update).detach()
        return self.mu

class GSUpdater(object):

    """Docstring for GSUpdater. """

    def __init__(self, mu, parameter):
        """TODO: to be defined.

        :mu: TODO
        :parameter: TODO

        """
        self.mu = mu
        self.parameter = parameter

    def step(self, Sx_L2O_norm, bool_term):
        update    = (1 - self.parameter) * self.mu
        bool_comp = 1.0 - bool_term
        self.mu   = (bool_comp * self.mu + bool_term * update).detach()
        return self.mu


class RTUpdater(object):

    """Docstring for RTUpdater. """

    def __init__(self, mu, parameter):
        """TODO: to be defined.

        :mu: TODO
        :parameter: TODO

        """
        self.mu = mu
        self.parameter = parameter

    def step(self, Sx_L2O_norm, bool_term):
        update = Sx_L2O_norm
        bool_comp = 1.0 - bool_term
        self.mu   = (bool_comp * self.mu + bool_term * update).detach()
        return self.mu


class RMUpdater(object):

    """Docstring for RMUpdater. """

    def __init__(self, mu, parameter):
        """TODO: to be defined.

        :mu: TODO
        :parameter: TODO

        """
        self.parameter = int(parameter)
        self.recent = mu.new_zeros((mu.shape[0], parameter))
        self.pointer = 0
        self.step(mu)

    def step(self, Sx_L2O_norm, bool_term=None):
        self.recent[:,self.pointer] = Sx_L2O_norm
        self.pointer = (self.pointer + 1) % self.parameter
        return self.recent.max(dim=1)


class BlankUpdater(object):

    """Docstring for BlankUpdater. """

    def __init__(self, mu, parameter):
        """TODO: to be defined.
        :mu: TODO
        :parameter: TODO
        """

    def step(self, Sx_L2O_norm, bool_term):
        return np.power(10.0,10)

mu_updater_dict = {
    'EMA': EMAUpdater,
    'GS': GSUpdater,
    'RT': RTUpdater,
    'RM': RMUpdater,
    'None': BlankUpdater
}
