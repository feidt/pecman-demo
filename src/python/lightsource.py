import numpy as np
from abc import ABCMeta, abstractmethod
import cmath

class LightSource(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def value(self,w):
        """
        this method returns the field strength inside the solid
        """
        raise NotImplementedError("Should implement fieldstrength()!")


class CWSource(LightSource):

    def __init__(self,field_strength=1.,central_energy=21.2):
        self.field_strength = field_strength
        self.central_energy = central_energy

    def value(self,w):
        return self.field_strength


class Laser(LightSource):
    def __init__(self,field_strength=1.,central_energy=21.2,spectral_width=0.02):
        self.field_strength = field_strength
        self.central_energy = central_energy
        self.spectral_width = spectral_width
        #sech: dw(in eV) = 2hbar/(pi*dt)

    def value(self,w):
        #return np.exp(-0.5*(w-self.central_energy)**2/(self.spectral_width**2))
        return 1./np.cosh((w-self.central_energy)/self.spectral_width)
