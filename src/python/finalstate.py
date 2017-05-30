import numpy as np
from abc import ABCMeta, abstractmethod
import cmath

class FinalState(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_kv(self,kx,ky,w):
        """
        this method returns the perpenticular momentum component
        inside the solid
        """
        raise NotImplementedError("Should implement get_kv()!")

    @abstractmethod
    def get_ks(self,kx,ky,w):
        """
        this method returns the perpenticular momentum component
        inside the solid
        """
        raise NotImplementedError("Should implement get_ks()")

class ILEEDState(FinalState):

    def __init__(self,inner_potential):
        self.inner_potential = inner_potential
        self.alpha =  0.0381 #hbar*hbar/2m_electron, [eV nm**2]

    def get_kv(self,kx,ky,w):
        """TO DO: self_energy corrections, mean free path """
        kperp = cmath.sqrt(1./self.alpha*(w-self.alpha*(kx**2+ky**2)))
        return kperp

    def get_ks(self,kx,ky,w):
        kperp = cmath.sqrt(1./self.alpha*(w-self.alpha*(kx**2+ky**2)+self.inner_potential))
        return kperp
