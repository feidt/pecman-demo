import numpy as np
import cmath
from abc import ABCMeta, abstractmethod

# eigenvalue calulator class
class EVC(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def eigenenergies(self,kx,ky,w):
        """ returns a list of eigenvalues of the photohole """
        raise NotImplementedError("Should implement get_evs()!")

    @abstractmethod
    def params_str(self):
        raise NotImplementedError("Should implement save_params()")

    @property
    def dim(self):
        raise NotImplementedError("Your EVC constructor has to define an attribute 'dim'! ")


    @staticmethod
    def format_param(p):
        return "{:5.5f}".format(p)

class SimpleTightBindingFCC(EVC):
    def __init__(self,gamma=-0.35,a=0.41,E0=7.38):
        self.dim = 1
        self.model_name = "SimpleTightBindingFCC"
        self.gamma = gamma
        self.a = a
        self.E0 = E0



    def eigenenergies(self,kx,ky,kz):
        return [self.E0 + 4.*self.gamma*(np.cos(0.5*kx*self.a)*np.cos(0.5*ky*self.a)+np.cos(0.5*kx*self.a)*np.cos(0.5*kz*self.a)+np.cos(0.5*ky*self.a)*np.cos(0.5*kz*self.a))]

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim


    def params_str(self):
        """ return a string list with all model parameters """
        pstr = []
        pstr.append("%s=%s%s" % ("gamma", self.format_param(self.gamma), "\n"))
        pstr.append("%s=%s%s" % ("a", self.format_param(self.a), "\n"))
        pstr.append("%s=%s%s" % ("E0", self.format_param(self.E0), "\n"))
        return "".join(pstr)

class CuO2(EVC):
    def __init__(self,t1=0.5,t2=0.5,a=0.41,E0=0.):

        self.dim = 1
        self.model_name = "Cu02"
        self.t1 = t1
        self.t2 = t2
        self.a = a
        self.E0 = E0

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim


    def eigenenergies(self,kx,ky,kz):
        return [self.E0-2.*self.t1*(np.cos(self.a*kx) + np.cos(self.a*ky))+4.*self.t2*np.cos(self.a*kx)*np.cos(self.a*ky)]

    def params_str(self):
        """ return a string list with all model parameters """
        pstr = []
        pstr.append("%s=%s%s" % ("t1", self.format_param(self.t1), "\n"))
        pstr.append("%s=%s%s" % ("t2", self.format_param(self.t2), "\n"))
        pstr.append("%s=%s%s" % ("a", self.format_param(self.a), "\n"))
        pstr.append("%s=%s%s" % ("E0", self.format_param(self.E0), "\n"))
        return "".join(pstr)



class SimpleTBPeroskite(EVC):
    def __init__(self,t=0.5,a=0.41,E0=0.):
        self.dim = 2
        self.model_name = "SimpleTBPerovskite"
        self.t = t
        self.a = a
        self.E0 = E0


    def eigenenergies(self,kx,ky,kz):
        return [self.E0+2.*self.t*np.sqrt(np.cos(self.a*kx*0.5)**2 + np.cos(self.a*ky*0.5)**2 + np.cos(self.a*kz*0.5)**2),
                self.E0-2.*self.t*np.sqrt(np.cos(self.a*kx*0.5)**2 + np.cos(self.a*ky*0.5)**2 + np.cos(self.a*kz*0.5)**2)]

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim


    def params_str(self):
        """ return a string list with all model parameters """
        pstr = []
        pstr.append("%s=%s%s" % ("t", self.format_param(self.t), "\n"))
        pstr.append("%s=%s%s" % ("a", self.format_param(self.a), "\n"))
        pstr.append("%s=%s%s" % ("E0", self.format_param(self.E0), "\n"))
        return "".join(pstr)




class SimpleTightBindingGraphene(EVC):
    def __init__(self,a=0.426,s0=0.129,gamma=1.,E0=0):
        self.dim = 2
        self.model_name = "SimpleTightBindingGraphene"
        self.a = a
        self.s0 = s0
        self.gamma = gamma
        self.E0 = E0

    def eigenenergies(self,kx,ky,kz):
        fk = np.exp(1j*ky*self.a/np.sqrt(3.)) + 2.*np.exp(-1j*ky*self.a*0.5/np.sqrt(3.))*np.cos(kx*self.a*0.5)
        return [(self.E0+self.gamma*np.abs(fk))/(1.-self.s0*np.abs(fk)), (self.E0-self.gamma*np.abs(fk))/(1.+self.s0*np.abs(fk))]

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim


    def params_str(self):
        """ return a string list with all model parameters """
        pstr = []
        pstr.append("%s=%s%s" % ("gamma", self.format_param(self.gamma), "\n"))
        pstr.append("%s=%s%s" % ("a", self.format_param(self.a), "\n"))
        pstr.append("%s=%s%s" % ("s0", self.format_param(self.s0), "\n"))
        pstr.append("%s=%s%s" % ("E0", self.format_param(self.E0), "\n"))
        return "".join(pstr)
