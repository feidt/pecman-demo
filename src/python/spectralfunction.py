import numpy as np

class SpectralFunction(object):
    def __init__(self,evc,gamma):
        self.evc = evc
        self.__gamma = gamma

    def value(self,kx,ky,kz,w,surface):
        """
        return the total sum of the spectral function of the photohole
        """
        kxn, kyn, kzn = self.align_surface(kx,ky,kz,surface)
        energyvalues = self.evc.eigenenergies(kxn,kyn,kzn)

        spec = 0.
        for energy in energyvalues[:]:
            spec += np.imag(self.get_selfenergy(kx,ky,kz,w))/((w-energy-np.real(self.get_selfenergy(kx,ky,kz,w)))**2 + np.imag(self.get_selfenergy(kx,ky,kz,w))**2)
        return spec

    def get_selfenergy(self,kx,ky,kz,w):
        selfenergy = self.__gamma
        return selfenergy

    @staticmethod
    def align_surface(kx,ky,kz,surface):
        if surface == '111':
            kxn = (0.408248290463863* kx - 0.7071067811865475* ky + 0.5773502691896258* kz)
            kyn = (0.40824829046386296* kx + 0.7071067811865476* ky + 0.5773502691896257* kz)
            kzn = 0. - 0.816496580927726* kx + 0.5773502691896257* kz

        elif surface == '110':
            kxn = (0.408248290463863* kx - 0.5773502691896258* ky + 0.7071067811865475* kz)
            kyn = (0.+ 0.816496580927726* kx + 0.5773502691896258* ky)
            kzn = -0.408248290463863* kx + 0.5773502691896258* ky + 0.7071067811865475* kz

        else:
            # for FCC cystals, default: surface = '001'
            kxn = kx
            kyn = ky
            kzn = kz

        return kxn, kyn, kzn

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self,gamma):
        self.__gamma = gamma
