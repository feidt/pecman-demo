import numpy as np
from evc import EVC
from numpy.linalg import norm
import os
from numpy.linalg import eigvals,eigvalsh
from scipy.special import sph_jn,lpn
from numpy import array,pi,dot,isnan
from cmath import sqrt,cos,sin
from sympy.functions.special.tensor_functions import KroneckerDelta


class CombIntSchemeFCC(EVC):

    def __init__(self,parameters):
        self.params = parameters
        self.model_name = "cis_fcc"
        # i do it this way, instead of simply defining the number here, because in the future there should be an
        # option which allows to choose different hamiltonian matrices (only d-states, only s-states, full matrix, ...)
        self.dim = self.eigenenergies(1e-5,1e-5,1e-5).shape[0]

    @property
    def dim(self):
        return self.__dim

    @dim.setter
    def dim(self, dim):
        self.__dim = dim

    def params_str(self):
        """ return a string list with all model parameters """
        """ TO DO: change to new string format style {}.format(...) """

        pstr = []
        pstr.append("{}={}{}".format("ensf", self.format_param(self.params.ensf), os.linesep))
        pstr.append("{}={}{}".format("bohr", self.format_param(self.params.bohr), os.linesep))
        pstr.append("{}={}{}".format("alpha", self.format_param(self.params.alpha), os.linesep))
        pstr.append("{}={}{}".format("S", self.format_param(self.params.S), os.linesep))
        pstr.append("{}={}{}".format("R", self.format_param(self.params.R), os.linesep))
        pstr.append("{}={}{}".format("Bt", self.format_param(self.params.Bt), os.linesep))
        pstr.append("{}={}{}".format("Be", self.format_param(self.params.Be), os.linesep))
        pstr.append("{}={}{}".format("a", self.format_param(self.params.a), os.linesep))

        pstr.append("{}={}{}".format("V000", self.format_param(self.params.V000), os.linesep))
        pstr.append("{}={}{}".format("V111", self.format_param(self.params.V111), os.linesep))
        pstr.append("{}={}{}".format("V200", self.format_param(self.params.V200), os.linesep))
        pstr.append("{}={}{}".format("V220", self.format_param(self.params.V220), os.linesep))
        pstr.append("{}={}{}".format("V311", self.format_param(self.params.V311), os.linesep))
        pstr.append("{}={}{}".format("V222", self.format_param(self.params.V222), os.linesep))
        pstr.append("{}={}{}".format("V331", self.format_param(self.params.V331), os.linesep))
        pstr.append("{}={}{}".format("V400", self.format_param(self.params.V400), os.linesep))
        pstr.append("{}={}{}".format("V420", self.format_param(self.params.V420), os.linesep))

        pstr.append("{}={}{}".format("A1", self.format_param(self.params.A1), os.linesep))
        pstr.append("{}={}{}".format("A2", self.format_param(self.params.A2), os.linesep))
        pstr.append("{}={}{}".format("A3", self.format_param(self.params.A3), os.linesep))
        pstr.append("{}={}{}".format("A4", self.format_param(self.params.A4), os.linesep))
        pstr.append("{}={}{}".format("A5", self.format_param(self.params.A5), os.linesep))
        pstr.append("{}={}{}".format("A6", self.format_param(self.params.A6), os.linesep))

        pstr.append("{}={}{}".format("E0", self.format_param(self.params.E0), os.linesep))
        pstr.append("{}={}{}".format("ED", self.format_param(self.params.ED), os.linesep))

        pstr.append("{}={}".format("xi", self.format_param(self.params.xi)))

        return "".join(pstr)


    def eigenenergies(self,kx,ky,kz):

        eigv = np.real(eigvals(self.HMFLS15(kx,ky,kz,self.params.xi,0.,0.)))
        index_sort = eigv.argsort()
        sorted_eigv = eigv[index_sort]
        return sorted_eigv



    """ Brillouin zone wavevectors for the 15 orthogonalized plane waves (OPWs) """
    def k0(self,kx,ky,kz): return array([kx,ky,kz],dtype=complex)
    def k1(self,kx,ky,kz): return array([kx + self.params.lu, ky + self.params.lu, kz + self.params.lu],dtype=complex)
    def k2(self,kx,ky,kz): return array([kx - self.params.lu, ky + self.params.lu, kz + self.params.lu],dtype=complex)
    def k3(self,kx,ky,kz): return array([kx + self.params.lu, ky - self.params.lu, kz + self.params.lu],dtype=complex)
    def k4(self,kx,ky,kz): return array([kx + self.params.lu, ky + self.params.lu, kz - self.params.lu],dtype=complex)
    def k5(self,kx,ky,kz): return array([kx - self.params.lu, ky - self.params.lu, kz + self.params.lu],dtype=complex)
    def k6(self,kx,ky,kz): return array([kx - self.params.lu, ky + self.params.lu, kz - self.params.lu],dtype=complex)
    def k7(self,kx,ky,kz): return array([kx + self.params.lu, ky - self.params.lu, kz - self.params.lu],dtype=complex)
    def k8(self,kx,ky,kz): return array([kx - self.params.lu, ky - self.params.lu, kz - self.params.lu],dtype=complex)
    def k9(self,kx,ky,kz): return array([kx + 2.*self.params.lu, ky, kz],dtype=complex)
    def k10(self,kx,ky,kz): return array([kx, ky + 2.*self.params.lu, kz],dtype=complex)
    def k11(self,kx,ky,kz): return array([kx, ky, kz + 2.*self.params.lu],dtype=complex)
    def k12(self,kx,ky,kz): return array([kx - 2.*self.params.lu, ky, kz],dtype=complex)
    def k13(self,kx,ky,kz): return array([kx, ky - 2.*self.params.lu, kz],dtype=complex)
    def k14(self,kx,ky,kz): return array([kx, ky, kz - 2.*self.params.lu],dtype=complex)
    def k15(self,kx,ky,kz): return array([kx - 2.*self.params.lu, ky - 2.*self.params.lu, kz],dtype=complex)


    """ hybridization functions """

    # 5-3d real valued spherical harmonics

    def Y21(self,k): return k[0]*k[1]/norm(k)**2
    def Y22(self,k): return k[1]*k[2]/norm(k)**2
    def Y23(self,k): return k[2]*k[0]/norm(k)**2
    def Y24(self,k): return 0.5 * (k[0]**2 - k[1]**2)/norm(k)**2
    def Y25(self,k): return sqrt(3.)/6. * (3.*k[2]**2 - norm(k)**2)/norm(k)**2


    def Hcdi1(self,k): return self.params.Bt*sph_jn(2,norm(k)*self.params.R*self.params.csd1)[0][2]*self.Y21(k*self.params.csd2)
    def Hcdi2(self,k): return self.params.Bt*sph_jn(2,norm(k)*self.params.R*self.params.csd1)[0][2]*self.Y22(k*self.params.csd2)
    def Hcdi3(self,k): return self.params.Bt*sph_jn(2,norm(k)*self.params.R*self.params.csd1)[0][2]*self.Y23(k*self.params.csd2)
    def Hcdi4(self,k): return self.params.Be*sph_jn(2,norm(k)*self.params.R*self.params.csd1)[0][2]*self.Y24(k*self.params.csd2)
    def Hcdi5(self,k): return self.params.Be*sph_jn(2,norm(k)*self.params.R*self.params.csd1)[0][2]*self.Y25(k*self.params.csd2)

    # hamiltonian matrix elements of the hybridization block
    def Hcd01(self,kx,ky,kz): return self.Hcdi1(self.k0(kx,ky,kz))
    def Hcd02(self,kx,ky,kz): return self.Hcdi2(self.k0(kx,ky,kz))
    def Hcd03(self,kx,ky,kz): return self.Hcdi3(self.k0(kx,ky,kz))
    def Hcd04(self,kx,ky,kz): return self.Hcdi4(self.k0(kx,ky,kz))
    def Hcd05(self,kx,ky,kz): return self.Hcdi5(self.k0(kx,ky,kz))

    def Hcd11(self,kx,ky,kz): return self.Hcdi1(self.k1(kx,ky,kz))
    def Hcd12(self,kx,ky,kz): return self.Hcdi2(self.k1(kx,ky,kz))
    def Hcd13(self,kx,ky,kz): return self.Hcdi3(self.k1(kx,ky,kz))
    def Hcd14(self,kx,ky,kz): return self.Hcdi4(self.k1(kx,ky,kz))
    def Hcd15(self,kx,ky,kz): return self.Hcdi5(self.k1(kx,ky,kz))

    def Hcd21(self,kx,ky,kz): return self.Hcdi1(self.k2(kx,ky,kz))
    def Hcd22(self,kx,ky,kz): return self.Hcdi2(self.k2(kx,ky,kz))
    def Hcd23(self,kx,ky,kz): return self.Hcdi3(self.k2(kx,ky,kz))
    def Hcd24(self,kx,ky,kz): return self.Hcdi4(self.k2(kx,ky,kz))
    def Hcd25(self,kx,ky,kz): return self.Hcdi5(self.k2(kx,ky,kz))

    def Hcd31(self,kx,ky,kz): return self.Hcdi1(self.k3(kx,ky,kz))
    def Hcd32(self,kx,ky,kz): return self.Hcdi2(self.k3(kx,ky,kz))
    def Hcd33(self,kx,ky,kz): return self.Hcdi3(self.k3(kx,ky,kz))
    def Hcd34(self,kx,ky,kz): return self.Hcdi4(self.k3(kx,ky,kz))
    def Hcd35(self,kx,ky,kz): return self.Hcdi5(self.k3(kx,ky,kz))

    def Hcd41(self,kx,ky,kz): return self.Hcdi1(self.k4(kx,ky,kz))
    def Hcd42(self,kx,ky,kz): return self.Hcdi2(self.k4(kx,ky,kz))
    def Hcd43(self,kx,ky,kz): return self.Hcdi3(self.k4(kx,ky,kz))
    def Hcd44(self,kx,ky,kz): return self.Hcdi4(self.k4(kx,ky,kz))
    def Hcd45(self,kx,ky,kz): return self.Hcdi5(self.k4(kx,ky,kz))

    def Hcd51(self,kx,ky,kz): return self.Hcdi1(self.k5(kx,ky,kz))
    def Hcd52(self,kx,ky,kz): return self.Hcdi2(self.k5(kx,ky,kz))
    def Hcd53(self,kx,ky,kz): return self.Hcdi3(self.k5(kx,ky,kz))
    def Hcd54(self,kx,ky,kz): return self.Hcdi4(self.k5(kx,ky,kz))
    def Hcd55(self,kx,ky,kz): return self.Hcdi5(self.k5(kx,ky,kz))

    def Hcd61(self,kx,ky,kz): return self.Hcdi1(self.k6(kx,ky,kz))
    def Hcd62(self,kx,ky,kz): return self.Hcdi2(self.k6(kx,ky,kz))
    def Hcd63(self,kx,ky,kz): return self.Hcdi3(self.k6(kx,ky,kz))
    def Hcd64(self,kx,ky,kz): return self.Hcdi4(self.k6(kx,ky,kz))
    def Hcd65(self,kx,ky,kz): return self.Hcdi5(self.k6(kx,ky,kz))

    def Hcd71(self,kx,ky,kz): return self.Hcdi1(self.k7(kx,ky,kz))
    def Hcd72(self,kx,ky,kz): return self.Hcdi2(self.k7(kx,ky,kz))
    def Hcd73(self,kx,ky,kz): return self.Hcdi3(self.k7(kx,ky,kz))
    def Hcd74(self,kx,ky,kz): return self.Hcdi4(self.k7(kx,ky,kz))
    def Hcd75(self,kx,ky,kz): return self.Hcdi5(self.k7(kx,ky,kz))

    def Hcd81(self,kx,ky,kz): return self.Hcdi1(self.k8(kx,ky,kz))
    def Hcd82(self,kx,ky,kz): return self.Hcdi2(self.k8(kx,ky,kz))
    def Hcd83(self,kx,ky,kz): return self.Hcdi3(self.k8(kx,ky,kz))
    def Hcd84(self,kx,ky,kz): return self.Hcdi4(self.k8(kx,ky,kz))
    def Hcd85(self,kx,ky,kz): return self.Hcdi5(self.k8(kx,ky,kz))

    def Hcd91(self,kx,ky,kz): return self.Hcdi1(self.k9(kx,ky,kz))
    def Hcd92(self,kx,ky,kz): return self.Hcdi2(self.k9(kx,ky,kz))
    def Hcd93(self,kx,ky,kz): return self.Hcdi3(self.k9(kx,ky,kz))
    def Hcd94(self,kx,ky,kz): return self.Hcdi4(self.k9(kx,ky,kz))
    def Hcd95(self,kx,ky,kz): return self.Hcdi5(self.k9(kx,ky,kz))

    def Hcd101(self,kx,ky,kz): return self.Hcdi1(self.k10(kx,ky,kz))
    def Hcd102(self,kx,ky,kz): return self.Hcdi2(self.k10(kx,ky,kz))
    def Hcd103(self,kx,ky,kz): return self.Hcdi3(self.k10(kx,ky,kz))
    def Hcd104(self,kx,ky,kz): return self.Hcdi4(self.k10(kx,ky,kz))
    def Hcd105(self,kx,ky,kz): return self.Hcdi5(self.k10(kx,ky,kz))

    def Hcd111(self,kx,ky,kz): return self.Hcdi1(self.k11(kx,ky,kz))
    def Hcd112(self,kx,ky,kz): return self.Hcdi2(self.k11(kx,ky,kz))
    def Hcd113(self,kx,ky,kz): return self.Hcdi3(self.k11(kx,ky,kz))
    def Hcd114(self,kx,ky,kz): return self.Hcdi4(self.k11(kx,ky,kz))
    def Hcd115(self,kx,ky,kz): return self.Hcdi5(self.k11(kx,ky,kz))

    def Hcd121(self,kx,ky,kz): return self.Hcdi1(self.k12(kx,ky,kz))
    def Hcd122(self,kx,ky,kz): return self.Hcdi2(self.k12(kx,ky,kz))
    def Hcd123(self,kx,ky,kz): return self.Hcdi3(self.k12(kx,ky,kz))
    def Hcd124(self,kx,ky,kz): return self.Hcdi4(self.k12(kx,ky,kz))
    def Hcd125(self,kx,ky,kz): return self.Hcdi5(self.k12(kx,ky,kz))

    def Hcd131(self,kx,ky,kz): return self.Hcdi1(self.k13(kx,ky,kz))
    def Hcd132(self,kx,ky,kz): return self.Hcdi2(self.k13(kx,ky,kz))
    def Hcd133(self,kx,ky,kz): return self.Hcdi3(self.k13(kx,ky,kz))
    def Hcd134(self,kx,ky,kz): return self.Hcdi4(self.k13(kx,ky,kz))
    def Hcd135(self,kx,ky,kz): return self.Hcdi5(self.k13(kx,ky,kz))

    def Hcd141(self,kx,ky,kz): return self.Hcdi1(self.k14(kx,ky,kz))
    def Hcd142(self,kx,ky,kz): return self.Hcdi2(self.k14(kx,ky,kz))
    def Hcd143(self,kx,ky,kz): return self.Hcdi3(self.k14(kx,ky,kz))
    def Hcd144(self,kx,ky,kz): return self.Hcdi4(self.k14(kx,ky,kz))
    def Hcd145(self,kx,ky,kz): return self.Hcdi5(self.k14(kx,ky,kz))

    def Hcd151(self,kx,ky,kz): return self.Hcdi1(self.k15(kx,ky,kz))
    def Hcd152(self,kx,ky,kz): return self.Hcdi2(self.k15(kx,ky,kz))
    def Hcd153(self,kx,ky,kz): return self.Hcdi3(self.k15(kx,ky,kz))
    def Hcd154(self,kx,ky,kz): return self.Hcdi4(self.k15(kx,ky,kz))
    def Hcd155(self,kx,ky,kz): return self.Hcdi5(self.k15(kx,ky,kz))


    """
    spin-orbit coupling matrix elements
    """
    # M* = -M, therefore, no complex conjugated M needed


    def M11(self,theta,phi): return 0.
    def M12(self,theta,phi): return 1j*0.5* sin(theta)*sin(phi)
    def M13(self,theta,phi): return -1j*0.5* sin(theta)*cos(phi)
    def M14(self,theta,phi): return 1j*0.5*2.*cos(theta)
    def M15(self,theta,phi): return 0.

    def M21(self,theta,phi): return -1j*0.5*sin(theta)*sin(phi)
    def M22(self,theta,phi): return 0.
    def M23(self,theta,phi): return 1j*0.5*cos(theta)
    def M24(self,theta,phi): return -1j*0.5*sin(theta)*cos(phi)
    def M25(self,theta,phi): return -1j*0.5*3.**(1./3.)*sin(theta)*cos(phi)

    def M31(self,theta,phi): return 1j*0.5*sin(theta)*cos(phi)
    def M32(self,theta,phi): return -1j*0.5*cos(theta)
    def M33(self,theta,phi): return 0.
    def M34(self,theta,phi): return -1j*0.5*sin(theta)*sin(phi)
    def M35(self,theta,phi): return 1j*0.5*3.**(1./3.)*sin(theta)*sin(phi)

    def M41(self,theta,phi): return -1j*0.5*2.*cos(theta)
    def M42(self,theta,phi): return 1j*0.5*sin(theta)*cos(phi)
    def M43(self,theta,phi): return 1j*0.5*sin(theta)*sin(phi)
    def M44(self,theta,phi): return 0.
    def M45(self,theta,phi): return 0.

    def M51(self,theta,phi): return 0.
    def M52(self,theta,phi): return 1j*0.5*3.**(1./3.)*sin(theta)*cos(phi)
    def M53(self,theta,phi): return -1j*0.5*3.**(1./3.)*sin(theta)*sin(phi)
    def M54(self,theta,phi): return 0.
    def M55(self,theta,phi): return 0.



    def N11(self,theta,phi): return 0.
    def N12(self,theta,phi): return 0.5*(cos(phi) + 1j*cos(theta)*sin(phi))
    def N13(self,theta,phi): return 0.5*(sin(phi) - 1j*cos(theta)*cos(phi))
    def N14(self,theta,phi): return -1j*sin(theta)
    def N15(self,theta,phi): return 0.

    def N21(self,theta,phi): return -0.5*(cos(phi) + 1j*cos(theta)*sin(phi))
    def N22(self,theta,phi): return 0.
    def N23(self,theta,phi): return -0.5*sin(theta)
    def N24(self,theta,phi): return 0.5*(sin(phi) - 1j*cos(theta)*cos(phi))
    def N25(self,theta,phi): return 3.**(1./3.)*0.5*(sin(phi) - 1j*cos(theta)*cos(phi))

    def N31(self,theta,phi): return -0.5*(cos(phi) - 1j*cos(theta)*cos(phi))
    def N32(self,theta,phi): return 1j*0.5*sin(theta)
    def N33(self,theta,phi): return 0.
    def N34(self,theta,phi): return -0.5*(cos(phi) + 1j*cos(theta)*sin(phi))
    def N35(self,theta,phi): return 3.**(1./3.)*0.5*(cos(phi) + 1j*cos(theta)*sin(phi))

    def N41(self,theta,phi): return 1j*sin(theta)
    def N42(self,theta,phi): return -0.5*(sin(phi)-1j*cos(theta)*cos(phi))
    def N43(self,theta,phi): return 0.5*(cos(phi) + 1j*cos(theta)*sin(phi))
    def N44(self,theta,phi): return 0.
    def N45(self,theta,phi): return 0.

    def N51(self,theta,phi): return 0.
    def N52(self,theta,phi): return -3.**(1./3.)*0.5*(sin(phi) - 1j*cos(theta)*cos(phi))
    def N53(self,theta,phi): return -3.**(1./3.)*0.5*(cos(phi) + 1j*cos(theta)*sin(phi))
    def N54(self,theta,phi): return 0.
    def N55(self,theta,phi): return 0.


    # complex conjugated N matrix
    def N11c(self,theta,phi): return 0.
    def N12c(self,theta,phi): return 0.5*(cos(phi) - 1j*cos(theta)*sin(phi))
    def N13c(self,theta,phi): return 0.5*(sin(phi) + 1j*cos(theta)*cos(phi))
    def N14c(self,theta,phi): return 1j*sin(theta)
    def N15c(self,theta,phi): return 0.

    def N21c(self,theta,phi): return -0.5*(cos(phi) - 1j*cos(theta)*sin(phi))
    def N22c(self,theta,phi): return 0.
    def N23c(self,theta,phi): return -0.5*sin(theta)
    def N24c(self,theta,phi): return 0.5*(sin(phi) + 1j*cos(theta)*cos(phi))
    def N25c(self,theta,phi): return 3.**(1./3.)*0.5*(sin(phi) + 1j*cos(theta)*cos(phi))

    def N31c(self,theta,phi): return -0.5*(cos(phi) + 1j*cos(theta)*cos(phi))
    def N32c(self,theta,phi): return -1j*0.5*sin(theta)
    def N33c(self,theta,phi): return 0.
    def N34c(self,theta,phi): return -0.5*(cos(phi) - 1j*cos(theta)*sin(phi))
    def N35c(self,theta,phi): return 3.**(1./3.)*0.5*(cos(phi) - 1j*cos(theta)*sin(phi))

    def N41c(self,theta,phi): return -1j*sin(theta)
    def N42c(self,theta,phi): return -0.5*(sin(phi) + 1j*cos(theta)*cos(phi))
    def N43c(self,theta,phi): return 0.5*(cos(phi) - 1j*cos(theta)*sin(phi))
    def N44c(self,theta,phi): return 0.
    def N45c(self,theta,phi): return 0.

    def N51c(self,theta,phi): return 0.
    def N52c(self,theta,phi): return -3.**(1./3.)*0.5*(sin(phi) + 1j*cos(theta)*cos(phi))
    def N53c(self,theta,phi): return -3.**(1./3.)*0.5*(cos(phi) - 1j*cos(theta)*sin(phi))
    def N54c(self,theta,phi): return 0.
    def N55c(self,theta,phi): return 0.



    # sp-band function
    def Hij(self,a,b,ka,kb,V):
        res = self.params.alpha * norm(ka)**2 * KroneckerDelta(a,b) + V - self.params.dV +\
                self.params.S * sph_jn(2,norm(ka) * self.params.R)[0][2] * sph_jn(2,norm(kb) * self.params.R)[0][2] * lpn(2,dot(ka,kb)/(norm(ka)*norm(kb)))[0][2]
        return res

    # d-band sector
    #-------------------------------------------------------------------------------
    def Hdd11(self,kx,ky,kz): return -4.*self.params.A1*cos(0.5*self.params.b*self.params.a*kx) * cos(0.5*self.params.b*self.params.a*ky) + 4.* self.params.A2*(cos(0.5*self.params.b*self.params.a*ky) * cos(0.5*self.params.b*self.params.a*kz) + cos(0.5*self.params.b*self.params.a*kz)*cos(0.5*self.params.b*self.params.a*kx)) + self.params.E0 - self.params.dE
    def Hdd22(self,kx,ky,kz): return -4.*self.params.A1*cos(0.5*self.params.b*self.params.a*ky) * cos(0.5*self.params.b*self.params.a*kz) + 4.* self.params.A2*(cos(0.5*self.params.b*self.params.a*kz) * cos(0.5*self.params.b*self.params.a*kx) + cos(0.5*self.params.b*self.params.a*kx)*cos(0.5*self.params.b*self.params.a*ky)) + self.params.E0 - self.params.dE
    def Hdd33(self,kx,ky,kz): return -4.*self.params.A1*cos(0.5*self.params.b*self.params.a*kz) * cos(0.5*self.params.b*self.params.a*kx) + 4.* self.params.A2*(cos(0.5*self.params.b*self.params.a*kx) * cos(0.5*self.params.b*self.params.a*ky) + cos(0.5*self.params.b*self.params.a*ky)*cos(0.5*self.params.b*self.params.a*kz)) + self.params.E0 - self.params.dE
    def Hdd44(self,kx,ky,kz): return 4.*self.params.A4*cos(0.5*self.params.b*self.params.a*kx) * cos(0.5*self.params.b*self.params.a*ky) - 4.* self.params.A5*(cos(0.5*self.params.b*self.params.a*ky) * cos(0.5*self.params.b*self.params.a*kz) + cos(0.5*self.params.b*self.params.a*kz)*cos(0.5*self.params.b*self.params.a*kx)) + self.params.E0 + 2.*self.params.ED - self.params.dE
    def Hdd55(self,kx,ky,kz): return -4./3.* (self.params.A4 + 4.*self.params.A5)*cos(0.5*self.params.b*self.params.a*kx) * cos(0.5*self.params.b*self.params.a*ky) + 4./3.*(2.* self.params.A4 - self.params.A5)*(cos(0.5*self.params.b*self.params.a*ky) * cos(0.5*self.params.b*self.params.a*kz) + cos(0.5*self.params.b*self.params.a*kz)*cos(0.5*self.params.b*self.params.a*kx)) + self.params.E0 + 2.*self.params.ED - self.params.dE

    def Hdd12(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd21(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd23(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*kx) * sin(0.5*self.params.b*self.params.a*ky)
    def Hdd32(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*kx) * sin(0.5*self.params.b*self.params.a*ky)
    def Hdd31(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd13(self,kx,ky,kz): return -4.*self.params.A3*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd14(self,kx,ky,kz): return 0.
    def Hdd41(self,kx,ky,kz): return 0.
    def Hdd24(self,kx,ky,kz): return -4.*self.params.A6*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd42(self,kx,ky,kz): return -4.*self.params.A6*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd34(self,kx,ky,kz): return 4.*self.params.A6*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd43(self,kx,ky,kz): return 4.*self.params.A6*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd15(self,kx,ky,kz): return -8./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*kx) * sin(0.5*self.params.b*self.params.a*ky)
    def Hdd51(self,kx,ky,kz): return -8./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*kx) * sin(0.5*self.params.b*self.params.a*ky)
    def Hdd25(self,kx,ky,kz): return 4./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd52(self,kx,ky,kz): return 4./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*ky) * sin(0.5*self.params.b*self.params.a*kz)
    def Hdd35(self,kx,ky,kz): return 4./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd53(self,kx,ky,kz): return 4./sqrt(3.)*self.params.A6*sin(0.5*self.params.b*self.params.a*kz) * sin(0.5*self.params.b*self.params.a*kx)
    def Hdd45(self,kx,ky,kz): return 4./sqrt(3.)*(self.params.A4 + self.params.A5)*(cos(0.5*self.params.b*self.params.a*ky)*cos(0.5*self.params.b*self.params.a*kz) - cos(0.5*self.params.b*self.params.a*kz)*cos(0.5*self.params.b*self.params.a*kx))
    def Hdd54(self,kx,ky,kz): return 4./sqrt(3.)*(self.params.A4 + self.params.A5)*(cos(0.5*self.params.b*self.params.a*ky)*cos(0.5*self.params.b*self.params.a*kz) - cos(0.5*self.params.b*self.params.a*kz)*cos(0.5*self.params.b*self.params.a*kx))
    #-------------------------------------------------------------------------------


    """ 3d-band Hamiltonian """
    def HMdd(self,kx,ky,kz):
        return array([\
                    [self.Hdd11(kx,ky,kz),  self.Hdd12(kx,ky,kz),   self.Hdd13(kx,ky,kz),   self.Hdd14(kx,ky,kz),   self.Hdd15(kx,ky,kz)],\
                    [self.Hdd21(kx,ky,kz),  self.Hdd22(kx,ky,kz),   self.Hdd23(kx,ky,kz),   self.Hdd24(kx,ky,kz),   self.Hdd25(kx,ky,kz)],\
                    [self.Hdd31(kx,ky,kz),  self.Hdd32(kx,ky,kz),   self.Hdd33(kx,ky,kz),   self.Hdd34(kx,ky,kz),   self.Hdd35(kx,ky,kz)],\
                    [self.Hdd41(kx,ky,kz),  self.Hdd42(kx,ky,kz),   self.Hdd43(kx,ky,kz),   self.Hdd44(kx,ky,kz),   self.Hdd45(kx,ky,kz)],\
                    [self.Hdd51(kx,ky,kz),  self.Hdd52(kx,ky,kz),   self.Hdd53(kx,ky,kz),   self.Hdd54(kx,ky,kz),   self.Hdd55(kx,ky,kz)]\
                    ])

    """ sp-band Hamiltonian """
    def HMsp(self,kx,ky,kz):
        return array([\
    								[self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200), self.Hij(0,15,self.k0(kx,ky,kz),self.k15(kx,ky,kz),self.params.V220)],\
    								[self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(1,15,self.k1(kx,ky,kz),self.k15(kx,ky,kz),self.params.V331)],\
    								[self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(2,15,self.k2(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311)],\
    								[self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(3,15,self.k3(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311)],\
    								[self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(4,15,self.k4(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311)],\
    								[self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311), self.Hij(5,15,self.k5(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111)],\
    								[self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(6,15,self.k6(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311)],\
    								[self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(7,15,self.k7(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311)],\
    								[self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(8,15,self.k8(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111)],\
    								[self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V400),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(9,15,self.k9(kx,ky,kz),self.k15(kx,ky,kz),self.params.V420)],\
    								[self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V400), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(10,15,self.k10(kx,ky,kz),self.k15(kx,ky,kz),self.params.V420)],\
    								[self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V400),	self.Hij(11,15,self.k11(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222)],\
    								[self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V400), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(12,15,self.k12(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200)],\
    								[self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V400),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(13,15,self.k13(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200)],\
    								[self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V400),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),	self.Hij(14,15,self.k14(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222)],\
    								[self.Hij(15,0,self.k15(kx,ky,kz),self.k0(kx,ky,kz),self.params.V220),	self.Hij(15,1,self.k15(kx,ky,kz),self.k1(kx,ky,kz),self.params.V331), self.Hij(15,2,self.k15(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(15,3,self.k15(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,4,self.k15(kx,ky,kz),self.k4(kx,ky,kz),self.params.V331),	self.Hij(15,5,self.k15(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(15,6,self.k15(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(15,7,self.k15(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,8,self.k15(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(15,9,self.k15(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(15,10,self.k15(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420), self.Hij(15,11,self.k15(kx,ky,kz),self.k11(kx,ky,kz),self.params.V222),\
    								 self.Hij(15,12,self.k15(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(15,13,self.k15(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(15,14,self.k15(kx,ky,kz),self.k14(kx,ky,kz),self.params.V222), self.Hij(15,15,self.k15(kx,ky,kz),self.k15(kx,ky,kz),self.params.V000)]\
    							],dtype=complex)



    #-------------------------------------------------------------------------------
    # spin orbit coupling matrices
    #-------------------------------------------------------------------------------

    # full sp-d matrix
    def HMF(self,kx,ky,kz):
        return array([\
    								[self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200), self.Hij(0,15,self.k0(kx,ky,kz),self.k15(kx,ky,kz),self.params.V220),\
    								 self.Hcd01(kx,ky,kz),	self.Hcd02(kx,ky,kz), self.Hcd03(kx,ky,kz),	self.Hcd04(kx,ky,kz),	self.Hcd05(kx,ky,kz)],\
    								[self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(1,15,self.k1(kx,ky,kz),self.k15(kx,ky,kz),self.params.V331),\
    								 self.Hcd11(kx,ky,kz),	self.Hcd12(kx,ky,kz), self.Hcd13(kx,ky,kz),	self.Hcd14(kx,ky,kz), self.Hcd15(kx,ky,kz)],\
    								[self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(2,15,self.k2(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd21(kx,ky,kz),	self.Hcd22(kx,ky,kz), self.Hcd23(kx,ky,kz),	self.Hcd24(kx,ky,kz), self.Hcd25(kx,ky,kz)],\
    								[self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(3,15,self.k3(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd31(kx,ky,kz),	self.Hcd32(kx,ky,kz), self.Hcd33(kx,ky,kz),	self.Hcd34(kx,ky,kz), self.Hcd35(kx,ky,kz)],\
    								[self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(4,15,self.k4(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd41(kx,ky,kz),	self.Hcd42(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd45(kx,ky,kz)],\
    								[self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311), self.Hij(5,15,self.k5(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd51(kx,ky,kz),	self.Hcd52(kx,ky,kz), self.Hcd53(kx,ky,kz),	self.Hcd54(kx,ky,kz), self.Hcd55(kx,ky,kz)],\
    								[self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(6,15,self.k6(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd61(kx,ky,kz),	self.Hcd62(kx,ky,kz), self.Hcd63(kx,ky,kz),	self.Hcd64(kx,ky,kz), self.Hcd65(kx,ky,kz)],\
    								[self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(7,15,self.k7(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd71(kx,ky,kz),	self.Hcd72(kx,ky,kz), self.Hcd73(kx,ky,kz),	self.Hcd74(kx,ky,kz), self.Hcd75(kx,ky,kz)],\
    								[self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(8,15,self.k8(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd81(kx,ky,kz),	self.Hcd82(kx,ky,kz), self.Hcd83(kx,ky,kz),	self.Hcd84(kx,ky,kz), self.Hcd85(kx,ky,kz)],\
    								[self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V420),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(9,15,self.k9(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd91(kx,ky,kz),	self.Hcd92(kx,ky,kz), self.Hcd93(kx,ky,kz),	self.Hcd94(kx,ky,kz), self.Hcd95(kx,ky,kz)],\
    								[self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V420), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(10,15,self.k10(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd101(kx,ky,kz), self.Hcd102(kx,ky,kz), self.Hcd103(kx,ky,kz), self.Hcd104(kx,ky,kz), self.Hcd105(kx,ky,kz)],\
    								[self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V420),	self.Hij(11,15,self.k11(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd111(kx,ky,kz), self.Hcd112(kx,ky,kz), self.Hcd113(kx,ky,kz), self.Hcd114(kx,ky,kz), self.Hcd115(kx,ky,kz)],\
    								[self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(12,15,self.k12(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd121(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd125(kx,ky,kz)],\
    								[self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(13,15,self.k13(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd131(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd135(kx,ky,kz)],\
    								[self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V420),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),	self.Hij(14,15,self.k14(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd141(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd145(kx,ky,kz)],\
    								[self.Hij(15,0,self.k15(kx,ky,kz),self.k0(kx,ky,kz),self.params.V220),	self.Hij(15,1,self.k15(kx,ky,kz),self.k1(kx,ky,kz),self.params.V331), self.Hij(15,2,self.k15(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(15,3,self.k15(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,4,self.k15(kx,ky,kz),self.k4(kx,ky,kz),self.params.V331),	self.Hij(15,5,self.k15(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(15,6,self.k15(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(15,7,self.k15(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,8,self.k15(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(15,9,self.k15(kx,ky,kz),self.k9(kx,ky,kz),V420), self.Hij(15,10,self.k15(kx,ky,kz),self.k10(kx,ky,kz),V420),	self.Hij(15,11,self.k15(kx,ky,kz),self.k11(kx,ky,kz),self.params.V222),\
    								 self.Hij(15,12,self.k15(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(15,13,self.k15(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(15,14,self.k15(kx,ky,kz),self.k14(kx,ky,kz),self.params.V222), self.Hij(15,15,self.k15(kx,ky,kz),self.k15(kx,ky,kz),self.params.V000),\
    								 self.Hcd151(kx,ky,kz), self.Hcd152(kx,ky,kz), self.Hcd153(kx,ky,kz), self.Hcd154(kx,ky,kz), self.Hcd155(kx,ky,kz)],\
    								[self.Hcd01(kx,ky,kz),	self.Hcd11(kx,ky,kz), self.Hcd21(kx,ky,kz), self.Hcd31(kx,ky,kz), self.Hcd41(kx,ky,kz),	self.Hcd51(kx,ky,kz), self.Hcd61(kx,ky,kz),	self.Hcd71(kx,ky,kz), self.Hcd81(kx,ky,kz),	self.Hcd91(kx,ky,kz), self.Hcd101(kx,ky,kz),\
    								 self.Hcd111(kx,ky,kz), self.Hcd121(kx,ky,kz), self.Hcd131(kx,ky,kz), self.Hcd141(kx,ky,kz), self.Hcd151(kx,ky,kz), self.Hdd11(kx,ky,kz), self.Hdd12(kx,ky,kz), self.Hdd13(kx,ky,kz), self.Hdd14(kx,ky,kz), self.Hdd15(kx,ky,kz)],\
    								[self.Hcd02(kx,ky,kz), self.Hcd12(kx,ky,kz), self.Hcd22(kx,ky,kz), self.Hcd32(kx,ky,kz), self.Hcd42(kx,ky,kz), self.Hcd52(kx,ky,kz), self.Hcd62(kx,ky,kz), self.Hcd72(kx,ky,kz), self.Hcd82(kx,ky,kz), self.Hcd92(kx,ky,kz), self.Hcd102(kx,ky,kz),\
    								 self.Hcd112(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd152(kx,ky,kz), self.Hdd21(kx,ky,kz), self.Hdd22(kx,ky,kz), self.Hdd23(kx,ky,kz), self.Hdd24(kx,ky,kz), self.Hdd25(kx,ky,kz)],\
    								[self.Hcd03(kx,ky,kz), self.Hcd13(kx,ky,kz), self.Hcd23(kx,ky,kz), self.Hcd33(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd53(kx,ky,kz), self.Hcd63(kx,ky,kz), self.Hcd73(kx,ky,kz), self.Hcd83(kx,ky,kz), self.Hcd93(kx,ky,kz), self.Hcd103(kx,ky,kz),\
    								 self.Hcd113(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd153(kx,ky,kz), self.Hdd31(kx,ky,kz), self.Hdd32(kx,ky,kz), self.Hdd33(kx,ky,kz), self.Hdd34(kx,ky,kz), self.Hdd35(kx,ky,kz)],\
    								[self.Hcd04(kx,ky,kz), self.Hcd14(kx,ky,kz), self.Hcd24(kx,ky,kz), self.Hcd34(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd54(kx,ky,kz), self.Hcd64(kx,ky,kz), self.Hcd74(kx,ky,kz), self.Hcd84(kx,ky,kz), self.Hcd94(kx,ky,kz), self.Hcd104(kx,ky,kz),\
    								 self.Hcd114(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd154(kx,ky,kz), self.Hdd41(kx,ky,kz), self.Hdd42(kx,ky,kz), self.Hdd43(kx,ky,kz), self.Hdd44(kx,ky,kz), self.Hdd45(kx,ky,kz)],\
    								[self.Hcd05(kx,ky,kz), self.Hcd15(kx,ky,kz), self.Hcd25(kx,ky,kz), self.Hcd35(kx,ky,kz), self.Hcd45(kx,ky,kz), self.Hcd55(kx,ky,kz), self.Hcd65(kx,ky,kz), self.Hcd75(kx,ky,kz), self.Hcd85(kx,ky,kz), self.Hcd95(kx,ky,kz), self.Hcd105(kx,ky,kz),\
    								 self.Hcd115(kx,ky,kz), self.Hcd125(kx,ky,kz), self.Hcd135(kx,ky,kz), self.Hcd145(kx,ky,kz), self.Hcd155(kx,ky,kz), self.Hdd51(kx,ky,kz), self.Hdd52(kx,ky,kz), self.Hdd53(kx,ky,kz), self.Hdd54(kx,ky,kz), self.Hdd55(kx,ky,kz)]\
    							],dtype=complex)


    """
    **************** with SPIN ORBIT coupling ***********************
    """
    # full sp-d matrix
    def HMFLS(self,kx,ky,kz,xi,theta,phi):
        return array([\
    								[self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200), self.Hij(0,15,self.k0(kx,ky,kz),self.k15(kx,ky,kz),self.params.V220),\
    								 self.Hcd01(kx,ky,kz),	self.Hcd02(kx,ky,kz), self.Hcd03(kx,ky,kz),	self.Hcd04(kx,ky,kz),	self.Hcd05(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(1,15,self.k1(kx,ky,kz),self.k15(kx,ky,kz),self.params.V331),\
    								 self.Hcd11(kx,ky,kz),	self.Hcd12(kx,ky,kz), self.Hcd13(kx,ky,kz),	self.Hcd14(kx,ky,kz), self.Hcd15(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(2,15,self.k2(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd21(kx,ky,kz),	self.Hcd22(kx,ky,kz), self.Hcd23(kx,ky,kz),	self.Hcd24(kx,ky,kz), self.Hcd25(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(3,15,self.k3(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd31(kx,ky,kz),	self.Hcd32(kx,ky,kz), self.Hcd33(kx,ky,kz),	self.Hcd34(kx,ky,kz), self.Hcd35(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(4,15,self.k4(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd41(kx,ky,kz),	self.Hcd42(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd45(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311), self.Hij(5,15,self.k5(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd51(kx,ky,kz),	self.Hcd52(kx,ky,kz), self.Hcd53(kx,ky,kz),	self.Hcd54(kx,ky,kz), self.Hcd55(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(6,15,self.k6(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd61(kx,ky,kz),	self.Hcd62(kx,ky,kz), self.Hcd63(kx,ky,kz),	self.Hcd64(kx,ky,kz), self.Hcd65(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(7,15,self.k7(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd71(kx,ky,kz),	self.Hcd72(kx,ky,kz), self.Hcd73(kx,ky,kz),	self.Hcd74(kx,ky,kz), self.Hcd75(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(8,15,self.k8(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd81(kx,ky,kz),	self.Hcd82(kx,ky,kz), self.Hcd83(kx,ky,kz),	self.Hcd84(kx,ky,kz), self.Hcd85(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V420),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(9,15,self.k9(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd91(kx,ky,kz),	self.Hcd92(kx,ky,kz), self.Hcd93(kx,ky,kz),	self.Hcd94(kx,ky,kz), self.Hcd95(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V420), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(10,15,self.k10(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd101(kx,ky,kz), self.Hcd102(kx,ky,kz), self.Hcd103(kx,ky,kz), self.Hcd104(kx,ky,kz), self.Hcd105(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V420),	self.Hij(11,15,self.k11(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd111(kx,ky,kz), self.Hcd112(kx,ky,kz), self.Hcd113(kx,ky,kz), self.Hcd114(kx,ky,kz), self.Hcd115(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(12,15,self.k12(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd121(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd125(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(13,15,self.k13(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd131(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd135(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V420),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),	self.Hij(14,15,self.k14(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd141(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd145(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(15,0,self.k15(kx,ky,kz),self.k0(kx,ky,kz),self.params.V220),	self.Hij(15,1,self.k15(kx,ky,kz),self.k1(kx,ky,kz),self.params.V331), self.Hij(15,2,self.k15(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(15,3,self.k15(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,4,self.k15(kx,ky,kz),self.k4(kx,ky,kz),self.params.V331),	self.Hij(15,5,self.k15(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(15,6,self.k15(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(15,7,self.k15(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,8,self.k15(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(15,9,self.k15(kx,ky,kz),self.k9(kx,ky,kz),V420), self.Hij(15,10,self.k15(kx,ky,kz),self.k10(kx,ky,kz),V420),	self.Hij(15,11,self.k15(kx,ky,kz),self.k11(kx,ky,kz),self.params.V222),\
    								 self.Hij(15,12,self.k15(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(15,13,self.k15(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(15,14,self.k15(kx,ky,kz),self.k14(kx,ky,kz),self.params.V222), self.Hij(15,15,self.k15(kx,ky,kz),self.k15(kx,ky,kz),self.params.V000),\
    								 self.Hcd151(kx,ky,kz), self.Hcd152(kx,ky,kz), self.Hcd153(kx,ky,kz), self.Hcd154(kx,ky,kz), self.Hcd155(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hcd01(kx,ky,kz),	self.Hcd11(kx,ky,kz), self.Hcd21(kx,ky,kz), self.Hcd31(kx,ky,kz), self.Hcd41(kx,ky,kz),	self.Hcd51(kx,ky,kz), self.Hcd61(kx,ky,kz),	self.Hcd71(kx,ky,kz), self.Hcd81(kx,ky,kz),	self.Hcd91(kx,ky,kz), self.Hcd101(kx,ky,kz),\
    								 self.Hcd111(kx,ky,kz), self.Hcd121(kx,ky,kz), self.Hcd131(kx,ky,kz), self.Hcd141(kx,ky,kz), self.Hcd151(kx,ky,kz),\
                                      self.Hdd11(kx,ky,kz) + xi*self.M11(theta,phi), self.Hdd12(kx,ky,kz) + xi*self.M12(theta,phi), self.Hdd13(kx,ky,kz) + xi*self.M13(theta,phi), self.Hdd14(kx,ky,kz) + xi*self.M14(theta,phi), self.Hdd15(kx,ky,kz) + xi*self.M15(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N11(theta,phi), xi*self.N12(theta,phi), xi*self.N13(theta,phi), xi*self.N14(theta,phi), xi*self.N15(theta,phi)],\
    								[self.Hcd02(kx,ky,kz), self.Hcd12(kx,ky,kz), self.Hcd22(kx,ky,kz), self.Hcd32(kx,ky,kz), self.Hcd42(kx,ky,kz), self.Hcd52(kx,ky,kz), self.Hcd62(kx,ky,kz), self.Hcd72(kx,ky,kz), self.Hcd82(kx,ky,kz), self.Hcd92(kx,ky,kz), self.Hcd102(kx,ky,kz),\
    								 self.Hcd112(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd152(kx,ky,kz),\
                                      self.Hdd21(kx,ky,kz) + xi*self.M21(theta,phi), self.Hdd22(kx,ky,kz) + xi*self.M22(theta,phi), self.Hdd23(kx,ky,kz) + xi*self.M23(theta,phi), self.Hdd24(kx,ky,kz) + xi*self.M24(theta,phi), self.Hdd25(kx,ky,kz) + xi*self.M25(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N21(theta,phi), xi*self.N22(theta,phi), xi*self.N23(theta,phi), xi*self.N24(theta,phi), xi*self.N25(theta,phi)],\
    								[self.Hcd03(kx,ky,kz), self.Hcd13(kx,ky,kz), self.Hcd23(kx,ky,kz), self.Hcd33(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd53(kx,ky,kz), self.Hcd63(kx,ky,kz), self.Hcd73(kx,ky,kz), self.Hcd83(kx,ky,kz), self.Hcd93(kx,ky,kz), self.Hcd103(kx,ky,kz),\
    								 self.Hcd113(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd153(kx,ky,kz),\
                                      self.Hdd31(kx,ky,kz) + xi*self.M31(theta,phi), self.Hdd32(kx,ky,kz) + xi*self.M32(theta,phi), self.Hdd33(kx,ky,kz) + xi*self.M33(theta,phi), self.Hdd34(kx,ky,kz) + xi*self.M34(theta,phi), self.Hdd35(kx,ky,kz) + xi*self.M35(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N31(theta,phi), xi*self.N32(theta,phi), xi*self.N33(theta,phi), xi*self.N34(theta,phi), xi*self.N35(theta,phi)],\
    								[self.Hcd04(kx,ky,kz), self.Hcd14(kx,ky,kz), self.Hcd24(kx,ky,kz), self.Hcd34(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd54(kx,ky,kz), self.Hcd64(kx,ky,kz), self.Hcd74(kx,ky,kz), self.Hcd84(kx,ky,kz), self.Hcd94(kx,ky,kz), self.Hcd104(kx,ky,kz),\
    								 self.Hcd114(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd154(kx,ky,kz),\
                                      self.Hdd41(kx,ky,kz) + xi*self.M41(theta,phi), self.Hdd42(kx,ky,kz) + xi*self.M42(theta,phi), self.Hdd43(kx,ky,kz) + xi*self.M43(theta,phi), self.Hdd44(kx,ky,kz) + xi*self.M44(theta,phi), self.Hdd45(kx,ky,kz) + xi*self.M45(theta,phi),
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N41(theta,phi), xi*self.N42(theta,phi), xi*self.N43(theta,phi), xi*self.N44(theta,phi), xi*self.N45(theta,phi)],\
    								[self.Hcd05(kx,ky,kz), self.Hcd15(kx,ky,kz), self.Hcd25(kx,ky,kz), self.Hcd35(kx,ky,kz), self.Hcd45(kx,ky,kz), self.Hcd55(kx,ky,kz), self.Hcd65(kx,ky,kz), self.Hcd75(kx,ky,kz), self.Hcd85(kx,ky,kz), self.Hcd95(kx,ky,kz), self.Hcd105(kx,ky,kz),\
    								 self.Hcd115(kx,ky,kz), self.Hcd125(kx,ky,kz), self.Hcd135(kx,ky,kz), self.Hcd145(kx,ky,kz), self.Hcd155(kx,ky,kz),\
                                      self.Hdd51(kx,ky,kz) + xi*self.M51(theta,phi), self.Hdd52(kx,ky,kz) + xi*self.M52(theta,phi), self.Hdd53(kx,ky,kz) + xi*self.M53(theta,phi), self.Hdd54(kx,ky,kz) + xi*self.M54(theta,phi), self.Hdd55(kx,ky,kz) + xi*self.M55(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N51(theta,phi), xi*self.N52(theta,phi), xi*self.N53(theta,phi), xi*self.N54(theta,phi), xi*self.N55(theta,phi)],\
                                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200), self.Hij(0,15,self.k0(kx,ky,kz),self.k15(kx,ky,kz),self.params.V220),\
    								 self.Hcd01(kx,ky,kz),	self.Hcd02(kx,ky,kz), self.Hcd03(kx,ky,kz),	self.Hcd04(kx,ky,kz),	self.Hcd05(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(1,15,self.k1(kx,ky,kz),self.k15(kx,ky,kz),self.params.V331),\
    								 self.Hcd11(kx,ky,kz),	self.Hcd12(kx,ky,kz), self.Hcd13(kx,ky,kz),	self.Hcd14(kx,ky,kz), self.Hcd15(kx,ky,kz)],\
    								[ 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.,\
                                     self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(2,15,self.k2(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd21(kx,ky,kz),	self.Hcd22(kx,ky,kz), self.Hcd23(kx,ky,kz),	self.Hcd24(kx,ky,kz), self.Hcd25(kx,ky,kz)],\
                                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),	self.Hij(3,15,self.k3(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd31(kx,ky,kz),	self.Hcd32(kx,ky,kz), self.Hcd33(kx,ky,kz),	self.Hcd34(kx,ky,kz), self.Hcd35(kx,ky,kz)],\
                                     [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.,\
                                     self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(4,15,self.k4(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd41(kx,ky,kz),	self.Hcd42(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd45(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311), self.Hij(5,15,self.k5(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd51(kx,ky,kz),	self.Hcd52(kx,ky,kz), self.Hcd53(kx,ky,kz),	self.Hcd54(kx,ky,kz), self.Hcd55(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(6,15,self.k6(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd61(kx,ky,kz),	self.Hcd62(kx,ky,kz), self.Hcd63(kx,ky,kz),	self.Hcd64(kx,ky,kz), self.Hcd65(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(7,15,self.k7(kx,ky,kz),self.k15(kx,ky,kz),self.params.V311),\
    								 self.Hcd71(kx,ky,kz),	self.Hcd72(kx,ky,kz), self.Hcd73(kx,ky,kz),	self.Hcd74(kx,ky,kz), self.Hcd75(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),	self.Hij(8,15,self.k8(kx,ky,kz),self.k15(kx,ky,kz),self.params.V111),\
    								 self.Hcd81(kx,ky,kz),	self.Hcd82(kx,ky,kz), self.Hcd83(kx,ky,kz),	self.Hcd84(kx,ky,kz), self.Hcd85(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V420),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(9,15,self.k9(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd91(kx,ky,kz),	self.Hcd92(kx,ky,kz), self.Hcd93(kx,ky,kz),	self.Hcd94(kx,ky,kz), self.Hcd95(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V420), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(10,15,self.k10(kx,ky,kz),self.k15(kx,ky,kz),V420),\
    								 self.Hcd101(kx,ky,kz), self.Hcd102(kx,ky,kz), self.Hcd103(kx,ky,kz), self.Hcd104(kx,ky,kz), self.Hcd105(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V420),	self.Hij(11,15,self.k11(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd111(kx,ky,kz), self.Hcd112(kx,ky,kz), self.Hcd113(kx,ky,kz), self.Hcd114(kx,ky,kz), self.Hcd115(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(12,15,self.k12(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd121(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd125(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),	self.Hij(13,15,self.k13(kx,ky,kz),self.k15(kx,ky,kz),self.params.V200),\
    								 self.Hcd131(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd135(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V420),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),	self.Hij(14,15,self.k14(kx,ky,kz),self.k15(kx,ky,kz),self.params.V222),\
    								 self.Hcd141(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd145(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(15,0,self.k15(kx,ky,kz),self.k0(kx,ky,kz),self.params.V220),	self.Hij(15,1,self.k15(kx,ky,kz),self.k1(kx,ky,kz),self.params.V331), self.Hij(15,2,self.k15(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(15,3,self.k15(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,4,self.k15(kx,ky,kz),self.k4(kx,ky,kz),self.params.V331),	self.Hij(15,5,self.k15(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(15,6,self.k15(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(15,7,self.k15(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(15,8,self.k15(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(15,9,self.k15(kx,ky,kz),self.k9(kx,ky,kz),V420), self.Hij(15,10,self.k15(kx,ky,kz),self.k10(kx,ky,kz),V420),	self.Hij(15,11,self.k15(kx,ky,kz),self.k11(kx,ky,kz),self.params.V222),\
    								 self.Hij(15,12,self.k15(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(15,13,self.k15(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(15,14,self.k15(kx,ky,kz),self.k14(kx,ky,kz),self.params.V222), self.Hij(15,15,self.k15(kx,ky,kz),self.k15(kx,ky,kz),self.params.V000),\
    								 self.Hcd151(kx,ky,kz), self.Hcd152(kx,ky,kz), self.Hcd153(kx,ky,kz), self.Hcd154(kx,ky,kz), self.Hcd155(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N11c(theta,phi), -xi*self.N12c(theta,phi), -xi*self.N13c(theta,phi), -xi*self.N14c(theta,phi), -xi*self.N15c(theta,phi),\
                                    self.Hcd01(kx,ky,kz),	self.Hcd11(kx,ky,kz), self.Hcd21(kx,ky,kz), self.Hcd31(kx,ky,kz), self.Hcd41(kx,ky,kz),	self.Hcd51(kx,ky,kz), self.Hcd61(kx,ky,kz),	self.Hcd71(kx,ky,kz), self.Hcd81(kx,ky,kz),	self.Hcd91(kx,ky,kz), self.Hcd101(kx,ky,kz),\
    								 self.Hcd111(kx,ky,kz), self.Hcd121(kx,ky,kz), self.Hcd131(kx,ky,kz), self.Hcd141(kx,ky,kz), self.Hcd151(kx,ky,kz),\
                                      self.Hdd11(kx,ky,kz) - xi*self.M11(theta,phi), self.Hdd12(kx,ky,kz) - xi*self.M12(theta,phi), self.Hdd13(kx,ky,kz) - xi*self.M13(theta,phi), self.Hdd14(kx,ky,kz) - xi*self.M14(theta,phi), self.Hdd15(kx,ky,kz) - xi*self.M15(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N21c(theta,phi), -xi*self.N22c(theta,phi), -xi*self.N23c(theta,phi), -xi*self.N24c(theta,phi), -xi*self.N25c(theta,phi),\
                                    self.Hcd02(kx,ky,kz), self.Hcd12(kx,ky,kz), self.Hcd22(kx,ky,kz), self.Hcd32(kx,ky,kz), self.Hcd42(kx,ky,kz), self.Hcd52(kx,ky,kz), self.Hcd62(kx,ky,kz), self.Hcd72(kx,ky,kz), self.Hcd82(kx,ky,kz), self.Hcd92(kx,ky,kz), self.Hcd102(kx,ky,kz),\
    								 self.Hcd112(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd152(kx,ky,kz),\
                                      self.Hdd21(kx,ky,kz) - xi*self.M21(theta,phi), self.Hdd22(kx,ky,kz) - xi*self.M22(theta,phi), self.Hdd23(kx,ky,kz) - xi*self.M23(theta,phi), self.Hdd24(kx,ky,kz) - xi*self.M24(theta,phi), self.Hdd25(kx,ky,kz) - xi*self.M25(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N31c(theta,phi), -xi*self.N32c(theta,phi), -xi*self.N33c(theta,phi), -xi*self.N34c(theta,phi), -xi*self.N35c(theta,phi),\
                                    self.Hcd03(kx,ky,kz), self.Hcd13(kx,ky,kz), self.Hcd23(kx,ky,kz), self.Hcd33(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd53(kx,ky,kz), self.Hcd63(kx,ky,kz), self.Hcd73(kx,ky,kz), self.Hcd83(kx,ky,kz), self.Hcd93(kx,ky,kz), self.Hcd103(kx,ky,kz),\
    								 self.Hcd113(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd153(kx,ky,kz),\
                                      self.Hdd31(kx,ky,kz) - xi*self.M31(theta,phi), self.Hdd32(kx,ky,kz) - xi*self.M32(theta,phi), self.Hdd33(kx,ky,kz) - xi*self.M33(theta,phi), self.Hdd34(kx,ky,kz) - xi*self.M34(theta,phi), self.Hdd35(kx,ky,kz) - xi*self.M35(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N41c(theta,phi), -xi*self.N42c(theta,phi), -xi*self.N43c(theta,phi), -xi*self.N44c(theta,phi), -xi*self.N45c(theta,phi),\
                                    self.Hcd04(kx,ky,kz), self.Hcd14(kx,ky,kz), self.Hcd24(kx,ky,kz), self.Hcd34(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd54(kx,ky,kz), self.Hcd64(kx,ky,kz), self.Hcd74(kx,ky,kz), self.Hcd84(kx,ky,kz), self.Hcd94(kx,ky,kz), self.Hcd104(kx,ky,kz),\
    								 self.Hcd114(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd154(kx,ky,kz),\
                                      self.Hdd41(kx,ky,kz) - xi*self.M41(theta,phi), self.Hdd42(kx,ky,kz) - xi*self.M42(theta,phi), self.Hdd43(kx,ky,kz) - xi*self.M43(theta,phi), self.Hdd44(kx,ky,kz) - xi*self.M44(theta,phi), self.Hdd45(kx,ky,kz) - xi*self.M45(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N51c(theta,phi), -xi*self.N52c(theta,phi), -xi*self.N53c(theta,phi), -xi*self.N54c(theta,phi), -xi*self.N55c(theta,phi),\
                                    self.Hcd05(kx,ky,kz), self.Hcd15(kx,ky,kz), self.Hcd25(kx,ky,kz), self.Hcd35(kx,ky,kz), self.Hcd45(kx,ky,kz), self.Hcd55(kx,ky,kz), self.Hcd65(kx,ky,kz), self.Hcd75(kx,ky,kz), self.Hcd85(kx,ky,kz), self.Hcd95(kx,ky,kz), self.Hcd105(kx,ky,kz),\
    								 self.Hcd115(kx,ky,kz), self.Hcd125(kx,ky,kz), self.Hcd135(kx,ky,kz), self.Hcd145(kx,ky,kz), self.Hcd155(kx,ky,kz),\
                                      self.Hdd51(kx,ky,kz) - xi*self.M51(theta,phi), self.Hdd52(kx,ky,kz) - xi*self.M52(theta,phi), self.Hdd53(kx,ky,kz) - xi*self.M53(theta,phi), self.Hdd54(kx,ky,kz) - xi*self.M54(theta,phi), self.Hdd55(kx,ky,kz) - xi*self.M55(theta,phi)]
    							],dtype=complex)

    """
    *****************************************************************
    """


    def HMFLS15(self,kx,ky,kz,xi,theta,phi):
        return array([\
    								[self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200),\
    								 self.Hcd01(kx,ky,kz),	self.Hcd02(kx,ky,kz), self.Hcd03(kx,ky,kz),	self.Hcd04(kx,ky,kz),	self.Hcd05(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd11(kx,ky,kz),	self.Hcd12(kx,ky,kz), self.Hcd13(kx,ky,kz),	self.Hcd14(kx,ky,kz), self.Hcd15(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd21(kx,ky,kz),	self.Hcd22(kx,ky,kz), self.Hcd23(kx,ky,kz),	self.Hcd24(kx,ky,kz), self.Hcd25(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd31(kx,ky,kz),	self.Hcd32(kx,ky,kz), self.Hcd33(kx,ky,kz),	self.Hcd34(kx,ky,kz), self.Hcd35(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd41(kx,ky,kz),	self.Hcd42(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd45(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),\
    								 self.Hcd51(kx,ky,kz),	self.Hcd52(kx,ky,kz), self.Hcd53(kx,ky,kz),	self.Hcd54(kx,ky,kz), self.Hcd55(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd61(kx,ky,kz),	self.Hcd62(kx,ky,kz), self.Hcd63(kx,ky,kz),	self.Hcd64(kx,ky,kz), self.Hcd65(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd71(kx,ky,kz),	self.Hcd72(kx,ky,kz), self.Hcd73(kx,ky,kz),	self.Hcd74(kx,ky,kz), self.Hcd75(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd81(kx,ky,kz),	self.Hcd82(kx,ky,kz), self.Hcd83(kx,ky,kz),	self.Hcd84(kx,ky,kz), self.Hcd85(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V420),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd91(kx,ky,kz),	self.Hcd92(kx,ky,kz), self.Hcd93(kx,ky,kz),	self.Hcd94(kx,ky,kz), self.Hcd95(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V420), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd101(kx,ky,kz), self.Hcd102(kx,ky,kz), self.Hcd103(kx,ky,kz), self.Hcd104(kx,ky,kz), self.Hcd105(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V420),\
    								 self.Hcd111(kx,ky,kz), self.Hcd112(kx,ky,kz), self.Hcd113(kx,ky,kz), self.Hcd114(kx,ky,kz), self.Hcd115(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd121(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd125(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd131(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd135(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V420),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),\
    								 self.Hcd141(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd145(kx,ky,kz),\
                                     0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.],\
    								[self.Hcd01(kx,ky,kz),	self.Hcd11(kx,ky,kz), self.Hcd21(kx,ky,kz), self.Hcd31(kx,ky,kz), self.Hcd41(kx,ky,kz),	self.Hcd51(kx,ky,kz), self.Hcd61(kx,ky,kz),	self.Hcd71(kx,ky,kz), self.Hcd81(kx,ky,kz),	self.Hcd91(kx,ky,kz), self.Hcd101(kx,ky,kz),\
    								 self.Hcd111(kx,ky,kz), self.Hcd121(kx,ky,kz), self.Hcd131(kx,ky,kz), self.Hcd141(kx,ky,kz),\
                                      self.Hdd11(kx,ky,kz) + xi*self.M11(theta,phi), self.Hdd12(kx,ky,kz) + xi*self.M12(theta,phi), self.Hdd13(kx,ky,kz) + xi*self.M13(theta,phi), self.Hdd14(kx,ky,kz) + xi*self.M14(theta,phi), self.Hdd15(kx,ky,kz) + xi*self.M15(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N11(theta,phi), xi*self.N12(theta,phi), xi*self.N13(theta,phi), xi*self.N14(theta,phi), xi*self.N15(theta,phi)],\
    								[self.Hcd02(kx,ky,kz), self.Hcd12(kx,ky,kz), self.Hcd22(kx,ky,kz), self.Hcd32(kx,ky,kz), self.Hcd42(kx,ky,kz), self.Hcd52(kx,ky,kz), self.Hcd62(kx,ky,kz), self.Hcd72(kx,ky,kz), self.Hcd82(kx,ky,kz), self.Hcd92(kx,ky,kz), self.Hcd102(kx,ky,kz),\
    								 self.Hcd112(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd142(kx,ky,kz),\
                                      self.Hdd21(kx,ky,kz) + xi*self.M21(theta,phi), self.Hdd22(kx,ky,kz) + xi*self.M22(theta,phi), self.Hdd23(kx,ky,kz) + xi*self.M23(theta,phi), self.Hdd24(kx,ky,kz) + xi*self.M24(theta,phi), self.Hdd25(kx,ky,kz) + xi*self.M25(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N21(theta,phi), xi*self.N22(theta,phi), xi*self.N23(theta,phi), xi*self.N24(theta,phi), xi*self.N25(theta,phi)],\
    								[self.Hcd03(kx,ky,kz), self.Hcd13(kx,ky,kz), self.Hcd23(kx,ky,kz), self.Hcd33(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd53(kx,ky,kz), self.Hcd63(kx,ky,kz), self.Hcd73(kx,ky,kz), self.Hcd83(kx,ky,kz), self.Hcd93(kx,ky,kz), self.Hcd103(kx,ky,kz),\
    								 self.Hcd113(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd143(kx,ky,kz),\
                                      self.Hdd31(kx,ky,kz) + xi*self.M31(theta,phi), self.Hdd32(kx,ky,kz) + xi*self.M32(theta,phi), self.Hdd33(kx,ky,kz) + xi*self.M33(theta,phi), self.Hdd34(kx,ky,kz) + xi*self.M34(theta,phi), self.Hdd35(kx,ky,kz) + xi*self.M35(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N31(theta,phi), xi*self.N32(theta,phi), xi*self.N33(theta,phi), xi*self.N34(theta,phi), xi*self.N35(theta,phi)],\
    								[self.Hcd04(kx,ky,kz), self.Hcd14(kx,ky,kz), self.Hcd24(kx,ky,kz), self.Hcd34(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd54(kx,ky,kz), self.Hcd64(kx,ky,kz), self.Hcd74(kx,ky,kz), self.Hcd84(kx,ky,kz), self.Hcd94(kx,ky,kz), self.Hcd104(kx,ky,kz),\
    								 self.Hcd114(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd144(kx,ky,kz),\
                                      self.Hdd41(kx,ky,kz) + xi*self.M41(theta,phi), self.Hdd42(kx,ky,kz) + xi*self.M42(theta,phi), self.Hdd43(kx,ky,kz) + xi*self.M43(theta,phi), self.Hdd44(kx,ky,kz) + xi*self.M44(theta,phi), self.Hdd45(kx,ky,kz) + xi*self.M45(theta,phi),
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N41(theta,phi), xi*self.N42(theta,phi), xi*self.N43(theta,phi), xi*self.N44(theta,phi), xi*self.N45(theta,phi)],\
    								[self.Hcd05(kx,ky,kz), self.Hcd15(kx,ky,kz), self.Hcd25(kx,ky,kz), self.Hcd35(kx,ky,kz), self.Hcd45(kx,ky,kz), self.Hcd55(kx,ky,kz), self.Hcd65(kx,ky,kz), self.Hcd75(kx,ky,kz), self.Hcd85(kx,ky,kz), self.Hcd95(kx,ky,kz), self.Hcd105(kx,ky,kz),\
    								 self.Hcd115(kx,ky,kz), self.Hcd125(kx,ky,kz), self.Hcd135(kx,ky,kz), self.Hcd145(kx,ky,kz),\
                                      self.Hdd51(kx,ky,kz) + xi*self.M51(theta,phi), self.Hdd52(kx,ky,kz) + xi*self.M52(theta,phi), self.Hdd53(kx,ky,kz) + xi*self.M53(theta,phi), self.Hdd54(kx,ky,kz) + xi*self.M54(theta,phi), self.Hdd55(kx,ky,kz) + xi*self.M55(theta,phi),\
                                      0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                      xi*self.N51(theta,phi), xi*self.N52(theta,phi), xi*self.N53(theta,phi), xi*self.N54(theta,phi), xi*self.N55(theta,phi)],\
                                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(0,0,self.k0(kx,ky,kz),self.k0(kx,ky,kz),self.params.V000),	self.Hij(0,1,self.k0(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111),	self.Hij(0,2,self.k0(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(0,3,self.k0(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,4,self.k0(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(0,5,self.k0(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111),	self.Hij(0,6,self.k0(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(0,7,self.k0(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(0,8,self.k0(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(0,9,self.k0(kx,ky,kz),self.k9(kx,ky,kz),self.params.V200),	self.Hij(0,10,self.k0(kx,ky,kz),self.k10(kx,ky,kz),self.params.V200),	self.Hij(0,11,self.k0(kx,ky,kz),self.k11(kx,ky,kz),self.params.V200),\
    								 self.Hij(0,12,self.k0(kx,ky,kz),self.k12(kx,ky,kz),self.params.V200),	self.Hij(0,13,self.k0(kx,ky,kz),self.k13(kx,ky,kz),self.params.V200), self.Hij(0,14,self.k0(kx,ky,kz),self.k14(kx,ky,kz),self.params.V200),\
    								 self.Hcd01(kx,ky,kz),	self.Hcd02(kx,ky,kz), self.Hcd03(kx,ky,kz),	self.Hcd04(kx,ky,kz),	self.Hcd05(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(1,0,self.k1(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111), self.Hij(1,1,self.k1(kx,ky,kz),self.k1(kx,ky,kz),self.params.V000), self.Hij(1,2,self.k1(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200), self.Hij(1,3,self.k1(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,4,self.k1(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(1,5,self.k1(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(1,6,self.k1(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(1,7,self.k1(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(1,8,self.k1(kx,ky,kz),self.k8(kx,ky,kz),self.params.V222),	self.Hij(1,9,self.k1(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(1,10,self.k1(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(1,11,self.k1(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(1,12,self.k1(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(1,13,self.k1(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(1,14,self.k1(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd11(kx,ky,kz),	self.Hcd12(kx,ky,kz), self.Hcd13(kx,ky,kz),	self.Hcd14(kx,ky,kz), self.Hcd15(kx,ky,kz)],\
    								[ 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.,\
                                     self.Hij(2,0,self.k2(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(2,1,self.k2(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(2,2,self.k2(kx,ky,kz),self.k2(kx,ky,kz),self.params.V000), self.Hij(2,3,self.k2(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(2,4,self.k2(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(2,5,self.k2(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(2,6,self.k2(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(2,7,self.k2(kx,ky,kz),self.k7(kx,ky,kz),self.params.V222),\
    								 self.Hij(2,8,self.k2(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(2,9,self.k2(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(2,10,self.k2(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(2,11,self.k2(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(2,12,self.k2(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(2,13,self.k2(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(2,14,self.k2(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd21(kx,ky,kz),	self.Hcd22(kx,ky,kz), self.Hcd23(kx,ky,kz),	self.Hcd24(kx,ky,kz), self.Hcd25(kx,ky,kz)],\
                                    [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(3,0,self.k3(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(3,1,self.k3(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(3,2,self.k3(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(3,3,self.k3(kx,ky,kz),self.k3(kx,ky,kz),self.params.V000),\
    								 self.Hij(3,4,self.k3(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(3,5,self.k3(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(3,6,self.k3(kx,ky,kz),self.k6(kx,ky,kz),self.params.V222),	self.Hij(3,7,self.k3(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(3,8,self.k3(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(3,9,self.k3(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(3,10,self.k3(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(3,11,self.k3(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(3,12,self.k3(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(3,13,self.k3(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(3,14,self.k3(kx,ky,kz),self.k14(kx,ky,kz),self.params.V311),\
    								 self.Hcd31(kx,ky,kz),	self.Hcd32(kx,ky,kz), self.Hcd33(kx,ky,kz),	self.Hcd34(kx,ky,kz), self.Hcd35(kx,ky,kz)],\
                                     [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                     0.,0.,0.,0.,0.,\
                                     self.Hij(4,0,self.k4(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(4,1,self.k4(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(4,2,self.k4(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(4,3,self.k4(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(4,4,self.k4(kx,ky,kz),self.k4(kx,ky,kz),self.params.V000),	self.Hij(4,5,self.k4(kx,ky,kz),self.k5(kx,ky,kz),self.params.V222), self.Hij(4,6,self.k4(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(4,7,self.k4(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(4,8,self.k4(kx,ky,kz),self.k8(kx,ky,kz),self.params.V220),	self.Hij(4,9,self.k4(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(4,10,self.k4(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(4,11,self.k4(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(4,12,self.k4(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(4,13,self.k4(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(4,14,self.k4(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd41(kx,ky,kz),	self.Hcd42(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd45(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(5,0,self.k5(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(5,1,self.k5(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(5,2,self.k5(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(5,3,self.k5(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(5,4,self.k5(kx,ky,kz),self.k4(kx,ky,kz),self.params.V222),	self.Hij(5,5,self.k5(kx,ky,kz),self.k5(kx,ky,kz),self.params.V000), self.Hij(5,6,self.k5(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(5,7,self.k5(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(5,8,self.k5(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(5,9,self.k5(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(5,10,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(5,11,self.k5(kx,ky,kz),self.k11(kx,ky,kz),self.params.V111),\
    								 self.Hij(5,12,self.k5(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(5,13,self.k5(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(5,14,self.k5(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),\
    								 self.Hcd51(kx,ky,kz),	self.Hcd52(kx,ky,kz), self.Hcd53(kx,ky,kz),	self.Hcd54(kx,ky,kz), self.Hcd55(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(6,0,self.k6(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(6,1,self.k6(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(6,2,self.k6(kx,ky,kz),self.k2(kx,ky,kz),self.params.V200),	self.Hij(6,3,self.k6(kx,ky,kz),self.k3(kx,ky,kz),self.params.V222),\
    								 self.Hij(6,4,self.k6(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(6,5,self.k6(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(6,6,self.k6(kx,ky,kz),self.k6(kx,ky,kz),self.params.V000),	self.Hij(6,7,self.k6(kx,ky,kz),self.k7(kx,ky,kz),self.params.V220),\
    								 self.Hij(6,8,self.k6(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(6,9,self.k6(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(6,10,self.k6(kx,ky,kz),self.k10(kx,ky,kz),self.params.V111),	self.Hij(6,11,self.k6(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(6,12,self.k6(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(6,13,self.k6(kx,ky,kz),self.k13(kx,ky,kz),self.params.V311), self.Hij(6,14,self.k6(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd61(kx,ky,kz),	self.Hcd62(kx,ky,kz), self.Hcd63(kx,ky,kz),	self.Hcd64(kx,ky,kz), self.Hcd65(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(7,0,self.k7(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(7,1,self.k7(kx,ky,kz),self.k1(kx,ky,kz),self.params.V200), self.Hij(7,2,self.k7(kx,ky,kz),self.k2(kx,ky,kz),self.params.V222),	self.Hij(7,3,self.k7(kx,ky,kz),self.k3(kx,ky,kz),self.params.V200),\
    								 self.Hij(7,4,self.k7(kx,ky,kz),self.k4(kx,ky,kz),self.params.V200),	self.Hij(7,5,self.k7(kx,ky,kz),self.k5(kx,ky,kz),self.params.V220), self.Hij(7,6,self.k7(kx,ky,kz),self.k6(kx,ky,kz),self.params.V220),	self.Hij(7,7,self.k7(kx,ky,kz),self.k7(kx,ky,kz),self.params.V000),\
    								 self.Hij(7,8,self.k7(kx,ky,kz),self.k8(kx,ky,kz),self.params.V200),	self.Hij(7,9,self.k7(kx,ky,kz),self.k9(kx,ky,kz),self.params.V111), self.Hij(7,10,self.k7(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(7,11,self.k7(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(7,12,self.k7(kx,ky,kz),self.k12(kx,ky,kz),self.params.V311),	self.Hij(7,13,self.k7(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(7,14,self.k7(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd71(kx,ky,kz),	self.Hcd72(kx,ky,kz), self.Hcd73(kx,ky,kz),	self.Hcd74(kx,ky,kz), self.Hcd75(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(8,0,self.k8(kx,ky,kz),self.k0(kx,ky,kz),self.params.V111),	self.Hij(8,1,self.k8(kx,ky,kz),self.k1(kx,ky,kz),self.params.V222), self.Hij(8,2,self.k8(kx,ky,kz),self.k2(kx,ky,kz),self.params.V220),	self.Hij(8,3,self.k8(kx,ky,kz),self.k3(kx,ky,kz),self.params.V220),\
    								 self.Hij(8,4,self.k8(kx,ky,kz),self.k4(kx,ky,kz),self.params.V220),	self.Hij(8,5,self.k8(kx,ky,kz),self.k5(kx,ky,kz),self.params.V200), self.Hij(8,6,self.k8(kx,ky,kz),self.k6(kx,ky,kz),self.params.V200),	self.Hij(8,7,self.k8(kx,ky,kz),self.k7(kx,ky,kz),self.params.V200),\
    								 self.Hij(8,8,self.k8(kx,ky,kz),self.k8(kx,ky,kz),self.params.V000),	self.Hij(8,9,self.k8(kx,ky,kz),self.k9(kx,ky,kz),self.params.V311), self.Hij(8,10,self.k8(kx,ky,kz),self.k10(kx,ky,kz),self.params.V311),	self.Hij(8,11,self.k8(kx,ky,kz),self.k11(kx,ky,kz),self.params.V311),\
    								 self.Hij(8,12,self.k8(kx,ky,kz),self.k12(kx,ky,kz),self.params.V111),	self.Hij(8,13,self.k8(kx,ky,kz),self.k13(kx,ky,kz),self.params.V111), self.Hij(8,14,self.k8(kx,ky,kz),self.k14(kx,ky,kz),self.params.V111),\
    								 self.Hcd81(kx,ky,kz),	self.Hcd82(kx,ky,kz), self.Hcd83(kx,ky,kz),	self.Hcd84(kx,ky,kz), self.Hcd85(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(9,0,self.k9(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(9,1,self.k9(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(9,2,self.k9(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(9,3,self.k9(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,4,self.k9(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(9,5,self.k9(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(9,6,self.k9(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(9,7,self.k9(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(9,8,self.k9(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(9,9,self.k9(kx,ky,kz),self.k9(kx,ky,kz),self.params.V000), self.Hij(9,10,self.k9(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(9,11,self.k9(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(9,12,self.k9(kx,ky,kz),self.k12(kx,ky,kz),self.params.V420),	self.Hij(9,13,self.k9(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(9,14,self.k9(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd91(kx,ky,kz),	self.Hcd92(kx,ky,kz), self.Hcd93(kx,ky,kz),	self.Hcd94(kx,ky,kz), self.Hcd95(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(10,0,self.k10(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(10,1,self.k10(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(10,2,self.k10(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(10,3,self.k10(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,4,self.k10(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(10,5,self.k10(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(10,6,self.k10(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(10,7,self.k10(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(10,8,self.k10(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(10,9,self.k10(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(10,10,self.k10(kx,ky,kz),self.k10(kx,ky,kz),self.params.V000),	self.Hij(10,11,self.k10(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(10,12,self.k10(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(10,13,self.k10(kx,ky,kz),self.k13(kx,ky,kz),self.params.V420), self.Hij(10,14,self.k10(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd101(kx,ky,kz), self.Hcd102(kx,ky,kz), self.Hcd103(kx,ky,kz), self.Hcd104(kx,ky,kz), self.Hcd105(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(11,0,self.k11(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(11,1,self.k11(kx,ky,kz),self.k1(kx,ky,kz),self.params.V111), self.Hij(11,2,self.k11(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(11,3,self.k11(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(11,4,self.k11(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(11,5,self.k11(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(11,6,self.k11(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(11,7,self.k11(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(11,8,self.k11(kx,ky,kz),self.k8(kx,ky,kz),self.params.V311),	self.Hij(11,9,self.k11(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(11,10,self.k11(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(11,11,self.k11(kx,ky,kz),self.k11(kx,ky,kz),self.params.V000),\
    								 self.Hij(11,12,self.k11(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(11,13,self.k11(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(11,14,self.k11(kx,ky,kz),self.k14(kx,ky,kz),self.params.V420),\
    								 self.Hcd111(kx,ky,kz), self.Hcd112(kx,ky,kz), self.Hcd113(kx,ky,kz), self.Hcd114(kx,ky,kz), self.Hcd115(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(12,0,self.k12(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(12,1,self.k12(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(12,2,self.k12(kx,ky,kz),self.k2(kx,ky,kz),self.params.V111),	self.Hij(12,3,self.k12(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,4,self.k12(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(12,5,self.k12(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(12,6,self.k12(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(12,7,self.k12(kx,ky,kz),self.k7(kx,ky,kz),self.params.V311),\
    								 self.Hij(12,8,self.k12(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(12,9,self.k12(kx,ky,kz),self.k9(kx,ky,kz),self.params.V420), self.Hij(12,10,self.k12(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(12,11,self.k12(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(12,12,self.k12(kx,ky,kz),self.k12(kx,ky,kz),self.params.V000),	self.Hij(12,13,self.k12(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(12,14,self.k12(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd121(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd125(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(13,0,self.k13(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(13,1,self.k13(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(13,2,self.k13(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(13,3,self.k13(kx,ky,kz),self.k3(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,4,self.k13(kx,ky,kz),self.k4(kx,ky,kz),self.params.V311),	self.Hij(13,5,self.k13(kx,ky,kz),self.k5(kx,ky,kz),self.params.V111), self.Hij(13,6,self.k13(kx,ky,kz),self.k6(kx,ky,kz),self.params.V311),	self.Hij(13,7,self.k13(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(13,8,self.k13(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(13,9,self.k13(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(13,10,self.k13(kx,ky,kz),self.k10(kx,ky,kz),self.params.V420),	self.Hij(13,11,self.k13(kx,ky,kz),self.k11(kx,ky,kz),self.params.V220),\
    								 self.Hij(13,12,self.k13(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(13,13,self.k13(kx,ky,kz),self.k13(kx,ky,kz),self.params.V000), self.Hij(13,14,self.k13(kx,ky,kz),self.k14(kx,ky,kz),self.params.V220),\
    								 self.Hcd131(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd135(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    0.,0.,0.,0.,0.,\
                                    self.Hij(14,0,self.k14(kx,ky,kz),self.k0(kx,ky,kz),self.params.V200),	self.Hij(14,1,self.k14(kx,ky,kz),self.k1(kx,ky,kz),self.params.V311), self.Hij(14,2,self.k14(kx,ky,kz),self.k2(kx,ky,kz),self.params.V311),	self.Hij(14,3,self.k14(kx,ky,kz),self.k3(kx,ky,kz),self.params.V311),\
    								 self.Hij(14,4,self.k14(kx,ky,kz),self.k4(kx,ky,kz),self.params.V111),	self.Hij(14,5,self.k14(kx,ky,kz),self.k5(kx,ky,kz),self.params.V311), self.Hij(14,6,self.k14(kx,ky,kz),self.k6(kx,ky,kz),self.params.V111),	self.Hij(14,7,self.k14(kx,ky,kz),self.k7(kx,ky,kz),self.params.V111),\
    								 self.Hij(14,8,self.k14(kx,ky,kz),self.k8(kx,ky,kz),self.params.V111),	self.Hij(14,9,self.k14(kx,ky,kz),self.k9(kx,ky,kz),self.params.V220), self.Hij(14,10,self.k14(kx,ky,kz),self.k10(kx,ky,kz),self.params.V220),	self.Hij(14,11,self.k14(kx,ky,kz),self.k11(kx,ky,kz),self.params.V420),\
    								 self.Hij(14,12,self.k14(kx,ky,kz),self.k12(kx,ky,kz),self.params.V220),	self.Hij(14,13,self.k14(kx,ky,kz),self.k13(kx,ky,kz),self.params.V220), self.Hij(14,14,self.k14(kx,ky,kz),self.k14(kx,ky,kz),self.params.V000),\
    								 self.Hcd141(kx,ky,kz), self.Hcd142(kx,ky,kz), self.Hcd143(kx,ky,kz), self.Hcd144(kx,ky,kz), self.Hcd145(kx,ky,kz)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N11c(theta,phi), -xi*self.N12c(theta,phi), -xi*self.N13c(theta,phi), -xi*self.N14c(theta,phi), -xi*self.N15c(theta,phi),\
                                    self.Hcd01(kx,ky,kz),	self.Hcd11(kx,ky,kz), self.Hcd21(kx,ky,kz), self.Hcd31(kx,ky,kz), self.Hcd41(kx,ky,kz),	self.Hcd51(kx,ky,kz), self.Hcd61(kx,ky,kz),	self.Hcd71(kx,ky,kz), self.Hcd81(kx,ky,kz),	self.Hcd91(kx,ky,kz), self.Hcd101(kx,ky,kz),\
    								 self.Hcd111(kx,ky,kz), self.Hcd121(kx,ky,kz), self.Hcd131(kx,ky,kz), self.Hcd141(kx,ky,kz),\
                                      self.Hdd11(kx,ky,kz) - xi*self.M11(theta,phi), self.Hdd12(kx,ky,kz) - xi*self.M12(theta,phi), self.Hdd13(kx,ky,kz) - xi*self.M13(theta,phi), self.Hdd14(kx,ky,kz) - xi*self.M14(theta,phi), self.Hdd15(kx,ky,kz) - xi*self.M15(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N21c(theta,phi), -xi*self.N22c(theta,phi), -xi*self.N23c(theta,phi), -xi*self.N24c(theta,phi), -xi*self.N25c(theta,phi),\
                                    self.Hcd02(kx,ky,kz), self.Hcd12(kx,ky,kz), self.Hcd22(kx,ky,kz), self.Hcd32(kx,ky,kz), self.Hcd42(kx,ky,kz), self.Hcd52(kx,ky,kz), self.Hcd62(kx,ky,kz), self.Hcd72(kx,ky,kz), self.Hcd82(kx,ky,kz), self.Hcd92(kx,ky,kz), self.Hcd102(kx,ky,kz),\
    								 self.Hcd112(kx,ky,kz), self.Hcd122(kx,ky,kz), self.Hcd132(kx,ky,kz), self.Hcd142(kx,ky,kz),\
                                      self.Hdd21(kx,ky,kz) - xi*self.M21(theta,phi), self.Hdd22(kx,ky,kz) - xi*self.M22(theta,phi), self.Hdd23(kx,ky,kz) - xi*self.M23(theta,phi), self.Hdd24(kx,ky,kz) - xi*self.M24(theta,phi), self.Hdd25(kx,ky,kz) - xi*self.M25(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N31c(theta,phi), -xi*self.N32c(theta,phi), -xi*self.N33c(theta,phi), -xi*self.N34c(theta,phi), -xi*self.N35c(theta,phi),\
                                    self.Hcd03(kx,ky,kz), self.Hcd13(kx,ky,kz), self.Hcd23(kx,ky,kz), self.Hcd33(kx,ky,kz), self.Hcd43(kx,ky,kz), self.Hcd53(kx,ky,kz), self.Hcd63(kx,ky,kz), self.Hcd73(kx,ky,kz), self.Hcd83(kx,ky,kz), self.Hcd93(kx,ky,kz), self.Hcd103(kx,ky,kz),\
    								 self.Hcd113(kx,ky,kz), self.Hcd123(kx,ky,kz), self.Hcd133(kx,ky,kz), self.Hcd143(kx,ky,kz),\
                                      self.Hdd31(kx,ky,kz) - xi*self.M31(theta,phi), self.Hdd32(kx,ky,kz) - xi*self.M32(theta,phi), self.Hdd33(kx,ky,kz) - xi*self.M33(theta,phi), self.Hdd34(kx,ky,kz) - xi*self.M34(theta,phi), self.Hdd35(kx,ky,kz) - xi*self.M35(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N41c(theta,phi), -xi*self.N42c(theta,phi), -xi*self.N43c(theta,phi), -xi*self.N44c(theta,phi), -xi*self.N45c(theta,phi),\
                                    self.Hcd04(kx,ky,kz), self.Hcd14(kx,ky,kz), self.Hcd24(kx,ky,kz), self.Hcd34(kx,ky,kz), self.Hcd44(kx,ky,kz), self.Hcd54(kx,ky,kz), self.Hcd64(kx,ky,kz), self.Hcd74(kx,ky,kz), self.Hcd84(kx,ky,kz), self.Hcd94(kx,ky,kz), self.Hcd104(kx,ky,kz),\
    								 self.Hcd114(kx,ky,kz), self.Hcd124(kx,ky,kz), self.Hcd134(kx,ky,kz), self.Hcd144(kx,ky,kz),\
                                      self.Hdd41(kx,ky,kz) - xi*self.M41(theta,phi), self.Hdd42(kx,ky,kz) - xi*self.M42(theta,phi), self.Hdd43(kx,ky,kz) - xi*self.M43(theta,phi), self.Hdd44(kx,ky,kz) - xi*self.M44(theta,phi), self.Hdd45(kx,ky,kz) - xi*self.M45(theta,phi)],\
    								[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,\
                                    -xi*self.N51c(theta,phi), -xi*self.N52c(theta,phi), -xi*self.N53c(theta,phi), -xi*self.N54c(theta,phi), -xi*self.N55c(theta,phi),\
                                    self.Hcd05(kx,ky,kz), self.Hcd15(kx,ky,kz), self.Hcd25(kx,ky,kz), self.Hcd35(kx,ky,kz), self.Hcd45(kx,ky,kz), self.Hcd55(kx,ky,kz), self.Hcd65(kx,ky,kz), self.Hcd75(kx,ky,kz), self.Hcd85(kx,ky,kz), self.Hcd95(kx,ky,kz), self.Hcd105(kx,ky,kz),\
    								 self.Hcd115(kx,ky,kz), self.Hcd125(kx,ky,kz), self.Hcd135(kx,ky,kz), self.Hcd145(kx,ky,kz),\
                                      self.Hdd51(kx,ky,kz) - xi*self.M51(theta,phi), self.Hdd52(kx,ky,kz) - xi*self.M52(theta,phi), self.Hdd53(kx,ky,kz) - xi*self.M53(theta,phi), self.Hdd54(kx,ky,kz) - xi*self.M54(theta,phi), self.Hdd55(kx,ky,kz) - xi*self.M55(theta,phi)]
    							],dtype=complex)
