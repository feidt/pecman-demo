'''***********************************************************************
 * A base implementation of PECMAN: The PhotoEmission Calculation MANager
 * ----------------------------------------------------------------------
 * Author: Martin Feidt
 * Contact: martin.feidt@runbox.com
 * Date: 29.05.2017
 *
 * This demo contains PECMAN's ARPES core for 3d-transition metals with
 * FCC structure
 ***********************************************************************'''

import numpy as np
import sys
import os
from os import getcwd
sys.path.append(getcwd())
from datetime import datetime as dti
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import quad

from spectralfunction import SpectralFunction
from finalstate import ILEEDState
from lightsource import CWSource
from lightsource import Laser

from evc import SimpleTightBindingFCC
from evc import SimpleTightBindingGraphene
from evc import SimpleTBPeroskite
from evc import CuO2
from cisfcc import CombIntSchemeFCC
from cisparameters import CombIntSchemeFCCParameters


FIG_DIM_X = 4
FIG_DIM_Y = 4
YTICK_SIZE = 16
XTICK_SIZE = 16
LABEL_SIZE = 15
LABEL_WEIGHT = 'medium'
TITLE_WEIGHT = 'book'
TITLE_SIZE = 20
CB_TICK_SIZE = 14
CB_MAX_BINS = 3
CB_LABEL_SIZE = 13
CB_LABEL_WEIGHT = 'medium'
TP_FLAG = False



METHOD2 = "kz_integrated"
METHOD1 = "em_conserved"

class Pecman(object):

    def __init__(self,spectral_function,final_state,light_source):
        self.fermi_energy = 6.49
        self.workfunction = 4.5
        self.alpha =  0.0381 #hbar*hbar/2m_electron, [eV nm**2]
        self.spectral_function = spectral_function
        self.final_state = final_state
        self.light_source = light_source
        if self.check_viridis():
            self.cmap = 'viridis'
        else:
            self.cmap = 'hot'

    """
    ================================================================
                        PHOTOEMISSION SIGNAL
    ================================================================
    """

    def pes_kernel(self,x,kx,ky,w,surface):
        """
        integration kernel of the photoemission signal for method = 'em_conserved'
        if the light source has a spectral width, the spectral function is convoluted with the light source:
        --------------------------------------------------------------------------------------------------------
        x:          convolution energy parameter [eV]
        kx,ky:      parallel momentum components [nm^-1]
        w:          kinetic energy [eV]
        --------------------------------------------------------------------------------------------------------
        """
        #integral kernel dx ~ Efield(ekin-x)*Asf(kx,ky,kz,x) dx

        # perpenticular wavevector inside the solid
        kg = np.real(self.final_state.get_ks(kx,ky,w))
        kgi = np.imag(self.final_state.get_ks(kx,ky,w))

        # perpendicular wavevector inside the vacuum
        kd = np.real(self.final_state.get_kv(kx,ky,w))

        # free electron cone * fermi_function * positive kinetic energy
        f0 = self.th(w-self.alpha*(kx**2+ky**2))*self.th( -self.workfunction - x)*self.th(w)

        # preliminary final state imaginary part of the perpendicular momentum
        k2 = 0.1+kgi
        # final contribution (Gr*Ga)
        f1 = kd/((kd+kg)**2+k2**2)*1./(2.*k2)

        # contribution of the light source
        f2 = self.light_source.value(w-x)

        # photohole contribution = spectral function: evaluated at k-perpendicular of the final state
        # inside the solid
        f3 = self.spectral_function.value(kx,ky,kg,x+self.fermi_energy+self.workfunction,surface)

        kernel = f0*f1*np.abs(f2)**2*f3
        return np.real(kernel)


    def pes_kernel_surface_scat(self,x,kx,ky,qx,qy,w,surface):
        """
        integration kernel of the surface scattering photoemission signal for method = 'em_conserved'
        if the light source has a spectral width, the spectral function is convoluted with the light source:
        --------------------------------------------------------------------------------------------------------
        x:          convolution energy parameter [eV]
        kx,ky:      parallel momentum components [nm^-1]
        w:          kinetic energy [eV]
        --------------------------------------------------------------------------------------------------------
        """

        # perpenticular wavevector inside the solid
        kg = np.real(self.final_state.get_ks(kx,ky,w))
        kgi = np.imag(self.final_state.get_ks(kx,ky,w))

        # perpendicular wavevector inside the vacuum
        kd = np.real(self.final_state.get_kv(kx,ky,w))

        # perpenticular wavevector inside the solid before surface scattering
        kmg = np.real(self.final_state.get_ks(kx-qx,ky-qy,w))
        kmgi = np.imag(self.final_state.get_ks(kx-qx,ky-qy,w))

        # free electron cone * fermi_function * positive kinetic energy
        f0 = self.th(w-self.alpha*(kx**2+ky**2))*self.th( -self.workfunction - x)*self.th(w)

        # preliminary final state imaginary part of the perpendicular momentum
        k2 = 0.1+kgi
        # final contribution (Gr*Ga)
        f1 = kd/((kd+kg)**2+k2**2)*1./(2.*k2)

        # contribution of the light source
        f2 = self.light_source.value(w-x)

        # photohole contribution = spectral function: evaluated at k-perpendicular of the final state before surface scattering
        # inside the solid

        f3 = self.spectral_function.value(kx-qx,ky-qy,kmg,x+self.fermi_energy+self.workfunction,surface)

        kernel = f0*f1*np.abs(f2)**2*f3
        return np.real(kernel)



    def pes_kernel_kz_integrated(self,x,kx,ky,kz,w,surface):
        """
        integration kernel of the photoemission signal for method = 'kz_integrated'
        if the light source has a spectral width, the spectral function is convoluted with the light source:
        --------------------------------------------------------------------------------------------------------
        x:          convolution energy parameter [eV]
        kx,ky:      parallel momentum components [nm^-1]
        kz:         perpenticular momentum component [nm^-1]
        w:          kinetic energy [eV]
        --------------------------------------------------------------------------------------------------------
        """
        #integral kernel dx ~ Efield(ekin-x)*Asf(kx,ky,kz,x) dx

        # perpenticular wavevector inside the solid
        kg = np.real(self.final_state.get_ks(kx,ky,w))
        kgi = np.imag(self.final_state.get_ks(kx,ky,w))

        # perpendicular wavevector inside the vacuum
        kd = np.real(self.final_state.get_kv(kx,ky,w))

        # free electron cone * fermi_function * positive kinetic energy
        f0 = self.th(w-self.alpha*(kx**2+ky**2))*self.th( -self.workfunction - x)*self.th(w)

        # preliminary final state imaginary part of the perpendicular momentum
        k2 = 0.1+kgi
        # final contribution (Gr*Ga)
        f1 = kd/((kd+kg)**2+k2**2)*1./(2.*k2)

        # contribution of the light source
        f2 = self.light_source.value(w-x)

        # photohole contribution = spectral function: evaluated at k-perpendicular of the final state
        # inside the solid
        # note: the spectral function is evaluated at kz, not at kg!!
        f3 = self.spectral_function.value(kx,ky,kz,x+self.fermi_energy+self.workfunction,surface)

        # kz-kernel

        kernel = f0*f1*np.abs(f2)**2*f3
        return np.real(kernel)

    def kz_kernel(self,kz,kx,ky,w):
        """
        calculate and return the photoemission kernel function (see. Dissertation M. Piecuch (2017) page 55):
        --------------------------------------------------------------------------------------------------------
        kz:         perpenticular momentum component [nm^-1]
        kx,ky:      parallel momentum components [nm^-1]
        w:          kinetic energy [eV]
        method:     options = ['em_conserved','kz_integrated']
        --------------------------------------------------------------------------------------------------------
        """
        # perpenticular wavevector inside the solid
        ksr = np.real(self.final_state.get_ks(kx,ky,w))
        ksi = np.imag(self.final_state.get_ks(kx,ky,w))

        #add a tiny constant to regularize (avoid division by zero)
        """so far final state has no imaginary part!!!"""
        kl = ksi+1e-10*+ .2#np.sqrt(0.232/self.alpha)
        numerator = kz*kz
        denominator = kl**4 + (ksr**2 - kz**2)**2 + 2.*kl**2*(ksr**2 + kz**2)
        return np.real(numerator/denominator)


    def pes(self,kx,ky,w,method='em_conserved',surface=''):
        """
        calculate and return the photoemission signal:
        --------------------------------------------------------------------------------------------------------
        kx,ky:      parallel momentum components
        w:          kinetic energy
        method:     options = ['em_conserved','kz_integrated']
        --------------------------------------------------------------------------------------------------------
        """
        if method == 'em_conserved':

            # check if light source is laser or cw
            if hasattr(self.light_source,'spectral_width'):
                # evaluate integral only around the current kinetic energy. other regions should give negligible contributions...
                lower_bound = w-self.light_source.central_energy - self.light_source.spectral_width*2.
                upper_bound = w-self.light_source.central_energy + self.light_source.spectral_width*2.
                result = quad(self.pes_kernel,lower_bound,upper_bound,args=(kx,ky,w,surface))[0]

            else:
                result = self.pes_kernel(w-self.light_source.central_energy,kx,ky,w,surface)

        elif method == 'kz_integrated':
            """TO DO: extent to laser intergration"""
            kz_mean = np.real(self.final_state.get_ks(kx,ky,w))
            kz_lower_bound = kz_mean - kz_mean*0.2
            kz_upper_bound = kz_mean + kz_mean*0.2

            # check if light source is laser or cw
            if hasattr(self.light_source,'spectral_width'):
                w_lower_bound = w-self.light_source.central_energy - self.light_source.spectral_width*2.
                w_upper_bound = w-self.light_source.central_energy + self.light_source.spectral_width*2.

                kernel = lambda q: (quad(self.pes_kernel_kz_integrated,w_lower_bound,w_upper_bound,args=(kx,ky,q,w,surface))[0])*self.kz_kernel(q,kx,ky,w)
                result = quad(kernel,kz_lower_bound, kz_upper_bound)[0]

            else:
                kernel = lambda q: self.pes_kernel_kz_integrated(w-self.light_source.central_energy,kx,ky,q,w,surface)*self.kz_kernel(q,kx,ky,w)
                result = quad(kernel,kz_lower_bound, kz_upper_bound)[0]
        else:
            #print("unknown method")
            """ test projected surface density of states calculation """
            result = 0.
            for kz in np.linspace(0.,10.,20):
                result += self.spectral_function.value(kx,ky,kz,w-self.light_source.central_energy+self.fermi_energy+self.workfunction,surface)
                #self.photoemission_kernel2(w-self.light_source.central_energy,kx,ky,kz,w)


        if(np.isnan(result)):
            return -1.
        else:
            return result



    def pes_surface_scat(self,kx,ky,qx,qy,w,method='em_conserved',surface=''):
        """
        calculate and return the photoemission signal:
        --------------------------------------------------------------------------------------------------------
        kx,ky:      parallel momentum components
        w:          kinetic energy
        method:     options = ['em_conserved','kz_integrated']
        --------------------------------------------------------------------------------------------------------
        """
        if method == 'em_conserved':

            # check if light source is laser or cw
            if hasattr(self.light_source,'spectral_width'):
                # evalute integral only around the current kinetic energy. other regions should give negligible contributions...
                # check if integral evaluation point (w dependency) is correct?
                lower_bound = w-self.light_source.central_energy - self.light_source.spectral_width*2.
                upper_bound = w-self.light_source.central_energy + self.light_source.spectral_width*2.
                result = quad(self.pes_kernel_surface_scat,lower_bound,upper_bound,args=(kx,ky,qx,qy,w,surface))[0]

            else:
                result = self.pes_kernel_surface_scat(w-self.light_source.central_energy,kx,ky,qx,qy,w,surface)

        else:
            print("not yet implemented")

        if(np.isnan(result)):
            return -1.
        else:
            return result


    """
    ==============================================================================================================
                                        Momentum Distribution Curves
    ==============================================================================================================
    """

    def mdc_surf_scat(self,n=20,kmax=21.,be=0.1,rot_angle=0.,sdir="",method='em_conserved',surface='',scat_list=None):
        """
        calculate and return a momentum distribution cut (mdc)(aka momentum map) as a function of parallel momentum for a particular kinetic energy (or binding energy):
        --------------------------------------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        be:         binding energy relative to the fermi level (note: for states below the fermi level
                    be is defined to be a positive number!)
        rot_angle:  azimutal (or polar) angle, the effect is to rotate the sample
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        scat_list:  a list of inverse surface lattice vectors
        --------------------------------------------------------------------------------------------------------
        """
        kxaxis = np.linspace(-kmax,kmax,n)
        kyaxis = np.linspace(-kmax,kmax,n)

        spectrum = np.zeros((n,n),dtype = float)
        #use test right after creating spectrum, to return at least an empty matrix
        if isinstance(scat_list, np.ndarray):
            if scat_list.shape[1] == 2:
                N = float(n*n)
                #kinetic energy
                w = self.light_source.central_energy-self.workfunction-be

                #convert from degree to rad
                delta = rot_angle*np.pi/180.

                progress = 0.
                start = dti.now()
                for iky,ky in enumerate(kyaxis):
                    for ikx,kx in enumerate(kxaxis):
                        kxn,kyn = self.polar_rot(kx,ky,delta)
                        signal = 0.
                        for vector in scat_list:
                            signal += self.pes_surface_scat(kxn,kyn,vector[0],vector[1],w,method,surface)
                        spectrum[ikx,iky] = signal

                        progress = self.update_progress((iky*ikx)/N,progress)

                end = dti.now()
                print(self.time_stamp_str() + "  " + "100.00% completed")
                total_time = str(end-start)
                print("total time: " + total_time)

                #save data
                function_tag =  "mdc"
                data_type = ".txt"
                filename = self.generate_filename(sdir, function_tag, data_type)
                if filename != "":
                    # save function parameters and spectrum
                    header = []
                    header.append("{}={:d}{}".format("n",n,os.linesep))
                    header.append("{}={:.2f}{}".format("kmax",kmax,os.linesep))
                    header.append("{}={:.2f}{}".format("be",be,os.linesep))
                    header.append("{}={:.2f}{}".format("rot_angle",rot_angle,os.linesep))
                    header.append("{}={}".format("total time",total_time))
                    np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

                    #save model parameters
                    model_name = self.spectral_function.evc.model_name
                    params = self.spectral_function.evc.params_str()
                    self.save_params_str(sdir,model_name,params)
            else:
                print("superlattice vector dimensional has to be != 2")

        else:
            print("no superlattice vectors given...")
        return spectrum




    def mdc(self,n=20,kmax=21.,be=0.1,rot_angle=0.,method='em_conserved',surface='',sdir=""):
        """
        calculate and return a momentum distribution cut (mdc)(aka momentum map) as a function of parallel momentum for a particular kinetic energy (or binding energy):
        --------------------------------------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        be:         binding energy relative to the fermi level (note: for states below the fermi level
                    be is defined to be a positive number!)
        rot_angle:  azimutal (or polar) angle, the effect is to rotate the sample
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """

        kxaxis = np.linspace(-kmax,kmax,n)
        kyaxis = np.linspace(-kmax,kmax,n)

        spectrum = np.zeros((n,n),dtype = float)
        N = float(n*n)

        #kinetic energy
        w = self.light_source.central_energy-self.workfunction-be

        #convert from degree to rad
        delta = rot_angle*np.pi/180.

        progress = 0.
        start = dti.now()

        for iky,ky in enumerate(kyaxis):
            for ikx,kx in enumerate(kxaxis):
                kxn,kyn = self.polar_rot(kx,ky,delta)
                spectrum[iky,ikx] = self.pes(kxn,kyn,w,method,surface)

                progress = self.update_progress((iky*ikx)/N,progress)

        end = dti.now()
        print('\r' + self.time_stamp_str() + "  " + "100.00% completed")

        total_time = str(end-start)
        print("total time: " + total_time)

        #save data
        function_tag =  "mdc"
        data_type = ".txt"
        filename = self.generate_filename(sdir, function_tag, data_type)

        if filename != "":
            # save function parameters and spectrum
            header = []
            header.append("{}={:d}{}".format("n",n,os.linesep))
            header.append("{}={:.2f}{}".format("kmax",kmax,os.linesep))
            header.append("{}={:.2f}{}".format("be",be,os.linesep))
            header.append("{}={:.2f}{}".format("rot_angle",rot_angle,os.linesep))
            header.append("{}={}{}".format("method",method,os.linesep))
            header.append("{}={}{}".format("surface",surface,os.linesep))
            header.append("{}={}".format("total time",total_time))
            np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

            #save model parameters

            model_name = self.spectral_function.evc.model_name
            params = self.spectral_function.evc.params_str()
            self.save_params_str(sdir,model_name,params)


        return spectrum



    """
    ==============================================================================================================
                                        Energy Distribution Curves
    ==============================================================================================================
    """


    def edc(self,n=50,kmax=21.,be_min=7.,be_max=0.,rot_angle=90.,method='em_conserved',surface='',sdir=""):
        """
        calculate and return an energy distribution cut (edc)(conventional ARPES spectrum) as a function of parallel momentum:
        --------------------------------------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        be_min:     minimum binding energy below the fermi level (note: defined to be a positive number!)
        be_max:     maximum binding energy above the fermi level (also a positive number)
        rot_angle:  azimutal (or polar) angle for the cut
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """
        Ekin_max = self.light_source.central_energy - self.workfunction
        Eaxis = np.linspace(Ekin_max-be_min,Ekin_max+be_max,n)
        kaxis = np.linspace(-kmax,kmax,n)

        spectrum = np.zeros((n,n),dtype = float)
        N = float(n*n)

        #convert from degree to rad
        delta = rot_angle*np.pi/180.

        progress = 0.
        start = dti.now()

        #calulcate spectrum
        for iw,w in enumerate(Eaxis[::-1]):
            for ik,k in enumerate(kaxis):
                kxn,kyn = self.polar_rot(k,0.,delta)
                spectrum[iw,ik] = self.pes(kxn,kyn,w,method,surface)

                #print progress on terminal
                progress = self.update_progress((iw*ik)/N,progress)

        end = dti.now()
        print('\r' + self.time_stamp_str() + "  " + "100.00% completed")
        total_time = str(end-start)
        print("total time: " + total_time)

        #save data
        function_tag =  "edc"
        data_type = ".txt"
        filename = self.generate_filename(sdir, function_tag, data_type)
        if filename != "":
            # save function parameters and spectrum
            header = []
            header.append("{}={:d}{}".format("n",n,os.linesep))
            header.append("{}={:.2f}{}".format("kmax",kmax,os.linesep))
            header.append("{}={:.2f}{}".format("be_min",be_min,os.linesep))
            header.append("{}={:.2f}{}".format("be_max",be_max,os.linesep))
            header.append("{}={:.2f}{}".format("rot_angle",rot_angle,os.linesep))
            header.append("{}={}{}".format("method",method,os.linesep))
            header.append("{}={}{}".format("surface",surface,os.linesep))
            header.append("{}={}".format("total time",total_time))
            np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

            #save model parameters
            model_name = self.spectral_function.evc.model_name
            params = self.spectral_function.evc.params_str()
            self.save_params_str(sdir,model_name,params)

        return spectrum



    def edc_seq_photon(self,n=50,kmax=21.,be_min=7.,be_max=0.,rot_angle=90.,method='em_conserved',surface='',m=1,sigma=1.,sdir=""):
        """

        --------------------------------------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        be_min:     minimum binding energy below the fermi level (note: defined to be a positive number!)
        be_max:     maximum binding energy above the fermi level (also a positive number)
        rot_angle:  azimutal (or polar) angle for the cut
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        m:          number of photon energy values in the
                    range [central_energy-sigma/2, central_energy+sigma/2 ]
        sigma:      photon energy interval in eV
        --------------------------------------------------------------------------------------------------------
        """

        kaxis = np.linspace(-kmax,kmax,n)
        if sigma < 0:
            sigma = -sigma

        # store original central energy, because this has to be changed for the sequence and has to be restored at the end
        central_energy = self.light_source.central_energy
        pe_min = central_energy-sigma*0.5
        pe_max = central_energy+sigma*0.5
        Paxis = np.linspace(pe_min,pe_max,m)

        N = float(n*n)

        #convert from degree to rad
        delta = rot_angle*np.pi/180.

        #save model parameters
        model_name = self.spectral_function.evc.model_name
        params = self.spectral_function.evc.params_str()
        self.save_params_str(sdir,model_name,params)

        #calculate photon sequence
        for ipe, pe in enumerate(Paxis):

            #change the photon energy to modify also the k_perpendicular values
            self.light_source.central_energy = pe

            progress = 0.
            start = dti.now()

            spectrum = np.zeros((n,n),dtype = float)

            #generate energy values based on current photon energy (pe)
            Ekin_max = pe - self.workfunction
            Eaxis = np.linspace(Ekin_max-be_min,Ekin_max+be_max,n)

            #calulcate spectrum
            for iw,w in enumerate(Eaxis[::-1]):
                for ik,k in enumerate(kaxis):
                    kxn,kyn = self.polar_rot(k,0.,delta)
                    spectrum[iw,ik] = self.pes(kxn,kyn,w,method,surface)

                    #print progress on terminal
                    progress = self.update_progress((iw*ik)/N,progress)

            end = dti.now()
            print('\r' + self.time_stamp_str() + "  " + "100.00% completed")
            total_time = str(end-start)
            print("total time: " + total_time)

            #save data
            function_tag =  str(ipe)
            data_type = ".txt"
            filename = self.generate_filename(sdir, function_tag, data_type)
            if filename != "":
                # save function parameters and spectrum
                header = []
                header.append("{}={:d}\n".format("n",n))
                header.append("{}={:.2f}\n".format("kmax",kmax))
                header.append("{}={:.2f}\n".format("be_min",be_min))
                header.append("{}={:.2f}\n".format("be_max",be_max))
                header.append("{}={:.2f}\n".format("rot_angle",rot_angle))
                header.append("{}={}\n".format("method",method))
                header.append("{}={}\n".format("surface",surface))
                header.append("{}={:.2f}\n".format("photon_energy",pe))
                header.append("{}={}".format("total time",total_time))
                np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

        #restore the original photon energy
        self.light_source.central_energy = central_energy


    """
    ============================================================================
                        Constant (Binding) Energy Curves
    ============================================================================
    """


    def cec(self,n=50,kmax=21.,ky=0.,be=0.1,rot_angle=90.,method='em_conserved',surface='',sigma=1.,sdir=""):
        """

        calculate and return a constant (binding) energy curve, i.e. photon energy
        vs. parallel momentum for a fixed binding energy:

        ------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        ky:         ky momentum in nm^-1
        be:         binding energy below the fermi level (note: defined
                    to be a positive number!
        rot_angle:  azimutal (or polar) angle for the cut
        m:          number of photon energy values in the
                    range [central_energy-sigma/2, central_energy+sigma/2 ]
        sigma:      photon energy interval in eV
        sdir:       name of the save directory, if sdir == "", nothing will be
                    saved
        ------------------------------------------------------------------------
        """
        #Ekin_max = self.light_source.central_energy - self.workfunction
        #Eaxis = np.linspace(Ekin_max-be_min,Ekin_max+be_max,n)
        kaxis = np.linspace(-kmax,kmax,n)

        if sigma < 0:
            sigma = -sigma

        # store original central energy, because this has to be changed for the sequence and has to be restored at the end
        central_energy = self.light_source.central_energy
        pe_min = central_energy-sigma*0.5
        pe_max = central_energy+sigma*0.5
        Paxis = np.linspace(pe_min,pe_max,n)

        spectrum = np.zeros((n,n),dtype = float)
        N = float(n*n)

        #convert from degree to rad
        delta = rot_angle*np.pi/180.

        progress = 0.
        start = dti.now()


        #calulcate spectrum
        for iw,w in enumerate(Paxis[::-1]):

            #change the photon energy to modify also the k_perpendicular values
            self.light_source.central_energy = w
            binding_energy = w - self.workfunction - be
            for ik,k in enumerate(kaxis):
                kxn,kyn = self.polar_rot(k,ky,delta)
                spectrum[iw,ik] = self.pes(kxn,kyn,binding_energy,method,surface)

                #print progress on terminal
                progress = self.update_progress((iw*ik)/N,progress)

        end = dti.now()
        print('\r' + self.time_stamp_str() + "  " + "100.00% completed")
        total_time = str(end-start)
        print("total time: " + total_time)

        #save data
        function_tag =  "pdc"
        data_type = ".txt"
        filename = self.generate_filename(sdir, function_tag, data_type)
        if filename != "":
            # save function parameters and spectrum
            header = []
            header.append("{}={:d}{}".format("n",n,os.linesep))
            header.append("{}={:.2f}{}".format("kmax",kmax,os.linesep))
            header.append("{}={:.2f}{}".format("be",be,os.linesep))
            header.append("{}={:.2f}{}".format("rot_angle",rot_angle,os.linesep))
            header.append("{}={}{}".format("method",method,os.linesep))
            header.append("{}={}{}".format("surface",surface,os.linesep))
            header.append("{}={}".format("total time",total_time))
            np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

            #save model parameters
            model_name = self.spectral_function.evc.model_name
            params = self.spectral_function.evc.params_str()
            self.save_params_str(sdir,model_name,params)

        #restore the original photon energy
        self.light_source.central_energy = central_energy

        return spectrum


    """
    ============================================================================
                        Constant Momentum Curves
    ============================================================================
    """


    def cmc(self,n=50,kx=0.,ky=0.,be=0.1,rot_angle=0.,method='em_conserved',surface='',sigma=1.,sdir=""):
        """
        calculate and return a constant (binding) energy curve, i.e. photon energy
        vs. parallel momentum for a fixed binding energy:

        ------------------------------------------------------------------------
        n:          integer resolution
        kmax:       maximum k_parallel value in nm^-1
        kx:         kx momentum in nm^-1
        ky:         ky momentum in nm^-1
        be_min:     minimum binding energy below the fermi level (note: defined to be a positive number!)
        be_max:     maximum binding energy above the fermi level (also a positive number)
        rot_angle:  azimutal (or polar) angle for the cut
        m:          number of photon energy values in the
                    range [central_energy-sigma/2, central_energy+sigma/2 ]
        sigma:      photon energy interval in eV
        sdir:       name of the save directory, if sdir == "", nothing will be
                    saved
        ------------------------------------------------------------------------
        """

        #Ekin_max = self.light_source.central_energy - self.workfunction
        #Eaxis = np.linspace(Ekin_max-be_min,Ekin_max+be_max,n)
        kaxis = np.linspace(-kmax,kmax,n)

        if sigma < 0:
            sigma = -sigma

        # store original central energy, because this has to be changed for the sequence and has to be restored at the end
        central_energy = self.light_source.central_energy
        pe_min = central_energy-sigma*0.5
        pe_max = central_energy+sigma*0.5
        Paxis = np.linspace(pe_min,pe_max,m)

        spectrum = np.zeros((n,n),dtype = float)
        N = float(n*n)

        #convert from degree to rad
        delta = rot_angle*np.pi/180.

        progress = 0.
        start = dti.now()


        #calulcate spectrum
        for iw,w in enumerate(Paxis[::-1]):

            #change the photon energy to modify also the k_perpendicular values
            self.light_source.central_energy = w
            binding_energy = w - self.workfunction - be
            for ik,k in enumerate(kaxis):
                kxn,kyn = self.polar_rot(k,ky,delta)
                spectrum[iw,ik] = self.pes(kxn,kyn,binding_energy,method,surface)

                #print progress on terminal
                progress = self.update_progress((iw*ik)/N,progress)

        end = dti.now()
        print('\r' + self.time_stamp_str() + "  " + "100.00% completed")
        total_time = str(end-start)
        print("total time: " + total_time)

        #save data
        function_tag =  "pdc"
        data_type = ".txt"
        filename = self.generate_filename(sdir, function_tag, data_type)
        if filename != "":
            # save function parameters and spectrum
            header = []
            header.append("{}={:d}{}".format("n",n,os.linesep))
            header.append("{}={:.2f}{}".format("kmax",kmax,os.linesep))
            header.append("{}={:.2f}{}".format("be",be,os.linesep))
            header.append("{}={:.2f}{}".format("rot_angle",rot_angle,os.linesep))
            header.append("{}={}{}".format("method",method,os.linesep))
            header.append("{}={}{}".format("surface",surface,os.linesep))
            header.append("{}={}".format("total time",total_time))
            np.savetxt(filename, spectrum, fmt='%1.5e', header = "".join(header))

            #save model parameters
            model_name = self.spectral_function.evc.model_name
            params = self.spectral_function.evc.params_str()
            self.save_params_str(sdir,model_name,params)

        #restore the original photon energy
        self.light_source.central_energy = central_energy

        return spectrum


    """
    ==============================================================================================================
                                                PLOTTING
    ==============================================================================================================
    """

    def mdc_plot(self,data,kmax=21.,sdir=""):
        """
        plot an energy distribution cut (edc)(conventional ARPES spectrum) and save the figure optionally:
        --------------------------------------------------------------------------------------------------------
        data:       2D MDC spectrum
        kmax:       maximum k_parallel value in nm^-1
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """
        fig = plt.figure(figsize=(FIG_DIM_X, FIG_DIM_Y))
        mp = kmax*0.1
        #plt.contour(data,extent=[-mp,mp,-mp,mp])
        plt.imshow(data,extent=[-mp,mp,-mp,mp],cmap=self.cmap)
        plt.xlabel("k$_\mathsf{x}$ [$\AA^{-1}$]",fontsize=LABEL_SIZE)
        plt.ylabel("k$_\mathsf{y}$ [$\AA^{-1}$]",fontsize=LABEL_SIZE)

        #save figure
        function_tag =  "mdc"
        data_type = ".png"
        filename = self.generate_filename(sdir, function_tag, data_type)
        if filename != "":
            fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=TP_FLAG)

        plt.show()


    def edc_plot(self,data,kmax=21.,be_min=7,be_max=0,sdir=""):
        """
        plot an energy distribution cut (edc)(conventional ARPES spectrum) and save the figure optionally:
        --------------------------------------------------------------------------------------------------------
        data:       2D EDC spectrum
        kmax:       maximum k_parallel value in nm^-1
        be_min:     minimum binding energy below the fermi level (note: defined to be a positive number!)
        be_max:     maximum binding energy above the fermi level (also a positive number)
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """
        fig = plt.figure(figsize=(FIG_DIM_X, FIG_DIM_Y))
        mp = kmax*0.1
        #plt.contour(data,extent=[-mp,mp,-mp,mp])
        plt.imshow(data,extent=[-mp,mp,-be_min,be_max],cmap=self.cmap)
        plt.xlabel("k$_\mathsf{||}$ [$\AA^{-1}$]",fontsize=LABEL_SIZE)
        plt.ylabel("E-E$_\mathsf{F}$ [eV]",fontsize=LABEL_SIZE)

        #save figure
        function_tag = "edc"
        data_type = ".png"
        filename = self.generate_filename(sdir, function_tag, data_type)
        if filename != "":
            fig.savefig(filename, bbox_inches='tight', pad_inches=0, transparent=TP_FLAG)

        plt.show()


    @classmethod
    def generate_filename(cls,sdir,function_tag,data_type):
        """ check and create directory and return a filename """

        """ TO DO: to be Linux and Windows compatible you have to use os.path.join instead of "/" !!!!!!!!!!! """
        if sdir != "":
            path = str(getcwd() + "/" + sdir)
            filename = str(path + "/" + function_tag + data_type)
            if not os.path.exists(path):
                os.makedirs(path)
            else:
                #if the directory already exists, check if the filename already exists. if so, append a time stamp on the filename
                if os.path.exists(filename):
                    filename = str(getcwd() + "/" + sdir) + "/" + function_tag + "_" + cls.time_stamp_str() + data_type
            return filename
        else:
             return ""


    """
    ==============================================================================================================
                                                        support
    ==============================================================================================================
    """

    @staticmethod
    def th(x):
        """ Heaviside step function """
        return (1. if x > 0. else 0.)

    @staticmethod
    def nf(x,T):
        """
        fermi distribution
        --------------------------------------
        x:          energy [eV]
        T:          temperature [K]
        --------------------------------------
        """
        kb = 8.6173303e-5 # [eV/K]
        beta = 1./(T*kb)
        return 1./(np.exp(beta*x)+1.)

    @staticmethod
    def polar_rot(kx,ky,angle):
        return kx*np.cos(angle)-ky*np.sin(angle), kx*np.sin(angle)+ky*np.cos(angle)


    def save_params_str(self,sdir,model_name,params_str):
        """ TO DO: to be Linux and Windows compatible you have to use os.path.join instead of / """
        filename = str(getcwd() + "/" + sdir + "/" + model_name + ".txt")
        if os.path.exists(filename):
            filename = str(str(getcwd() + "/" + sdir) + "/" + model_name + "_" + self.time_stamp_str() + ".txt")

        params_file = open(filename, "w")
        params_file.write("{}={:.2f}\n".format("fermi_energy",self.fermi_energy))
        params_file.write("{}={:.2f}\n".format("workfunction",self.workfunction))
        params_file.write(params_str)
        params_file.close()
    #    else:

    @classmethod
    def update_progress(cls,n,progress):
        """ print percental progress, used in edc(), mdc(), ..."""
        if(n >= progress):
            os.system('clear')
            #sys.stdout.write('PECMAN 1.0 running simulation...\n--------------------------------\n')
            sys.stdout.write("\r")
            sys.stdout.write(cls.time_stamp_str() + "  " + "{:2.2f}".format(progress*100) + "% completed");
            #flush command gives error when module is used with MATLAB
            #sys.stdout.flush()
            progress = progress + 0.01
        return progress



    # available since matplotlib 1.5 (or so)
    @staticmethod
    def check_viridis():
        """ check if viridis colormap is available and return a boolean"""
        awesome = False
        for cmap in plt.colormaps():
            if cmap == 'viridis':
                awesome = True
        return awesome

    @classmethod
    def time_stamp_str(cls):
        """ return datetime without milliseconds """
        return dti.now().strftime("%Y-%m-%d_%H:%M:%S")



    """
    ==============================================================================================================
                            BAND STRUCTURE FUNCTION (only for the combined interpolation scheme)
    ==============================================================================================================
    """

    def bandstructure_overview(self,dk=0.4,sdir=""):
        """
        calculate the bandstructure along high symmetry points (only for the combined interpolation scheme model): G-X-W-L-G-K, as in Laesser, Smith PRB 24 (1981)
        --------------------------------------------------------------------------------------------------------
        dk:         k-resolution [nm^-1], e.g. (2 pi/a) / 100 ~ 0.15
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """
        if self.spectral_function.evc.__class__.__name__ == "CombIntSchemeFCC":
            tiny = 1e-12
            #note: for the simple tight binding models, the lattice parameter is directly implemented in EVC, for the CIS it's in the parameters class
            lattice_param = self.spectral_function.evc.params.a
            bzp = np.pi/lattice_param

            l_gx = 2.*bzp
            l_xw = bzp
            l_wl = np.sqrt(2.)*bzp
            l_lg = np.sqrt(3.)*bzp
            l_gk = 3.*np.sqrt(1./2.)*bzp

            n_gx = int(l_gx/dk)
            n_xw = int(l_xw/dk)
            n_wl = int(l_wl/dk)
            n_lg = int(l_lg/dk)
            n_gk = int(l_gk/dk)

            band_dim = self.spectral_function.evc.dim
            k_dim = n_gx + n_xw + n_wl + n_lg + n_gk

            bandstructure = np.zeros((band_dim, k_dim))

            """ G-X """
            for ik, k in enumerate(np.linspace(0., 2.*bzp,n_gx)):
                try:
                    eigv = self.spectral_function.evc.eigenenergies(tiny, tiny+k, tiny)
                except:
                    eigv = np.zeros(band_dim)

                for idev,ev in enumerate(eigv):
                    bandstructure[idev,ik] = ev - self.fermi_energy

            """ X-W """
            for ik, k in enumerate(np.linspace(0., bzp, n_xw)):
                try:
                    eigv = self.spectral_function.evc.eigenenergies(tiny+k, 2.*bzp, tiny)
                except:
                    eigv = np.zeros(band_dim)

                for idev,ev in enumerate(eigv):
                    bandstructure[idev,ik+n_gx] = ev - self.fermi_energy

            """ W-L """
            for ik, k in enumerate(np.linspace(0., bzp, n_wl)):
                try:
                    eigv = self.spectral_function.evc.eigenenergies(bzp, 2.*bzp-k, tiny+k)
                except:
                    eigv = np.zeros(band_dim)

                for idev,ev in enumerate(eigv):
                    bandstructure[idev,ik+n_gx+n_xw] = ev - self.fermi_energy

            """ L-G """
            for ik, k in enumerate(np.linspace(0., bzp, n_lg)):
                try:
                    eigv = self.spectral_function.evc.eigenenergies(bzp-k + tiny, bzp-k + tiny, bzp-k+tiny)
                except:
                    eigv = np.zeros(band_dim)

                for idev,ev in enumerate(eigv):
                    bandstructure[idev,ik + n_gx + n_xw + n_wl] = ev - self.fermi_energy

            """ G-K """
            for ik, k in enumerate(np.linspace(0., 1.5*bzp, n_gk)):
                try:
                    eigv = self.spectral_function.evc.eigenenergies(tiny + k, tiny+k, tiny)
                except:
                    eigv = np.zeros(band_dim)

                for idev,ev in enumerate(eigv):
                    bandstructure[idev,ik + n_gx + n_xw + n_wl + n_lg] = ev - self.fermi_energy

            return bandstructure

        else:
            print("at present only defined for CombIntSchemeFCC()")
            return None


    def bandstructure_overview_plot(self, bandstructure, dk, be_min=7., be_max=0.):
        """
        plot and save a bandstructure overview calculation along the path G-X-W-L-G-K, as in Laesser, Smith PRB 24 (1981)
        --------------------------------------------------------------------------------------------------------
        dk:         k-resolution [nm^-1], e.g. (2 pi/a) / 100 ~ 0.15
        be_min:     minimum binding energy below the fermi level (note: defined to be a positive number!)
        be_max:     maximum binding energy above the fermi level (also a positive number)
        sdir:       name of the save directory, if sdir == "", nothing will be saved
        --------------------------------------------------------------------------------------------------------
        """
        tiny = 1e-12
        #note: for the simple tight binding models, the lattice parameter is directly implemented in EVC, for the CIS it`s in the parameters class
        lattice_param = self.spectral_function.evc.params.a
        bzp = np.pi/lattice_param

        l_gx = 2.*bzp
        l_xw = bzp
        l_wl = np.sqrt(2.)*bzp
        l_lg = np.sqrt(3.)*bzp
        l_gk = 3.*np.sqrt(1./2.)*bzp

        n_gx = int(l_gx/dk)
        n_xw = int(l_xw/dk)
        n_wl = int(l_wl/dk)
        n_lg = int(l_lg/dk)
        n_gk = int(l_gk/dk)

        #band_dim = self.spectral_function.evc.dim
        k_dim = n_gx + n_xw + n_wl + n_lg + n_gk

        plt.figure(figsize=(10,6))
        BOUNDARY_COLOR = "gray"
        BAND_COLOR = "darkblue"
        EDGE_COLOR = "darkblue"
        POINT_SIZE = 2.
        PLOT_STYLE = "o"


        for idband, band in enumerate(bandstructure):
            plt.plot(band,lw=1.5,c="r")

        wid = 2.
        plt.vlines(n_gx,-be_min,be_max,color = BOUNDARY_COLOR,lw=wid)
        plt.vlines(n_xw+n_gx,-be_min,be_max,color = BOUNDARY_COLOR,lw=wid)
        plt.vlines(n_wl+n_xw+n_gx,-be_min,be_max,color = BOUNDARY_COLOR,lw=wid)
        plt.vlines(n_lg+n_wl+n_xw+n_gx,-be_min,be_max,color = BOUNDARY_COLOR,lw=wid)

        plt.xlim(0,k_dim)
        plt.ylim(-be_min, be_max)

        plt.ylabel('E-E$_\mathsf{_F}$ [eV]',fontsize=16,weight='medium')
        plt.tick_params(axis='both', which='major', labelsize=16)
        #title(plottitle,fontsize = TITLE_SIZE,weight=TITLE_WEIGHT)
        plt.xlabel("Reduced wave vector",fontsize = 16,weight='medium')

        plt.xticks([0,n_gx, n_gx+n_xw, n_gx+n_xw+n_wl, n_gx+n_xw+n_wl+\
                    n_lg,n_gx+n_xw+n_wl+n_lg+n_gk],\
                   [r'$\Gamma$',r"$X$", r"$W$", r"$L$",r"$\Gamma$",r"$K$"])

        plt.show()
