import sys
import os
sys.path.append(os.getcwd() + '/src/python')
import numpy as np
import pecman as pm
from pecman import Pecman

# binding energy
be = 1. #1 eV below EFermi
nb_cuts = 100 #number of data files

# photon energy series
# calcuate momentum maps for a single FCC sp band 100x100 px
path = "data"
if not os.path.exists(path):
    os.makedirs(path)
for idp,photon in enumerate(np.linspace(10,40,nb_cuts)):
    pecman = Pecman(pm.SpectralFunction(pm.SimpleTightBindingFCC(E0=5.),1j*0.2),pm.ILEEDState(12.),pm.CWSource(1.,photon))
    data = pecman.mdc(n=100,be=be,surface='111',rot_angle=60,kmax=21.)
    np.savetxt("data/" + str(idp) + ".txt",np.real(data))
