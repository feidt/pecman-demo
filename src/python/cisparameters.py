import numpy as np

class CombIntSchemeFCCParameters(object):

    def __init__(self):

        self.ensf = 1.

        self.bohr = 0.52917721092
        self.Ry = 13.605698066

        self.alpha = 0.01050*self.Ry *self.bohr*self.bohr
        self.S = 1.430*self.Ry
        self.R = 0.294 *self.bohr
        self.Bt = 1.442*self.Ry
        self.Be = 1.442*self.Ry
        self.a = 0.40855 # [nm]

        self.V000 = -0.05254*self.Ry
        self.V111 = 0.05123*self.Ry
        self.V200 = 0.04838*self.Ry
        self.V220 = 0.00101*self.Ry
        self.V311 = 0.08484*self.Ry
        self.V222 = 0.12428*self.Ry
        self.V331 = 0.
        self.V400 = 0.
        self.V420 = 0.

        self.A1 =  0.01114*self.Ry
        self.A2 = 0.00215*self.Ry
        self.A3 = 0.00547*self.Ry
        self.A4 = 0.00663*self.Ry
        self.A5 = 0.00164*self.Ry
        self.A6 = 0.00438*self.Ry
        self.E0 = 0.1070*self.Ry
        self.ED = -0.0039*self.Ry

        self.xi = 0.02*self.Ry

        self.lu = 2.*np.pi/self.a

        # these parameters are not from the original CIS but are for test purposes
        self.csd1 = 1.
        self.csd2 = 1.
        self.b = 1.
        self.dV = 0.
        self.dE = 0.
        self.dB1 = 1.
        self.dB2 = 1.
