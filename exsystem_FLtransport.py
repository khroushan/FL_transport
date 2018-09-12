# file: exsystem_FLtransport.py

# This program regenerates the results that are mention is paper:
# "Calculating electron transport in a tight binding model of a
# field -driven molecular wire: Floquet theory approach"
# A. Tikhonov, et. al J. Chem. Phys. 116 10909 (2002)

# The paper provides a technique to calculate transport in a
# field driven system.

# author: Amin Ahmadi
# date: April 12, 2018
############################################################
import numpy as np
import scipy.special as scp
import numpy.linalg as lg

##################################################
# Functions

def lead_den(E):
    """ To calculate the density of state of a 1D system using
    relation given explicitly.
    input:
    ------
    E: float, energy.

    return:
    -------
    den: float, density of state
    """
    # constants 
    gamma = 0.5
    v2_gma = 0.12

    if np.abs(E) < 2*gamma :
        den = v2_gma * np.sqrt( 1-(E/(2*gamma))**2 )
    else:
        den = 0
    return den
########################################

def lead_g(E):
    """ To calculate the Green's function of 1D lead.
    input:
    ------
    E: float, energy.

    return:
    -------
    gg: complex, Green's function of 1D lead
    """
    # there is a discrepancy between DOS and GF definition
    
    gamma = 0.5                      # inter-chain hopping amplitude
        
    if np.abs(E) <= 2*gamma :
        gg = E/(2*gamma**2) - (1.j/gamma) * np.sqrt( 1. - (E/(2*gamma))**2 )
    else:
        # There is no energy state beyond this energy
        gg = 0. # E/(2*gamma**2) - (1./gamma) * np.sqrt((E/(2*gamma))**2 -1. )

    return gg
########################################

def H_eff():
    """ To construct the effective Hamiltonian of 3-state 
    bridge system between two leads.
    input:
    ------
    EB: float, on-site energy.
    E: float, energy of the system

    return:
    -------
    H_eff: np.array(3,3), complex, effective Hamiltonian
    """

    # constants
    EB = 0.8
    aB = 0.2
    VB = 0.1                    # intra-bridge coupling
    h12 = VB * scp.jv(0,aB)
    H_eff = np.array([[EB, h12,0],
                      [h12, EB, h12],
                      [0, h12, EB]], dtype=float)

    return H_eff
########################################

def g_eff(E, omega):
    """ Effective Green's function of chain in presence of leads.
    """
    gamma = 0.5                     # hopping amplitude
    v2_gma = 0.12

    aL = 2.                        # hopping amplitude between lead and chain
    ieps = 1.e-6j
    
    # effective self-enery
    sig_eff = 0.
    for i in range(-5,6): #{-3,-3,...,2,3}
        sigma =  lead_g(E-i*omega)    # static self-energy
        sig_eff += sigma * scp.jv(i,aL)**2  

    sig_eff *= v2_gma  * gamma 
    sig_mat = np.array([[sig_eff, 0., 0.],
                        [0., 0., 0.     ],
                        [0., 0., sig_eff]], dtype=complex)

    aux = ( H_eff() - E - sig_mat  - ieps)

    return lg.inv(aux)
########################################

##################################################
###               Main Program                 ###
##################################################
import matplotlib.pyplot as pl
from matplotlib import rc
rc('font', **{'family':'sans-serif',
              'sans-serif':['Helvetica']})
rc('text', usetex=True)

aL = 2.
Ei0 = 0.                         # left lead energy
Np = 0
Vio = 1.

Nres = 200
dRsum = np.zeros((Nres), float)
omRange = np.linspace(0,1,Nres)
colors = ['b-', 'c-', 'k-', 'g-', 'y-']

for Np in range(-2,3):
    print("Np: ", Np)
    i=0
    dataR = np.zeros((Nres), float)
    for omega in omRange:
        sumG = 0

        for im in range(-9,10):
            sumG += scp.jv(im,aL) * g_eff(Ei0-im*omega, omega)[0,2] \
                    * scp.jv(Np-im,aL)
        
        Rio = 2 * Vio**2 * lead_den(Ei0 - Np*omega) * (np.abs(sumG)**2)
        dataR[i] = Rio
        i+=1
    dRsum += dataR
    pl.plot(omRange, dataR, colors[Np%5], label=str(Np))
    


pl.xlim(0,1)
pl.ylim(0,0.5)
pl.plot(omRange, dRsum, 'k--', linewidth=3)
pl.xlabel(r'$\omega$', fontsize=16)
pl.ylabel(r'$\Delta\times R_{io}$', fontsize=16)
pl.legend()
pl.show()
