# file: driven_transport.py

# This program, tries to calculate the dc current through
# a system in presence of periodic driven field.

# Refrences:
# (1) T. Kitagawa, et. al. "Transport properties of non-equilibrium systems
# under the application of light: Photo-induced quantum Hall insulators
# without Landau levels", Phys. Rev. B 84, 235108 (2011)
#
# (2) S. Kohler, et. al., "Driven quantum transport on the nanoscale",
# Physics Reports, 406(6), 379-443 (2005)

# author: Amin Ahmadi
# date: June 18, 2018
############################################################

# developing process:
# 1- constructing Fourier transform augmented Hamiltonian, finite (M) matrix
# 2- evaluate the coupling of the system to lead \Gamma(\omega)
# 3- calculate \mg(n,\omega), where \mg = \mathcal{G}
# 4- calculate the transmission probability T_{\alpha,\beta}(n,\omega)
# 5- sum over harmonics (n) and lead connection (beta) to find
#  pumping and reservoir current
# zero-temperature formulation:
# J_res = \delta\mu \sum_{n=-M}^M T_{LR}(n,\mu), with
# T_{LR}(n,\mu) = \gamma(\mu)\gamma(\mu+n\Omega)|\mg(n,\mu)|^2


import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL         # extra function in file Floquet_2D.py
import matplotlib.pyplot as pl

############################################################
##############         Functions        ####################
############################################################
def H_aug_generate(N_harmonic, mlat, J):
    """ construct the augmented field periodic Hamiltonian.
    input:
    ------
    N_harmonic: int, number of harmonic that H(t) will be expanded to
    mlat: int, width of armchair graphene
    J: float, hopping amplitude

    return:
    -------
    H_aug: (aug_dim,aug_dim) complex, augmented Hamiltonian
    tau_aug: (aug_dim,aug_dim) complex, augmented Hopping matrix
    taudg_aug: (aug_dim,aug_dim) complex, augmented Hopping matrix (H.C.) 
    """
    T = 1
    N_t = 3
    Omega = 2*np.pi/T                    # Frequency of the driven field
    NN = 2*mlat                          # number of sites in one translational unitcell
    aug_dim = NN*(2*N_harmonic+1)        # dimension of augmented Hamiltonian
    H_aug = np.zeros((aug_dim, aug_dim), dtype=complex)
    tau_aug = np.zeros((aug_dim, aug_dim), dtype=complex)
    taudg_aug = np.zeros((aug_dim, aug_dim), dtype=complex)   

    for i, im in enumerate(range(-N_harmonic,N_harmonic+1)):
        H_aug[i*NN:(i+1)*NN,i*NN:(i+1)*NN] += -im*Omega*np.eye(NN) 
        for j, il in enumerate(range(-N_harmonic,N_harmonic+1)):
            if (il == im):
                aux = 1./N_t 
                aux2 = 0
            else:
                aux = (2.j*np.sin( (il-im)*np.pi/N_t ) ) / ((il-im)*Omega)
                aux2 = 2.j*np.pi*(il-im)/N_t
                
            haux = np.zeros((NN,NN), dtype=complex)
            taux = np.zeros((NN,NN), dtype=complex)
            tdgaux = np.zeros((NN,NN), dtype=complex)
                
            for it in range(N_t):
                if   (it==0 ):
                    Jtup = (delta*J,J,J)
                elif (it==1):
                    Jtup = (J,delta*J,J)
                elif (it==2):
                    Jtup = (J,J,delta*J)

                # construct the Hamiltonian and hopping matrices
                h, tau = FL.make_Gr(mlat, *Jtup)
                tau_dg = tau.conj().T

                haux += h*np.exp(aux2*(it+1))
                taux += tau*np.exp(aux2*(it+1))
                tdgaux += tau_dg*np.exp(aux2*(it+1))
            # End of it-loop
            # print('i,j are: ', i, j, im, il)
            # print('M dim: ', i*NN, (i+1)*NN,j*NN, (j+1)*NN)
            H_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] += aux*haux
            tau_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] = aux*taux
            taudg_aug[i*NN:(i+1)*NN,j*NN:(j+1)*NN] = aux*tdgaux
        # endof il-loop
    #endof im-loop

    return H_aug, tau_aug, taudg_aug
########################################
def g_lead_dec(Nd,E, tau, h):
    """ Compute the lead's Green's function using decimation
    method. 
    
    input:
    ------
    Nd: integer, dimension of the Hamiltonian
    E: float, energy
    h: (Nd,Nd) complex matrix, hamiltonian
    tau: (Nd,Nd) complex matrix, hopping matrix between 
    superlattice

    return:
    -------
    gl: (Nd,Nd) complex matrix, lead's Green's function
    """
    
    eta = 1.e-7j               # infinitesimal imaginary for retarded G.Fs.
    I = np.eye(Nd,dtype=complex)
    
    ee = E + eta
    # initialize alpha, beta, eps, eps_s
    alpha = tau
    beta = tau.conj().T
    eps = h
    eps_s = h

          
    for i_dec in range(40):
        aux = lg.inv(ee*I - eps)
        aux1 = np.dot(alpha,np.dot(aux,beta))
        eps_s = eps_s + aux1
        eps = eps + aux1
        aux1 = np.dot(beta, np.dot(aux, alpha))
        eps = eps + aux1
        alpha = np.dot(alpha,np.dot(aux, alpha))
        beta = np.dot(beta, np.dot(aux, beta))


    gl = lg.inv(ee*I - eps_s)
    return gl
########################################
def self_aug(Nd, N_harmonic, Omega, E, tau, h):
    """ calculate the matrix of augmented self energy for left
    and right lead.
    input:
    ------
    Nd: integer, dimension of static Hamiltonian
    N_harmonic: integer, num of harmonics
    E: float, energy

    return:
    -------
    slf_aug: array (Nd*Nh,Nd*Nh)
    """
    Nh = 2*N_harmonic+1
    I = np.eye((Nd), dtype=float)
    tau_dg = tau.conj().T
    slf_augL = np.zeros((Nd*Nh,Nd*Nh), dtype=complex)
    slf_augR = np.zeros((Nd*Nh,Nd*Nh), dtype=complex)
    nOmeg = np.zeros((Nd*Nh,Nd*Nh), dtype=float)

    nrange = range(-N_harmonic,N_harmonic+1)
    
    for i, nharm in enumerate(nrange):
        # The lead's Green's function
        gl = g_lead_dec(Nd, ee+nharm*Omega, tau, h)
        gr = g_lead_dec(Nd, ee+nharm*Omega, tau_dg, h)

        # The self-energy due to the Left and right reservoirs
        sigma_l = np.dot(tau, np.dot(gl,tau_dg))
        sigma_l_dg = sigma_l.conj().T
        
        sigma_r = np.dot(tau_dg, np.dot(gr,tau))
        sigma_r_dg = sigma_r.conj().T
        
        slf_augL[i*Nd:(i+1)*Nd,i*Nd:(i+1)*Nd] = sigma_l
        slf_augR[i*Nd:(i+1)*Nd,i*Nd:(i+1)*Nd] = sigma_r

        nOmeg[i*Nd:(i+1)*Nd,i*Nd:(i+1)*Nd] = nharm*Omega*I
    # return diagonal matrix
    return slf_augL, slf_augR, nOmeg
########################################
###          Floquet Transport       ###
########################################
NE = 300                        # energy resolution
eta = 1.e-7j                   
T = 1                           # time-period
Omega = 2*np.pi/T
N_harmonic = 1                  # number of harmonic to construct augmented H

# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = int(input("Graphene strip width: "))   # width of strip
Nd = 2*mlat                                   # number of sites in one T-unit cell
Nd_aug = Nd*(2*N_harmonic+1)                  # dim of augmented H and G

# different hopping amplitude
delta = float(input("Enter the hopping difference coefficient: ")) 
J = np.pi/16.                             # hopping amplitude 
I = np.eye(Nd_aug,dtype=complex)

###########################
# calculate the current
J_arr = np.zeros((NE), float)   # to save current
Ei = -3.5
Ef = 3.5
DE = (Ef-Ei)/NE

# construct augmented Hamiltonian
H_aug, tau_aug, tau_augdg  = H_aug_generate(N_harmonic, mlat, J)

Jtup = (J,J,J)
# construct the Hamiltonian and hopping matrices
h, tau = FL.make_Gr(mlat, *Jtup)

# loop over energy 
for ie in range(NE):

    ee = Ei +  ie*DE

    
    # augmented self-energy
    sL, sR, nOmg = self_aug(Nd, N_harmonic, Omega, ee, tau, h)


    # Full Green's function
    Gd = lg.inv(ee*I + nOmg - H_aug - sL - sR)
    Gd_dg = Gd.conj().T

    gamma_r = -1j * ( sR - sR.conj().T )
    gamma_l = -1j * ( sL - sL.conj().T )

    # This procedure shoud work for zero harmonic problem
    # G * gamma_l * G_dg * gamma_r
    auxg = np.dot(Gd,np.dot(gamma_l,np.dot(Gd_dg,gamma_r)))

    gg = np.trace(auxg)
    
    J_arr[ie] = gg.real
# Endof energy-loop

# conductance
E = np.linspace(Ei,Ef,NE)


###################
# bands
pl.figure()
pl.plot(E, J_arr)

# k = np.linspace(ki,kf,Nk)
# for i in range(Nd):
#     pl.plot(k,E_arr[:,i])

######################################################
######          Floquet Dispersion          ##########
######################################################
# N_k = 300                                # Num of k-points, 
# data_plot = np.zeros((N_k, Nd), dtype=float)

# k_range = np.linspace(-np.pi, np.pi, N_k)
# # loop over k, first BZ
# ik = 0
# for ka in k_range:
    
#     data_plot[ik,:] = HF_calculator(ka,J,delta,1)[0]
#     ik+=1
######################################################
##########       BandStructure  Plot   ###############
######################################################
# Use plot_FL.py file for plotting

# fig, ax = pl.subplots(1)
# mm = ['-r', '-k', '-c', '-b', '-y', '-g']
# for i in range(NN):
#     pl.plot(k_range, data_plot[:,i], '-k', markersize=1)
    


pl.show()
