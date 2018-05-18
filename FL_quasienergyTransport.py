# file: FL_quasienergyTransport.py

# This program, calculate the Floquet quasi energy and
# effective Floquet Hamiltonian of a periodic filed-driven
# system. Then the transport is calculated using H_F such
# that the system is static.

# author: Amin Ahmadi
# date: May 18, 2018
############################################################

import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL         # extra function in file Floquet_2D.py
import matplotlib.pyplot as pl

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
def HF_calculator(ka, J, delta, koff):
    """ Compute the Floquet Hamiltonian,
    input:
    ------
    ka: float, ka.
    J: float, hopping amplitude
    delta: float, time-dependent hopping amplitude
    koff: float, koff=0 ==> no k-Fourier
    
    return:
    -------
    E_sort: vec(N) real, quasienergies at ka
    HF: mat(N,N) complex, Floquet Hamiltonian
    tauFL: mat(N,N) complex, transformed Hamiltonian
    """

    N_t = 3
    M_eff = np.eye((NN), dtype=complex)      # aux matrix
    H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
    E_k = np.zeros((NN), dtype=complex)      # eigenenergies
    E_real = np.zeros((NN), dtype=float)     # eigenenergies

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

        # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
        # and diagonalization
        H_k = h + koff * (np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg)

        # return eigenenergies and vectors
        E_k, U = lg.eig(H_k)

        # U^-1 * exp(H_d) U
        U_inv = lg.inv(U)

        # construct a digonal matrix out of a vector
        #H_M= np.diag(np.exp((-1j/3.)*E_k*T))
        M1 = (np.exp((-1.j)*E_k*T/3) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)


    E_Fl, UF = lg.eig(M_eff)
    UF_inv = lg.inv(UF)
    
    E_real = np.log(E_Fl).imag

    # constructing Floquet Hamiltonian
    Mf1 = (-E_real * UF_inv.T).T
    # #MM = np.dot(U_inv,np.dot(H_M, U))
    HF = np.dot(UF,Mf1)

    # construct the Floquet hoping matrix tau
    tauFL = lg.inv(M_eff).dot(tau.dot(M_eff))
    indx = np.argsort(E_real)
    E_sort = E_real[indx]
    
    return E_sort, HF, tauFL



########################################
###          Floquet Transport       ###
########################################
NE = 300
eta = 1.e-7j
T = 1

# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = int(input("Graphene strip width: "))   # width of strip
NN = 2*mlat
Nd = 2*mlat                     # number of sites in a SL
# different hopping amplitude
delta = float(input("Enter the hopping difference coefficient: ")) 
J = np.pi/16.                             # hopping amplitude 
I = np.eye(Nd,dtype=complex)

h, tau = HF_calculator(0,J,delta,0)[1:]
tau_dg = tau.conj().T

gl = np.zeros((Nd,Nd), dtype=complex)

###########################
# calculate the conductance
con_arr = np.zeros((NE), float)
Ei = -3.5
Ef = 3.5
DE = (Ef-Ei)/NE
# loop over energy 
for ie in range(NE):

    ee = Ei +  ie*DE
    

    # The lead's Green's function
    gl = g_lead_dec(Nd,ee, tau, h)
    gr = g_lead_dec(Nd,ee, tau_dg, h)

    # The self-energy due to the Left and right reservoirs
    sigma_l = np.dot(tau, np.dot(gl,tau_dg))
    sigma_l_dg = sigma_l.conj().T

    sigma_r = np.dot(tau_dg, np.dot(gr,tau))
    sigma_r_dg = sigma_r.conj().T
    

    # Full Green's function 
    Gd = lg.inv(ee*I - h - sigma_r - sigma_l )
    Gd_dg = Gd.conj().T

    gamma_r = -1j * ( sigma_r - sigma_r_dg)
    gamma_l = -1j * ( sigma_l - sigma_l_dg)

    # G * gamma_l * G_dg * gamma_r
    auxg = np.dot(Gd,np.dot(gamma_l,np.dot(Gd_dg,gamma_r)))

    gg = np.trace(auxg)
    
    con_arr[ie] = gg.real
# Endof energy-loop

# conductance
E = np.linspace(Ei,Ef,NE)


###################
# bands
pl.figure()
pl.plot(E, con_arr)

# k = np.linspace(ki,kf,Nk)
# for i in range(Nd):
#     pl.plot(k,E_arr[:,i])

######################################################
######          Floquet Dispersion          ##########
######################################################
N_k = 300                                # Num of k-points, 
data_plot = np.zeros((N_k, NN), dtype=float)

k_range = np.linspace(-np.pi, np.pi, N_k)
# loop over k, first BZ
ik = 0
for ka in k_range:
    
    data_plot[ik,:] = HF_calculator(ka,J,delta,1)[0]
    ik+=1
######################################################
##########       BandStructure  Plot   ###############
######################################################
# Use plot_FL.py file for plotting

fig, ax = pl.subplots(1)
mm = ['-r', '-k', '-c', '-b', '-y', '-g']
for i in range(NN):
    pl.plot(k_range, data_plot[:,i], '-k', markersize=1)
    


pl.show()
