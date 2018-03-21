# file: gr_transport.py

# to study transport in a graphene lattice. First I start
# with a dc tight-binding Hamiltonian and then I'll try to
# develop a code to study transport using Floquet
# Hamiltonian. Right now I don't have a clear idea if the
# trasport calculated using the Floquet Hamiltonian is
# related to any real physical interpretation.


# author: Amin Ahmdi
# date: March 15, 2018
############################################################
import numpy as np
import numpy.linalg as lg

########################################
###           Functions              ###
########################################
def make_Gr(mlat, J1=1, J2=1, J3=1):
    """ Constructs the Hamiltonian and the connection 
    matrix of an armchair graphene strip
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    |  |  |  |
    o  0--o  0
    |  |  |  |
    0--o  0--o
    
    returns: unitcell hamiltonian h
             hopping matrix       tau
    """
    NN = 2*mlat                 # # of sites in one super unitcell
    tau = -np.zeros((NN, NN),dtype=complex)
    h = np.zeros((NN,NN), dtype=complex)

    # translational cell's Hamiltonian
    for i in range(mlat-1):
        if (i%2==0):
            h[i,i+1] = J1
            h[mlat+i,mlat+i+1] = J2
            h[i,mlat+i] = J3    # horizoltal connection
        elif (i%2==1):
            h[i,i+1] = J2
            h[mlat+i,mlat+i+1] = J1
            
            
    h = h + h.conj().T          # make it hermitian
    # Hopping matrix
    for i in range(1,mlat,2):
        tau[i+mlat,i] = J3

    return h, tau
##################################################

# def make_Gr(mlat):
#     """ Constructs the Hamiltonian and the connection 
#     matrix of an armchair graphene strip. Lattice structure:

#     0--o  0--o
#     |  |  |  |
#     o  0--o  0
#     |  |  |  |
#     0--o  0--o
#     |  |  |  |
#     o  0--o  0
#     |  |  |  |
#     0--o  0--o
    
#     input:
#     ------
#     mlat: integer, number of sites in width
    
#     return:
#     --------
#     h: (Nd,Nd) complex matrix, hamiltonian
#     tau: (Nd,Nd) complex matrix, hopping matrix between 
#     superlattice
#     """
    
#     Nd = 2*mlat                 # # of sites in one super unitcell
#     tau = np.zeros((Nd, Nd),dtype=complex)
#     h = np.zeros((Nd,Nd), dtype=complex)
#     t = -1.                      # hopping amplitude

#     # translational cell's Hamiltonian
#     for i in range(mlat-1):
#         h[i,i+1] = t
#         h[mlat+i,mlat+i+1] = t
#         if (i%2==0):
#             h[i,mlat+i] = t    # horizoltal connection
        
            
            
#     h = h + h.conj().T          # make it hermitian
#     # Hopping matrix
#     for i in range(1,mlat,2):
#         tau[i+mlat,i] = t

#     return h, tau
########################################
def make_sq(mlat):
    """ Constructs the Hamiltonian and the connection 
    matrix of a square lattice. Lattice structure:

    0--0--0--0
    |  |  |  |
    0--0--0--0
    |  |  |  |
    0--0--0--0
    |  |  |  |
    0--0--0--0
    |  |  |  |
    0--0--0--0
    
    input:
    ------
    mlat: integer, number of sites in width
    
    return:
    --------
    h: (Nd,Nd) complex matrix, hamiltonian
    tau: (Nd,Nd) complex matrix, hopping matrix between 
    superlattice
    """
    
    Nd = mlat                 # # of sites in one super unitcell
    tau = np.zeros((Nd, Nd),dtype=complex)
    h = np.zeros((Nd,Nd), dtype=complex)
    t = -1.                      # hopping amplitude

    # translational cell's Hamiltonian
    for i in range(mlat-1):
        h[i,i+1] = t
                    
    h = h + h.conj().T          # make it hermitian
    # Hopping matrix
    for i in range(0,mlat):
        tau[i,i] = t

    return h, tau

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
#######################################

########################################
###           Main Program           ###
########################################
NE = 200
eta = 1.e-7j
mlat = 9                       # two rows to mimic spin stat
Nd = 2*mlat                     # number of sites in a SL

I = np.eye(Nd,dtype=complex)

h, tau = make_Gr(mlat)
tau_dg = tau.conj().T

gl = np.zeros((Nd,Nd), dtype=complex)

###########################
# calculate the conductance
con_arr = np.zeros((NE), float)
Ei = -1.5
Ef = 1.5
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
    Gd = lg.inv(ee*I - h - sigma_r - sigma_l -eta)
    Gd_dg = Gd.conj().T

    gamma_r = -1j * ( sigma_r - sigma_r_dg)
    gamma_l = -1j * ( sigma_l - sigma_l_dg)

    # G * gamma_l * G_dg * gamma_r
    auxg = np.dot(Gd,np.dot(gamma_l,np.dot(Gd_dg,gamma_r)))

    gg = np.trace(auxg)
    
    con_arr[ie] = gg.real
# Endof energy-loop

##########################
# # calculate the band structure 
ki = -np.pi/2
kf = np.pi/2
Nk = 300
Dk = (kf-ki)/Nk
E_arr = np.zeros((Nk, Nd), float)

h, tau = make_Gr(mlat)
tau_dg = tau.conj().T
# loop over k
for ik in range(Nk):
    ka = ki + ik*Dk

    Hk = h + np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg

    E_k = lg.eigvals(Hk)

    E_arr[ik] = np.sort(E_k.real)
# Endof k-loop

########################################
###            Plotting              ###
########################################
import matplotlib.pyplot as pl
# conductance
E = np.linspace(Ei,Ef,NE)

pl.plot(E, con_arr)
###################
# bands
pl.figure()
k = np.linspace(ki,kf,Nk)
for i in range(Nd):
    pl.plot(k,E_arr[:,i])

pl.show()

