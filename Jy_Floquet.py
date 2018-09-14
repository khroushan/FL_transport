# Program to calculate the transverse current j^y in 
# in a square lattice with stroboscopic driven field
# 
# 
# Author: Amin Ahmdi
# Date: Sep 7, 2018

# ################################################################
import numpy as np
import numpy.linalg as lg
import Floquet_2D as FL
########################################
###           Functions              ###
########################################
def j_y(mlat, i):
    """ transverse current j_y at point (i,j) in the lattice
    input:
    ------
    mlat: int, width of lattice
    i,j: int, lattice index

    return:
    -------
    j_y: (NxN) complex array, j_y current operator
    
    """
    N = 2*mlat                  # # of points in the unit lattice
    j_y = np.zeros((N,N), dtype=complex) # current operator
    
    # check if index is in lattice size range
    if ( i>= mlat-1 or i<1):
        print('index is not valid')
    else:
        j_y[i,i-1] = 1.j
        j_y[i-1,i] = -1.j
    

    return j_y
    
############################################################
##############         Main Program     ####################
############################################################
N_k = 100                                # Num of k-points, 
                                         # odd number to exclude zero
N_t = 5                                  # Num of time intervals
T = 1.                                   # One period of driven field
dAB = 0                                  # on_site energy difference

# During each iterval T/N_t, the Hamiltonian is time-independent
mlat = int(input("Graphene strip width: "))   # width of strip
NN = 2*mlat
H_k = np.zeros((NN,NN), dtype=complex)   # k-representation H
E_k = np.zeros((NN), dtype=complex)      # eigenenergies
E_real = np.zeros((NN), dtype=float)     # eigenenergies
psi_k = np.zeros((NN,NN), dtype=complex) # matrix of eigenvectors
# UF_F = np.zeros((mlat,mlat), dtype=complex) # U up to Fermi level
W_loop = np.eye((mlat), dtype=complex)# matrix to to Wilson loop calculation

# different hopping amplitude
delta = float(input("Enter the hopping difference coefficient: ")) 
J = np.pi/16.                             # hopping amplitude 
data_plot = np.zeros((N_k, NN+1), dtype=float)

################################
###      QuasiEnergies      ####
################################

Jtup = J*np.ones((4), float)
# loop over k, first BZ 
for ik in range(N_k):

    ka = -np.pi + ik*(2.*np.pi)/(N_k)
    # ka = ik*(np.pi/N_k)
    M_eff = np.eye((NN), dtype=complex)   # aux matrix
    for it in range(N_t):
        if   (it==4 ):
            Jtup = J*np.zeros((4), float)
        else :
            Jtup[it] = delta*J


        # construct the Hamiltonian and hopping matrices
        h, tau = FL.make_sq(mlat, dAB, Jtup)
        tau_dg = tau.conj().T 

        # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
        # and diagonalization
        H_k = h + np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg

        # return eigenenergies and vectors
        E_k, U = lg.eig(H_k)    

        # U^-1 * exp(H_d) U
        U_inv = lg.inv(U)

        # construct a digonal matrix out of a vector
        #H_M= np.diag(np.exp((-1j/3.)*E_k*T))
        M1 = (np.exp((-1.j)*E_k*T) * U_inv.T).T
        #MM = np.dot(U_inv,np.dot(H_M, U))
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)


    E_Fl, UF = lg.eig(M_eff)
        
    E_real = np.log(E_Fl).imag
    indx = np.argsort(E_real)
    E_sort = E_real[indx]
    UF_F = UF[indx[:int(NN/2)]]
    UF_inv = UF_F.T.conj()
    # print("UF shape is ", UF_F.shape)
    # print("UF_inv shape is ", UF_inv.shape)
    # print("W_loop shape is ", W_loop.shape)
    # Wilson loop winding number calculation
    if (ik%2 == 0):
        W_loop = np.dot(W_loop, UF_F)
    elif (ik%2==1):
        W_loop = np.dot(W_loop, UF_inv)
    
    data_plot[ik,0] = ka
    data_plot[ik,1:] = E_sort/(T)

wind_num = (np.log(lg.det(W_loop)).imag)/(2.*np.pi)
# wind_num = lg.det(W_loop)
print(wind_num)
# save the data
# np.savetxt("./Result/FL_disert.dat", data_plot, fmt="%.2e")

################################
###  Transverse Current     ####
################################

T = 1
NT = 100                 # time-resolution of each interval
N_t = 5                  # # of intervals in one period
t_intval = T/N_t         # one time interval

tRange = np.linspace(0,t_intval, NT)
# we compute the current for a single k-point
ka = np.pi/4

Jtup = J*np.ones((4), float)        # hoping amplitudes tuple
for it in range(N_t):
    if   (it==4 ):
        Jtup = J*np.zeros((4), float)
    else :
        Jtup[it] = delta*J

    # construct the Hamiltonian and hopping matrices
    # needed for each interval
    h, tau = FL.make_sq(mlat, dAB, *Jtup)
    tau_dg = tau.conj().T 
    
    
    # Construct matrix: [h + tau*exp(ika) + tau^+ * exp(-ika)]
    H_k = h + np.exp(1.j*ka)*tau + np.exp(-1.j*ka)*tau_dg
    
    # one diagonalization for each interval
    E_k, U = lg.eig(H_k)    

    # U^-1 * exp(H_d) U
    U_inv = lg.inv(U)

    for t in tRange:
        M_eff = np.eye((NN), dtype=complex)   # aux matrix

        # construct a digonal matrix out of a vector
        M1 = (np.exp(-1.j)*E_k*t) * U_inv.T).T
        MM = np.dot(U,M1)
        M_eff = np.dot(M_eff,MM)



################################
###            Plot         ####
################################
# Use plot_FL.py file for plotting

import matplotlib.pyplot as pl
fig, ax = pl.subplots(1)
mm = ['-r', '-k', '-c', '-b', '-y', '-g']
for i in range(1,NN+1):
    pl.plot(data_plot[:,0], data_plot[:,i], '-k', markersize=1)
    
pl.show()
        
