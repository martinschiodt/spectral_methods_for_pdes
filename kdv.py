import numpy as np
from scipy.integrate import solve_ivp
from utils import FourierTransform, InverseFourierTransform, FourierFrequencies
from utils import DownsampleSpectralArray, UpsampleSpectralArray
import matplotlib.pyplot as plt

""" kdv equations """
def kdv_TravellingWaveSolution(x, t, c, x0):
    # Range of cosh is [1, \infty) thus following operation allowed
    # sech(x) = 1/cosh(x)
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*(x-c*t-x0))**(-2)
    return u

def kdv_SpectralTimeDerivative(t, uh, K, Nwave):
    """ Convolution sum """
    u = InverseFourierTransform(uh, (0,))
    ux = InverseFourierTransform(1j*K*uh, (0,))
    uux = u*ux
    uuxh = FourierTransform(uux, (0,))
    
    # Derivative
    uh_dot = (1j/Nwave**3) * K**3 * uh - (6/Nwave) * uuxh
    return uh_dot

""" -------------- START OF DRIVER --------------"""
""" Function parameters """
Nwave = 20
c = 0.25
x0 = 1.0

""" Spatial domain """
nx = 128+1
dims = (nx,)
Lx = 2*np.pi*Nwave
dx = Lx / (nx+1)
x = np.linspace(-Lx/2+dx,Lx/2-dx,nx)

""" Temporal domain """
nt = 2
t0 = 0.0
tf = 40.0
t_eval = np.linspace(t0,tf,nt+1)
dt = t_eval[1]-t_eval[0]

""" Spectral parameters """
nb = (nx,)
Kfull = FourierFrequencies(nx, dx, Lx)

""" Initial condition """
u0 = kdv_TravellingWaveSolution(x, t0, c, x0)
uh0_full = FourierTransform(u0, (0,))

""" Downsample to use only nb basis functions in each coordinate """
uh0 = DownsampleSpectralArray(uh0_full, nb, dims)
K = DownsampleSpectralArray(Kfull, nb, dims)


""" Solve IVP """
sol = solve_ivp(kdv_SpectralTimeDerivative, # function
                [t0, tf], # t_start and t_stop
                uh0, # initial conditions
                args=(K,Nwave), # passed to function
                method='RK45', # method
                t_eval=t_eval,
                dense_output=True)
Uh = sol.y  # Fourier coefficients 

""" Compute solution in physical domain """
G = np.zeros((nx,nt+1))
for i in range(0,nt+1):
    Uh_up = UpsampleSpectralArray(np.reshape(Uh[:,i], nb), nb, dims)
    G[:,i] = InverseFourierTransform(Uh_up, (0,))

""" -------------- END OF DRIVER --------------"""

""" -------------- START OF PLOT --------------"""
""" Plot true solution """
plt.figure()
plt.plot(x, kdv_TravellingWaveSolution(x, t0, c, x0), 'b',linewidth=0.5)
plt.plot(x, kdv_TravellingWaveSolution(x, tf/2, c, x0), 'g',linewidth=0.5)
plt.plot(x, kdv_TravellingWaveSolution(x, tf, c, x0), 'r',linewidth=0.5)

""" Plot approximation on true solution"""
plt.plot(x,G[:,0],markerfacecolor='None',
          markeredgecolor='b',
          marker='o',
          markersize=2,
          linestyle='none')
plt.plot(x,G[:,int(nt/2)],markerfacecolor='None',
          markeredgecolor='g',
          marker='o',
          markersize=2,
          linestyle='none')
plt.plot(x,G[:,nt],markerfacecolor='None',
          markeredgecolor='r',
          marker='o',
          markersize=2,
          linestyle='none')


""" Plot error as function of time """
norm_error = np.zeros((nt+1,))
for j in range(0,nt+1):
    norm_error[j] = np.linalg.norm(G[:,j] - kdv_TravellingWaveSolution(x, t_eval[j], c, x0))

plt.figure()
plt.plot(t_eval,norm_error)
plt.xlabel(r'$t$')
plt.ylabel(r'$\ell _2$-error')

""" -------------- END OF PLOT --------------"""