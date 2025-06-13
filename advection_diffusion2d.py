from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from utils import FourierTransform, InverseFourierTransform, FourierFrequencies
from utils import DownsampleSpectralArray, UpsampleSpectralArray

def AdvectionDiffusionTrueSol(t, x, y, nu, u, v, x0, y0):
    f = (1/(4*t+1)) * np.exp(-( (x-u*t-x0)**2 + (y-v*t-y0)**2 ) / (nu*(4*t+1)))
    return f

def AdvectionDiffusionSpectralTimeDerivative(t, fh, nu, u, v, K, L):
    """ 
    Kopriva Algorithm 44 - Computes time derivative of the Fourier coefficients 
    for the Advection Diffusion Equation (4.13)
    
    ------ Input -------
    t: timestep
    fh: Fourier coefficients
    nu>0: diffusion coefficient
    u: flow velocity in x coordinate
    v: flow velocity in y coordinate
    K, L: Helper vectors containing angular frequency for each wavenumber combination
    
    Requires that M=N is an even number!
    
    ------ Output ------
    fh_dot: Temporal derivative of fh
    """
    
    # Derivative 
    fh_dot = -(1j*(u*K + v*L)*np.pi + nu*(K*K + L*L)*np.pi**2)*fh
    
    return fh_dot


""" -------------- START OF DRIVER --------------"""
""" Function parameters """
nu = 0.01
u = 0.8
v = u
x0 = -0.5
y0 = x0

""" Spatial domain """
nx = 129
ny = nx
dims = (nx,ny)
Lx = 2
Ly = 2
dx = Lx / (nx+1)
dy = Ly / (ny+1)
x = np.linspace(-Lx/2+dx,Lx/2-dx,nx)
y = np.linspace(-Ly/2+dy,Ly/2-dy,ny)
X, Y = np.meshgrid(x, y, indexing='ij')

"""" Temporal domain """
nt = 600
t0 = 0.0
tf = 1.5
t_eval = np.linspace(t0,tf,nt+1)
dt = t_eval[1]-t_eval[0]

""" Spectral parameters """
nb = (55,55)
trans_axes = (0,1)
Kfull = FourierFrequencies(nx, dx, Lx)
Lfull = FourierFrequencies(ny, dy, Ly)
K2d, L2d = np.meshgrid(Kfull,Lfull,indexing='ij')

""" Initial condition """
f0 = AdvectionDiffusionTrueSol(0.0, X, Y, nu, u, v, x0, y0)
fh0_full = FourierTransform(f0, trans_axes)

""" Downsample to use only nb basis functions in each coordinate """
fh0_d = DownsampleSpectralArray(fh0_full, nb, dims)
K2d_d = DownsampleSpectralArray(K2d, nb, dims)
L2d_d = DownsampleSpectralArray(L2d, nb, dims)

""" Reshape to vectors """
fh0 = np.reshape(fh0_d, (nb[0]*nb[1]))
K = np.reshape(K2d_d, (nb[0]*nb[1]))
L = np.reshape(L2d_d, (nb[0]*nb[1]))

""" Solve IVP """
sol = solve_ivp(AdvectionDiffusionSpectralTimeDerivative, # function
                [t0, tf], # t_start and t_stop
                fh0, # initial conditions
                args=(nu, u, v, K, L), # passed to function
                method='RK45', # method
                t_eval=t_eval,
                dense_output=True)

Fh = sol.y

""" Compute solution in real space """
G = np.zeros((nt+1,nx,ny))
for i in range(0,nt+1):
    Fh_up = UpsampleSpectralArray(np.reshape(Fh[:,i], nb), nb, dims)
    G[i] = InverseFourierTransform(Fh_up, trans_axes)

""" Set near-zero values to zero """
zerobnd = 1e-03
G[G < zerobnd] = 0

""" -------------- END OF DRIVER --------------"""

""" -------------- START OF PLOT --------------"""
""" Contourplot approximation """
plt.figure()
plt.contour(X,Y,G[0,:,:])
plt.contour(X,Y,G[200,:,:])
plt.contour(X,Y,G[400,:,:])
plt.colorbar()
plt.clim(1e-02,0.2)

""" Show mean profile, taking mean over y coordinate """
plt.figure()
# True solution
plt.plot(X[:,0], np.mean(AdvectionDiffusionTrueSol(0.0, X, Y, nu, u, v, x0, y0),axis=1), 'b',linewidth=0.5)
plt.plot(X[:,0], np.mean(AdvectionDiffusionTrueSol(0.5, X, Y, nu, u, v, x0, y0),axis=1), 'g',linewidth=0.5)
plt.plot(X[:,0], np.mean(AdvectionDiffusionTrueSol(1.0, X, Y, nu, u, v, x0, y0),axis=1), 'r',linewidth=0.5)

# Approximation
plt.plot(X[:,0], np.mean(G[0,:,:],axis=1),
         markerfacecolor='None',
         markeredgecolor='b',
         marker='o',
         markersize=2,
         linestyle='none')
plt.plot(X[:,0], np.mean(G[200,:,:],axis=1),
         markerfacecolor='None',
         markeredgecolor='g',
         marker='o',
         markersize=2,
         linestyle='none')
plt.plot(X[:,0], np.mean(G[400,:,:],axis=1),
         markerfacecolor='None',
         markeredgecolor='r',
         marker='o',
         markersize=2,
         linestyle='none')
""" -------------- END OF PLOT --------------"""

