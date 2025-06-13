import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def AdvectionDiffusionTrueSol(t, x, nu):
    N_true = 100
    K = np.linspace(start=-N_true/2,stop=N_true/2,num=N_true+1)
    fh0_true = AdvectionDiffusionIntialSpectralCoefficients(N_true)
    f = np.zeros(x.shape)
    for k in range(0,N_true+1):
        f = f + fh0_true[k] * np.exp(1j*K[k]*(x-t)-nu*K[k]*K[k]*t)
    return f.real

def AdvectionDiffusionSpectralTimeDerivative(t, fh, nu):
    """ 
    Kopriva Algorithm 44 - Computes time derivative of the Fourier coefficients 
    for the Advection Diffusion Equation (4.13)
    
    ------ Input -------
    t: timestep
    fh: Fourier coefficients
    nu>0: diffusion coefficient
    
    Requires that len(fh)-1 is an even number!
    
    ------ Output ------
    fh_dot: Temporal derivative of fh
    """
    
    # Helpers
    N = int(len(fh)-1)
    K = np.linspace(start=-N/2,stop=N/2,num=N+1)
    
    # Derivative 
    fh_dot = -(1j*K + nu*K*K)*fh
    
    return fh_dot

def AdvectionDiffusionIntialSpectralCoefficients(N):
    """ Using equations (4.26) and (4.27) to produce initial Fourier Coefficients
    Use initial condition 
    f_0(x) = 3 / (5-4cos(x)) 
    which has exact Fourier coefficients
    fh_{0,k} = 2^(-|k|)
    """
    K = np.linspace(start=-N/2,stop=N/2,num=N+1)
    fh0 = 2**(-abs(K)) + 0j # Has to be complex for solve_ivp!!
    return fh0

def EvaluateFourierGalerkinSolution(x, fh):
    """ 
    Computes Fourier polynomial, g, using coefficients. 
    g approximates original function f 
    """
    N = int(len(fh)-1)
    K = np.linspace(start=-N/2,stop=N/2,num=N+1)
    g = np.zeros(x.shape)
    for k in range(0,N+1):
        g = g + fh[k] * np.exp(1j*K[k]*x)
    return g.real

""" -------------- START OF DRIVER --------------"""
""" Function parameters """
iv = lambda x: 3 / (5-4*np.cos(x))
nu = 0.2

""" Spatial domain """
nx = 50
Lx = 2*np.pi
dx = Lx / (nx+1)
x = np.linspace(dx,Lx-dx,nx)

"""" Temporal domain """
nt = 800
t0 = 0.0
tf = 2.0
t_eval = np.linspace(t0,tf,nt+1)
dt = t_eval[1]-t_eval[0]

""" Spectral parameters and initial condition """
N = 16
fh0 = AdvectionDiffusionIntialSpectralCoefficients(N)

""" Solve IVP """
sol = solve_ivp(AdvectionDiffusionSpectralTimeDerivative, # function
                [t0, tf], # t_start and t_stop
                fh0, # initial conditions
                args=(nu,), # passed to function
                method='RK45', # method
                t_eval=t_eval,
                dense_output=True)
Fh = sol.y # Fourier coefficients 

""" Compute solution in real space myself """
G = np.zeros((nx,nt+1))
for i in range(0,nt+1):
    G[:,i] = EvaluateFourierGalerkinSolution(x, Fh[:,i])
    
""" -------------- END OF DRIVER --------------"""

""" -------------- START OF PLOT --------------"""
""" Plot true solution """
plt.figure()
plt.plot(x, AdvectionDiffusionTrueSol(0.0, x, nu), 'b',linewidth=0.5)
plt.plot(x, AdvectionDiffusionTrueSol(1.0, x, nu), 'g',linewidth=0.5)
plt.plot(x, AdvectionDiffusionTrueSol(2.0, x, nu), 'r',linewidth=0.5)

""" Plot approximation on true solution"""
plt.plot(x,G[:,0],markerfacecolor='None',
         markeredgecolor='b',
         marker='o',
         markersize=2,
         linestyle='none')
plt.plot(x,G[:,400],markerfacecolor='None',
         markeredgecolor='g',
         marker='o',
         markersize=2,
         linestyle='none')
plt.plot(x,G[:,800],markerfacecolor='None',
         markeredgecolor='r',
         marker='o',
         markersize=2,
         linestyle='none')


""" Plot spectral coefficients at t=0.0, t=1.0, t=2.0 """
K = np.linspace(start=-N/2,stop=N/2,num=N+1)
plt.figure()
plt.subplot(2,1,1)
plt.plot(K,Fh[:,0].real,
         markerfacecolor='None',
         markeredgecolor='b',
         marker='o',
         markersize=2,
         linestyle='-',
         color='b')
plt.plot(K,Fh[:,400].real,
         markerfacecolor='None',
         markeredgecolor='g',
         marker='o',
         markersize=2,
         linestyle='-',
         color='g')
plt.plot(K,Fh[:,800].real,
         markerfacecolor='None',
         markeredgecolor='r',
         marker='o',
         markersize=2,
         linestyle='-',
         color='r')
plt.title('Real component of coefficients')

plt.subplot(2,1,2)
plt.plot(K,Fh[:,0].imag,
         markerfacecolor='None',
         markeredgecolor='b',
         marker='o',
         markersize=2,
         linestyle='-',
         color='b')
plt.plot(K,Fh[:,400].imag,
         markerfacecolor='None',
         markeredgecolor='g',
         marker='o',
         markersize=2,
         linestyle='-',
         color='g')
plt.plot(K,Fh[:,800].imag,
         markerfacecolor='None',
         markeredgecolor='r',
         marker='o',
         markersize=2,
         linestyle='-',
         color='r')
plt.title('Imaginary component of coefficients')
""" -------------- END OF PLOT --------------"""


print("Error:",np.linalg.norm(G[:,800]-AdvectionDiffusionTrueSol(2.0, x, nu)))