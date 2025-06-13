import numpy as np
from utils import FourierTransform, InverseFourierTransform, FourierFrequencies
from utils import DownsampleSpectralArray, UpsampleSpectralArray
import matplotlib.pyplot as plt

""" Test of utils """
def testfun(t, x, y):
    f = (1/(4*t+1)) * np.exp(-( (x-t)**2 + (y-t)**2 ) / (4*t+1))
    return f

""" Spatial domain """
nx = 65
ny = 33
dims = (nx,ny)
Lx = 2
Ly = 2
dx = Lx / (nx+1)
dy = Ly / (ny+1)
x = np.linspace(-Lx/2+dx,Lx/2-dx,nx)
y = np.linspace(-Ly/2+dy,Ly/2-dy,ny)

X, Y = np.meshgrid(x, y, indexing='ij')

""" Evaluate test function at t0 """
U = testfun(0.0, X, Y)
# plt.figure()
# plt.contourf(u0)

""" Do 1d fft across x coordinate for centerline y """
u = U[:,(ny-1)//2]
plt.figure()
plt.plot(u)

# Spectral parameters 
nb = (9,)
trans_axes = (0,)
uh = FourierTransform(u, trans_axes)
uh_down = DownsampleSpectralArray(uh, nb, (nx,))
uh_up = UpsampleSpectralArray(uh_down, nb, (nx,))

plt.figure()
plt.plot(uh)

# Spectral parameters 
trans_axes = (0,1)
Uh = FourierTransform(U, trans_axes)
Uapp = InverseFourierTransform(Uh, trans_axes)

true_err_max = np.max(abs(U-Uapp))
red_err_max = np.zeros(((nx-1)//2)+1)
runidx = 0
for i in range(1,nx+1,2):
    nb = (i, ny)
    Uh_down = DownsampleSpectralArray(Uh, nb, dims)
    Uh_up = UpsampleSpectralArray(Uh_down, nb, dims)
    Ured_app = InverseFourierTransform(Uh_up, trans_axes).real
    red_err_max[runidx] = np.max(abs(U-Ured_app))
    runidx = runidx + 1
    
plt.figure()
plt.plot([0,nx-1],[true_err_max,true_err_max])
plt.plot(np.linspace(0,nx-1,(nx-1)//2+1), red_err_max)

plt.figure()
plt.loglog(red_err_max[:-1])

K = FourierFrequencies(nx, dx, Lx)
L = FourierFrequencies(ny, dy, Ly)

K2d, L2d = np.meshgrid(K,L, indexing='ij')

plt.figure()
plt.plot(K)
plt.plot(L)
