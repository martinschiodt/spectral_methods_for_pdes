import numpy as np
from numpy.fft import fftn, ifftn, fftshift, ifftshift, fftfreq
from scipy import signal
import matplotlib.pyplot as plt


""" ------------ Fourier transform functions ---------------- """
def FourierTransform(u, trans_axes):
    """ Centered Fourier transform 
    
    ------ Input ------
    u: data to be transformed
    trans_axes: coordinates in which we want to transform
    """
    return fftshift(fftn(u, axes=trans_axes), axes=trans_axes)

def InverseFourierTransform(uh, trans_axes):
    """ Inverse of Centered Fourier Transform """
    return ifftn(ifftshift(uh, axes=trans_axes), axes=trans_axes)

def FourierFrequencies(nx, dx, Lx):
    """ 
    Computes wave frequencies on periodic domain´
    
    ------ Input ------
    nx: number of spatial grid points 
    dx: step size with equidistant spacing
    Lx: Domain length
    """
    K = fftshift(fftfreq(nx, d=dx)) * Lx
    return K

""" ------------ Up and downsample functions ---------------- """
def DownsampleSpectralArray(uh, nb, dims):
    uh_down = np.zeros(nb, dtype='complex')
    if len(nb) == 1:
        uh_down[:] = uh[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1]
    if len(nb) == 2:
        uh_down[:,:] = uh[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1,
                          (dims[1]-1)//2-nb[1]//2:(dims[1]-1)//2+nb[1]//2+1]
    if len(nb) == 3:
        uh_down[:,:,:] = uh[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1,
                            (dims[1]-1)//2-nb[1]//2:(dims[1]-1)//2+nb[1]//2+1,
                            (dims[2]-1)//2-nb[2]//2:(dims[2]-1)//2+nb[2]//2+1]
    return uh_down

def UpsampleSpectralArray(uh, nb, dims):
    uh_up = np.zeros(dims, dtype='complex')
    if len(nb) == 1:    
        uh_up[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1] = uh[:]
    if len(nb) == 2:    
        uh_up[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1,
              (dims[1]-1)//2-nb[1]//2:(dims[1]-1)//2+nb[1]//2+1] = uh[:,:]
    if len(nb) == 3:    
        uh_up[(dims[0]-1)//2-nb[0]//2:(dims[0]-1)//2+nb[0]//2+1,
              (dims[1]-1)//2-nb[1]//2:(dims[1]-1)//2+nb[1]//2+1,
              (dims[2]-1)//2-nb[2]//2:(dims[2]-1)//2+nb[2]//2+1] = uh[:,:,:]
    return uh_up

""" ------------ Convolution sum functions ---------------- """
def DirectConvolutionSum(vh,wh):
    """ 
    Directly computes the convolution sum of vh and wh (Kopriva Algorithm 48)
    
    ------ Input ------
    vh = {vh_k}_{k=-N/2}^{N/2}: Fourier Coefficients for function v
    wh = {wh_k}_{k=-N/2}^{N/2}: Fourier Coefficients for function w
    len(vh) = len(wh) = N+1, requires that N is even
    """
    N = int(len(vh)-1)
    Nhalf = int(N/2)
    conv_sum = np.zeros(vh.shape, dtype='complex')

    for k in range(-Nhalf,Nhalf+1):
        bot = int(np.max([-N/2, k-N/2]))
        top = int(np.min([N/2, k+N/2]))
        for p in range(bot,top):
            conv_sum[Nhalf+k] = conv_sum[Nhalf+k] + vh[Nhalf+k-p]*wh[Nhalf+p]
    
    return conv_sum

def FastConvolutionSum(vh, wh):
    conv_sum = signal.fftconvolve(vh,wh, mode='same')    
    return conv_sum
