"""
Module for Wavelet decomposition computation and utils
"""

import scipy.io.wavfile as wav
from mfcc import *
from subprocess import call
import numpy as np
import pylab as mpl
import pywt

"""
Wavelet classes:
Morlet
MorletReal
MexicanHat
Paul2      : Paul order 2
Paul4      : Paul order 4
DOG1       : 1st Derivative Of Gaussian
DOG4       : 4th Derivative Of Gaussian
Haar       : Unnormalised version of continuous Haar transform
HaarW      : Normalised Haar

Usage e.g.
wavelet=Morlet(data, largestscale=2, notes=0, order=2, scaling="log")
 data:  Numeric array of data (float), with length ndata.
        Optimum length is a power of 2 (for FFT)
        Worst-case length is a prime
 largestscale:
        largest scale as inverse fraction of length
        scale = len(data)/largestscale
        smallest scale should be >= 2 for meaningful data
 notes: number of scale intervals per octave
        if notes == 0, scales are on a linear increment
 order: order of wavelet for wavelets with variable order
        [Paul, DOG, ..]
 scaling: "linear" or "log" scaling of the wavelet scale.
        Note that feature width in the scale direction
        is constant on a log scale.
        
Attributes of instance:
wavelet.cwt:       2-d array of Wavelet coefficients, (nscales,ndata)
wavelet.nscale:    Number of scale intervals
wavelet.scales:    Array of scale values
                   Note that meaning of the scale will depend on the family
wavelet.fourierwl: Factor to multiply scale by to get scale
                   of equivalent FFT
                   Using this factor, different wavelet families will
                   have comparable scales
"""

class Cwt:
    """
    Base class for continuous wavelet transforms
    Implements cwt via the Fourier transform
    Used by subclass which provides the method wf(self,s_omega)
    wf is the Fourier transform of the wavelet function.
    Returns an instance.
    """

    fourierwl=1.00

    def _log2(self, x):
        # utility function to return (integer) log2
        return int( np.log(float(x))/ np.log(2.0)+0.0001 )

    def __init__(self, data, largestscale=1, notes=0, order=2, scaling='linear'):
        """
        Continuous wavelet transform of data

        data:    data in array to transform, length must be power of 2
        notes:   number of scale intervals per octave
        largestscale: largest scale as inverse fraction of length
                 of data array
                 scale = len(data)/largestscale
                 smallest scale should be >= 2 for meaningful data
        order:   Order of wavelet basis function for some families
        scaling: Linear or log
        """
        ndata = len(data)
        self.order=order
        self.scale=largestscale
        self._setscales(ndata,largestscale,notes,scaling)
        self.cwt= np.zeros((self.nscale,ndata), np.complex64)
        omega= np.array(range(0,ndata/2)+range(-ndata/2,0))*(2.0*np.pi/ndata)
        datahat=np.fft.fft(data)
        self.fftdata=datahat
        #self.psihat0=self.wf(omega*self.scales[3*self.nscale/4])
        # loop over scales and compute wvelet coeffiecients at each scale
        # using the fft to do the convolution
        for scaleindex in range(self.nscale):
            currentscale=self.scales[scaleindex]
            self.currentscale=currentscale  # for internal use
            s_omega = omega*currentscale
            psihat=self.wf(s_omega)
            psihat = psihat *  np.sqrt(2.0*np.pi*currentscale)
            convhat = psihat * datahat
            W    = np.fft.ifft(convhat)
            self.cwt[scaleindex,0:ndata] = W 
        return
    
    def _setscales(self,ndata,largestscale,notes,scaling):
        """
        if notes non-zero, returns a log scale based on notes per ocave
        else a linear scale
        (25/07/08): fix notes!=0 case so smallest scale at [0]
        """
        if scaling=="log":
            if notes<=0: notes=1 
            # adjust nscale so smallest scale is 2 
            noctave=self._log2( ndata/largestscale/2 )
            self.nscale=notes*noctave
            self.scales=np.zeros(self.nscale,float)
            for j in range(self.nscale):
                self.scales[j] = ndata/(self.scale*(2.0**(float(self.nscale-1-j)/notes)))
        elif scaling=="linear":
            nmax=ndata/largestscale/2
            self.scales=np.arange(float(2),float(nmax))
            self.nscale=len(self.scales)
        else: raise ValueError, "scaling must be linear or log"
        return
    
    def getdata(self):
        """
        returns wavelet coefficient array
        """
        return self.cwt
    def getcoefficients(self):
        return self.cwt
    def getpower(self):
        """
        returns square of wavelet coefficient array
        """
        return (self.cwt* np.conjugate(self.cwt)).real
    def getscales(self):
        """
        returns array containing scales used in transform
        """
        return self.scales
    def getnscale(self):
        """
        return number of scales
        """
        return self.nscale

# wavelet classes    
class Morlet(Cwt):
    """
    Morlet wavelet
    """
    _omega0=5.0
    fourierwl=4* np.pi/(_omega0+ np.sqrt(2.0+_omega0**2))
    def wf(self, s_omega):
        H= np.ones(len(s_omega))
        n=len(s_omega)
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat=0.75112554*( np.exp(-(s_omega-self._omega0)**2/2.0))*H
        return xhat

class MorletReal(Cwt):
    """
    Real Morlet wavelet
    """
    _omega0=5.0
    fourierwl=4* np.pi/(_omega0+ np.sqrt(2.0+_omega0**2))
    def wf(self, s_omega):
        H= np.ones(len(s_omega))
        n=len(s_omega)
        for i in range(len(s_omega)):
            if s_omega[i] < 0.0: H[i]=0.0
        # !!!! note : was s_omega/8 before 17/6/03
        xhat=0.75112554*( np.exp(-(s_omega-self._omega0)**2/2.0)+ np.exp(-(s_omega+self._omega0)**2/2.0)- np.exp(-(self._omega0)**2/2.0)+ np.exp(-(self._omega0)**2/2.0))
        return xhat
    
class Paul4(Cwt):
    """
    Paul m=4 wavelet
    """
    fourierwl=4* np.pi/(2.*4+1.)
    def wf(self, s_omega):
        n=len(s_omega)
        xhat= np.zeros(n)
        xhat[0:n/2]=0.11268723*s_omega[0:n/2]**4* np.exp(-s_omega[0:n/2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class Paul2(Cwt):
    """
    Paul m=2 wavelet
    """
    fourierwl=4* np.pi/(2.*2+1.)
    def wf(self, s_omega):
        n=len(s_omega)
        xhat= np.zeros(n)
        xhat[0:n/2]=1.1547005*s_omega[0:n/2]**2* np.exp(-s_omega[0:n/2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class Paul(Cwt):
    """
    Paul order m wavelet
    """
    def wf(self, s_omega):
        Cwt.fourierwl=4* np.pi/(2.*self.order+1.)
        m=self.order
        n=len(s_omega)
        normfactor=float(m)
        for i in range(1,2*m):
            normfactor=normfactor*i
        normfactor=2.0**m/ np.sqrt(normfactor)
        xhat= np.zeros(n)
        xhat[0:n/2]=normfactor*s_omega[0:n/2]**m* np.exp(-s_omega[0:n/2])
        #return 0.11268723*s_omega**2*exp(-s_omega)*H
        return xhat

class MexicanHat(Cwt):
    """
    2nd Derivative Gaussian (mexican hat) wavelet
    """
    fourierwl=2.0* np.pi/ np.sqrt(2.5)
    def wf(self, s_omega):
        a=s_omega**2
        b=s_omega**2/2
        return a* np.exp(-b)/1.1529702

class DOG4(Cwt):
    """
    4th Derivative Gaussian wavelet
    """
    fourierwl=2.0* np.pi/ np.sqrt(4.5)
    def wf(self, s_omega):
        return s_omega**4* np.exp(-s_omega**2/2.0)/3.4105319

class DOG1(Cwt):
    """
    1st Derivative Gaussian wavelet
    """
    fourierwl=2.0* np.pi/ np.sqrt(1.5)
    def wf(self, s_omega):
        dog1= np.zeros(len(s_omega),complex64)
        dog1.imag=s_omega* np.exp(-s_omega**2/2.0)/sqrt(pi)
        return dog1

class DOG(Cwt):
    """
    Derivative Gaussian wavelet of order m
    """
    def wf(self, s_omega):
        try:
            from scipy.special import gamma
        except ImportError:
            print "Requires scipy gamma function"
            raise ImportError
        Cwt.fourierwl=2* np.pi/ np.sqrt(self.order+0.5)
        m=self.order
        dog=1.0J**m*s_omega**m* np.exp(-s_omega**2/2)/ np.sqrt(gamma(self.order+0.5))
        return dog

class Haar(Cwt):
    """
    Continuous version of Haar wavelet
    """

    fourierwl=1.0#1.83129  #2.0
    def wf(self, s_omega):
        haar= np.zeros(len(s_omega),complex64)
        om = s_omega[:]/self.currentscale
        om[0]=1.0  #prevent divide error
        haar.imag=4.0* np.sin(s_omega/4)**2/om
        return haar

class HaarW(Cwt):
    """
    Continuous version of Haar wavelet (norm)
    """
    fourierwl=1.83129*1.2  #2.0
    def wf(self, s_omega):
        haar= np.zeros(len(s_omega),complex64)
        om = s_omega[:]#/self.currentscale
        om[0]=1.0  #prevent divide error
        haar.imag=4.0* np.sin(s_omega/2)**2/om
        return haar

def scalogram(filename, savename):
    """
    Computes and prints scalogram of signal given predefined Wavelet form.
    :param filename: string. String containing the filename/absolute path to the signal to be analysed.
    :param savename: string. String containing the filename/absolute path to save the final scalogram.
    """

    #signal reading
    (rate,signal) = wav.read(filename)

    #ignore other bands for primary treatment
    if signal.shape[1] > 1:
        signal = signal[:,0]

    #clip the signal
    max_energy = max(energy)
    start_frame = 0
    for k in range(len(energy)):
        if energy[k] >= max_energy*0.01:
            start_frame = k
            break

    end_frame = start_frame
    for k in range(start_frame,len(energy)):
        if energy[k] < max_energy*0.001:
            end_frame = k
            break

    if(end_frame == start_frame):
        for k in range(start_frame,len(energy)):
            if energy[k] < max_energy*0.01:
                end_frame = k
                break

    samples_per_frame = rate * 0.01
    signal = signal[start_frame*samples_per_frame:end_frame*samples_per_frame]


    wavelet=DOG4
    maxscale=10
    notes=100
    scaling='log'#"log" #or "linear"
    plotpower2d=True

    Ns=1024
    #limits of analysis
    Nlo=0 
    Nhi=Ns

    # Wavelet transform
    cw=wavelet(signal,maxscale,notes,scaling=scaling)
    scales=cw.getscales()     
    cwt=cw.getdata()
    # power spectrum
    pwr=cw.getpower()
    scalespec=np.sum(pwr,axis=1)/scales # calculate scale spectrum
    # scales
    y=cw.fourierwl*scales
    x=np.arange(Nlo*1.0,Nhi*1.0,1.0)
    
    #mpl.tight_layout()
    mpl.axis('off')
    fig=mpl.figure(1)

    # 2-d coefficient plot
    plotcwt=np.clip(np.fabs(cwt.real), 0., 1000.)
    if plotpower2d: plotcwt=pwr
    im=mpl.imshow(plotcwt,cmap=mpl.cm.jet,extent=[x[0],x[-1],y[-1],y[0]],aspect='auto')
    mpl.ylim(y[0],y[-1])
    theposition=mpl.gca().get_position()

    mpl.tight_layout()
    mpl.savefig(savename)

def dwt_coefs(signal, a_noise, wavelet_name, decomposition_level):
    """
    Compute coeficients from discrete Wavelet transform from pywt.
    :param signal: numpy array. Array containing the signal where the transform should be applied to.
    :param a_noise: int. SNR white noise level to be added to the signal
    :param wavelet_name: string. Wavelet transformation name to apply to the signal.
    :param decomposition_level: int. Desired decomposition wavelet decomposition from signal.
    """

    if a_noise > 0:
        noise = np.random.randn(len(signal))*a_noise + np.mean(signal)
        signal += noise

    coefs = []
    ca = signal
    for i in range(decomposition_level):
        ca, cd = pywt.dwt(ca, wavelet_name)
        coefs.extend(ca)
 
    return coefs
