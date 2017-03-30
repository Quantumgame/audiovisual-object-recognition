from math import *
import numpy as np
import sigproc
import audio_noise
from scipy.fftpack import dct
import scipy.io.wavfile as wav

#Reads signal from wav file
def read(filename):
    """
    Reads signal from wav file
    :param filename: string.
    """
    (fs,signal) = wav.read(filename)

    if signal.shape[1] > 1:
        signal = signal[:,0]

    return (fs,signal)

def get_SNR(signal, noise):
    """
    Returns the SNR (in dB) from signal and noise.
    :param signal: array.
    :param noise: array.
    :param noise_type: string. 'white' or 'pink'
    """

    assert len(signal) == len(noise)

    I_s = 0
    I_n = 0

    for i in range(len(signal)):
        I_s += signal[i] ** 2
        I_n += noise[i] ** 2

    return 10*log(I_s/I_n, 10)

def get_noisy_signal(signal, SNR, noise_type='white'):
    """
    Adds noise to given signal based o desired noise type and SNR level.
    :param signal: array. Signal to add noise to.
    :param SNR: int. Signal to Noise Ration (in dB).
    :param noise_type: string. 'white' or 'pink'
    """
    print SNR

    #If lim SNR -> inf, almost pure signal, returns signal as it is
    if SNR == float('inf'):
        return signal


    noise = None
    if noise_type == 'pink':
        noise = audio_noise.pink(shape = len(signal))
    else:
        noise = audio_noise.white(shape = len(signal))

    #If lim SNR -> -inf, almost pure noise, returns only noise
    if SNR == float('-inf'):
        return np.asarray(noise)


    mean = np.mean(signal)
    signal = [s - mean for s in signal]    

    mean = np.mean(noise)
    noise = [n - mean for n in noise]

    desired_var = np.var(signal) / (10 ** (SNR/10))

    factor = sqrt(desired_var / np.var(noise))
    noise = [n * factor for n in noise]

    for i in range(len(signal)):
        signal[i] += noise[i]

    return np.asarray(signal)



def t60_plot(signal, fs, window_size, step):
    """
    Computes the time and energy of T60 for given signal. Used for ploting pourposes.
    :param signal: array.
    :param fs: int. Signal's frequency.
    :param window_size: int. Number of samples to consider for sliding window of FT.
    :param step_size: int. Step size to consider for sliding window of FT.
    """
    decay = 60 #in dB
    
    samples_per_slice = int(fs * window_size)
    #step = math.ceil(samples_per_slice*step)

    squared = np.multiply(signal,signal)

    energy = []

    #with step = 1
    e = sum(squared[0:samples_per_slice])
    energy.append(e)

    for i in range(1, len(squared) - samples_per_slice + 1):
        e -= squared[i - 1]
        e += squared[i + samples_per_slice - 1]
        energy.append(e)

    I_0 = np.argmax(energy)
    I = None

    for i in range(I_0, len(signal) - samples_per_slice + 1):
        #print energy[I_0], energy[i]
        if(10*log(energy[I_0]/energy[i], 10) >= decay):
            I = i
            break

    return (I - I_0)*1000.0/fs, I_0, I, energy

def rtn(signal, fs, window_size, decay):
    """
    Computes time, clipped signal and energy of desired reverberation time decay.
    :param signal: array. The audio signal to clip.
    :param fs: int. The signal's frequency.
    :param window_size: int. Number of samples to consider for sliding window of FT.
    :param decay: int. Desired decay (in dB) for clipping.
    """
    
    samples_per_slice = int(fs * window_size)

    squared = [s**2 for s in signal]

    energy = []

    #with step = 1
    e = sum(squared[0:samples_per_slice])
    energy.append(e)

    for i in range(1, len(squared) - samples_per_slice + 1):

        e -= squared[i - 1]
        e += squared[i + samples_per_slice - 1]

        energy.append(e)

    #Getting start and end of signal given desired decay
    I_0 = np.argmax(energy)
    I = []

    for d in decay:
        for i in range(I_0, len(signal) - samples_per_slice + 1):
            if(10*log(energy[I_0]/energy[i], 10) >= d):
                I.append(i)
                break

        
    #padding with index of last sample of signal in case the later decay levels do not exist in signal
    I.extend([len(signal) - samples_per_slice + 1]*(len(decay) - len(I)))

    time = []
    for i in range(len(I)):
        t = ((I[i] - I_0)*1000.0/fs)
        time.append(t)

    I = [i - I_0 for i in I]

    return time, signal[I_0:], I


def fft_iza(signal,fs, window_size, step):
    """
    Computes flattened FT for entire signal.
    :param signal: array. Signal from which cft is computed.
    :param fs: int. Signal's frequency.
    :param window_size: int. Number of samples to consider for sliding window of FT.
    :param step_size: int. Step size to consider for sliding window of FT.
    """

    samples_per_slice = int(fs * window_size)
    samples_per_step = int(fs * step)

    coefs = []

    for i in range(0, len(signal) - samples_per_slice + 1, samples_per_step):
        c = np.fft.rfft(signal[i:i + samples_per_slice])

        coefs.extend(np.absolute(c))

    return coefs


def cft(signal,fs, window_size, step):
    """
    Computes compact Fourier Transform for given signal.
    :param signal: array. Signal from which cft is computed.
    :param fs: int. Signal's frequency.
    :param window_size: int. Number of samples to consider for sliding window of FT.
    :param step_size: int. Step size to consider for sliding window of FT.
    """
    samples_per_slice = int(fs * window_size)
    samples_per_step = int(fs * step)

    coefs = []

    for i in range(0, len(signal) - samples_per_slice + 1, samples_per_step):
        c = np.fft.rfft(signal[i:i + samples_per_slice])
        coefs.append(np.absolute(c))

    coefs = np.sum(coefs, axis = 0)

    max_value = np.max(coefs)

    coefs = coefs.tolist()
    
    #normalization
    coefs = [c/max_value for c in coefs]

    return coefs


def amplify(signal):
    """
    Amplify signal.
    :param signal: array. Signal to be amplified.
    """
    max_signal = max(abs(signal))
    factor = 1.0/max_signal
    return signal*factor
