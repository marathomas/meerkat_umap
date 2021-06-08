#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
Created on Created on Tue May  4 12:20:04 2021

Collection of spectrogramming functions

@author: marathomas
"""
import librosa
import numpy as np

def generate_mel_spectrogram(data, rate, n_mels, window, fft_win , fft_hop, fmax):
    
    """
    Function that generates mel spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    rate: numeric(integer)
          samplerate in Hz
    n_mels: numeric (integer)
            number of mel bands
    window: string
            spectrogram window generation type ('hann'...)
    fft_win: numeric (float)
             window length in s
    fft_hop: numeric (float)
             hop between window start in s 

    Returns
    -------
    result : 2D np.array
             Mel-transformed spectrogram, dB scale

    Example
    -------
    >>> 
    
    """
    n_fft  = int(fft_win * rate) 
    hop_length = int(fft_hop * rate) 
        
    s = librosa.feature.melspectrogram(y = data ,
                                       sr = rate, 
                                       n_mels = n_mels , 
                                       fmax = fmax, 
                                       n_fft = n_fft,
                                       hop_length = hop_length, 
                                       window = window, 
                                       win_length = n_fft)

    spectro = librosa.power_to_db(s, ref=np.max)

    return spectro


def generate_stretched_mel_spectrogram(data, sr, duration, n_mels, window, fft_win , fft_hop):
    """
    Function that generates stretched mel spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    sr: numeric(integer)
          samplerate in Hz
    duration: numeric (float)
              duration of audio in seconds
    n_mels: numeric (integer)
            number of mel bands
    window: string
            spectrogram window generation type ('hann'...)
    fft_win: numeric (float)
             window length in s
    fft_hop: numeric (float)
             hop between window start in s 

    Returns
    -------
    result : 2D np.array
             stretched, mel-transformed spectrogram, dB scale
    -------
    >>> 
    
    """
    n_fft  = int(fft_win * sr) 
    hop_length = int(fft_hop * sr) 
    stretch_rate = duration/MAX_DURATION
    
    # generate normal spectrogram (NOT mel transformed)
    D = librosa.stft(y=data, 
                     n_fft = n_fft,
                     hop_length = hop_length,
                     window=window,
                     win_length = n_fft
                     )
    
    # Stretch spectrogram using phase vocoder algorithm
    D_stretched = librosa.core.phase_vocoder(D, stretch_rate, hop_length=hop_length) 
    D_stretched = np.abs(D_stretched)**2
    
    # mel transform
    spectro = librosa.feature.melspectrogram(S=D_stretched,  
                                            sr=sr,
                                            n_mels=n_mels,
                                            fmax=4000)
        
    # Convert to db scale
    s = librosa.power_to_db(spectro, ref=np.max)

    return s


def generate_freq_spectrogram(data, sr, window, fft_win , fft_hop):
    """
    Function that generates freq spectrogram from audio data using librosa functions

    Parameters
    ----------
    data: 1D numpy array (float)
          Audio data
    sr: numeric(integer)
          samplerate in Hz
    window: string
            spectrogram window generation type ('hann'...)
    fft_win: numeric (float)
             window length in s
    fft_hop: numeric (float)
             hop between window start in s 

    Returns
    -------
    result : 2D np.array
             DB scale spectrogram
    -------
    >>> 
    
    """
    n_fft  = int(fft_win * sr) 
    hop_length = int(fft_hop * sr) 
    
    # generate normal spectrogram (NOT mel transformed)
    D = librosa.stft(y=data, 
                     n_fft = n_fft,
                     hop_length = hop_length,
                     window=window,
                     win_length = n_fft
                     )
        
    # Convert to db scale
    S = np.abs(D)
    s = librosa.power_to_db(S**2, ref=np.max)
    
    # Equalize frequency resolution

    return s

def generate_ampli_spectrogram(data, rate, window, fft_win , fft_hop):
    n_fft  = int(fft_win * rate) 
    hop_length = int(fft_hop * rate) 
      
    s = librosa.stft(y=data, # spectrogramming
                     n_fft = n_fft,
                     hop_length = hop_length,
                     window=window,
                     win_length = n_fft
                     )

    #power_to_db(np.abs(D)**2)

    return np.abs(s)