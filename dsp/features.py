import numpy as np

def mel_to_hz(m):
    """
    Convert the input signal array in Mel scale to linear frequency scale. 

    Args:
        m: The input signal array in Mel scale (mel)
    
    Returns:
        f: The output signal array in linear frequency scale (Hz).
    """

    f = 700 * (10 ** (m/2595) - 1)
    return f

def hz_to_mel(f):
    """
    Convert the input signal array in linear frequency scale to Mel scale.

    Args:
        f: The input signal array in linear frequency scale (Hz)

    Returns:
        m: The output signal array in Mel scale (mel)
    """

    m = 2595 * np.log10(1 + (f/700))
    return m

def mel_filterbank(sr, n_fft, n_mels):
    """
    Compute Mel Filterbank
    
    Args:
        sr: sampling rate for separate each frequency bins and maximum of filterbank. (Hz)
        n_fft: frame length for FFT (Sample)
        n_mels: number of filters 
    
    Returns:
        fb: numpy array shape (n_mels, (n_fft // 2) + 1)
    """

    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    mel_pts = np.linspace(hz_to_mel(0), hz_to_mel(sr/2), n_mels+2)
    hz_pts = mel_to_hz(mel_pts)
    fb = []

    for i in range(n_mels):
        left, center, right = hz_pts[i], hz_pts[i+1], hz_pts[i+2] 
        w1 = (freqs - left)/(center - left)
        w2 = (freqs - right)/(center - right)
        fb.append(np.maximum(0, np.minimum(w1, w2)))

    return np.array(fb)

if __name__ == "__main__":
    F = [0, 700, 1000]
    print("Test Program.")
    for f in F:
        print(f"Change {f} Hz to {round(hz_to_mel(f),2)} mel.")
        