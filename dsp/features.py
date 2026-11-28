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

if __name__ == "__main__":
    F = [0, 700, 1000]
    print("Test Program.")
    for f in F:
        print(f"Change {f} Hz to {round(hz_to_mel(f),2)} mel.")
        