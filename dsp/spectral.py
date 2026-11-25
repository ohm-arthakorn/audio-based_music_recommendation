import numpy as np

def frame_signal(x, frame_len, hop):
    """
    Framing and hopping signal for STFT.    

    Args:
        x: input signal for framing and hopping
        frame_len: frame size 
        hop: hop size

    Returns: 
        The np.array contains each frame of signal.
    """

    N = len(x)    # check number of sample
    x_out = []
    n_frame = ((N-frame_len)//hop)+1 # the number of frame is ((sample-frame_size)/hop_size)+1

    for m in range(0, n_frame):     
        x_out.append(x[m*hop: m*hop + frame_len])

    return np.array(x_out)

def stft(x, frame_len, hop):
    """
        Compute the Short time Fourier Transform (STFT) of an audio signal.

        Args:
        x: 1D array representing the input time-domain signal.
        frame_len: Number of samples each segments.
        hop: Number of samples between adjacent frames.

        Returns:
        S: 2D array rows representing a frames of signal (time duration) and 
        column representing a FFT of signal.
    """
    frames = frame_signal(x, frame_len, hop)
    frames_window = frames * np.hanning(frame_len)

    # Performs FFT each rows (axis=1 means row.)
    S = np.abs(np.fft.rfft(frames_window, axis=1))
    return S

if __name__ == "__main__":
    test_1 = frame_signal(np.arange(10), 4, 2)

    # Test frame_signal function.
    print(test_1)
    print(f"Check with np.allclose() : {np.allclose(test_1, [[0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]])}")
