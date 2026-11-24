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

if __name__ == "__main__":
    test_1 = frame_signal(np.arange(10), 4, 2)

    # Test frame_signal function.
    print(test_1)
    print(f"Check with np.allclose() : {np.allclose(test_1, [[0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]])}")
