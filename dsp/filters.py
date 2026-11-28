import numpy as np

def convolve(x, h):
    """
    Compute convolution with the input signal (x) and the filter coefficients (h).

    Args:
        x: The input signal for compute convolution.
        h: The filter coefficients for filter.

    Returns:
        The output array of Convolution.
    """
    y_size = len(x)+len(h)-1    # Check size of results output with len(x) + len(h) - 1 
    y = np.zeros(y_size)        # Create output array for collect value from Convolution.
    for n in range(y_size):
        for k in range(0, len(h)):
            i = n - k
            if (i<0) or (i>len(x)-1): 
                continue
            y[n] += h[k] * x[i]
    return y

def fft_convolve(x, h):
    """
        Compute convolution with convolution theorem, convolve(x,h) = IDFT(DFT(x) * DFT(h)) 

        Args:
            x: Input signal array. 
            h: The filter coefficients or window (In ThinkDSP).

        Returns:
            The output array of Convolution.
    """
    n = len(x)+len(h)-1
    X = np.fft.fft(x, n=n)
    H = np.fft.fft(h, n=n)
    Y = X * H
    y = np.fft.ifft(Y).real
    return y

if __name__ == "__main__":
    # Test Program
    x = np.array([1,2,3,4])
    h = np.array([0.5, 0.5])

    # print("Test program.")
    # print(convolve(x,h))

    # print("Verify whether the value of np.convolve(x,h) = convolve(x,h) by using np.allclose()")
    # print(f"{np.allclose(convolve(x,h),np.convolve(x,h))}")

    # Test fft_convolve()
    print("Test fft_convolve()")
    print(fft_convolve(x,h))
    
    print("Verify whether the value of np.convolve(x,h) = fft_convolve(x,h) by using np.allclose()")
    print(f"{np.allclose(fft_convolve(x,h),np.convolve(x,h))}")
