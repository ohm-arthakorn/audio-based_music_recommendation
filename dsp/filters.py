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
    
if __name__ == "__main__":
    # Test Program
    x = np.array([1,2,3,4])
    h = np.array([0.5, 0.5])

    print("Test program.")
    print(convolve(x,h))

    print("Verify whether the value of np.convolve(x,h) = convolve(x,h) by using np.allclose()")
    print(f"{np.allclose(convolve(x,h),np.convolve(x,h))}")

