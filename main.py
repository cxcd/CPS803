#import os
from numba import jit
import numpy as np

x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting


def main():
    print(go_fast(x))

if __name__ == '__main__':
    main()
    #main(train_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data.txt')