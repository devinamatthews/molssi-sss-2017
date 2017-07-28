import numpy as np
import os


def array_to_c(array, filename):
    np.savetxt(filename, array.ravel(), delimiter=' ')


if __name__ == "__main__":

    # A small example and test
    arr = np.random.rand(20, 20)
    filename = "test_array.data"

    # Write the file
    array_to_c(arr, filename)

    # Give a test
    arr_test = np.loadtxt(filename, delimiter=" ")
    print("Array read/write matches: %s" % np.allclose(arr_test, arr.ravel()))

    # Cleanup the temporary file
    os.remove(filename)
