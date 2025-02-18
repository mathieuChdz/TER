import numpy as np
import matplotlib.pyplot as plt
import time

def cpu_matrix_multiplication(size):
    """
    Multiplies two random matrices on CPU.
    
    Args:
        size (int): The size of the square matrices to multiply.
    
    Returns:
        np.ndarray: Result of the matrix multiplication.
    """
    A = np.random.rand(size, size).astype(np.float32)
    B = np.random.rand(size, size).astype(np.float32)
    
    start_time = time.time()
    
    C = np.dot(A, B)
    
    end_time = time.time()
    return end_time - start_time

def multiple_multiplications(n, max_size):
    """
    Multiplies n random matrices of increasing size on CPU.

    Args:
        n (int): The number of matrix multiplications to perform.
        max_size (int): The maximum size of the square matrices to multiply.
    
    Returns:
        tuple: Two lists containing the size of the matrices and the time taken to multiply them.
    """
    x = []
    y = []
    for size in range(0, max_size, max_size // n): :
        time = cpu_matrix_multiplication(size)
        x.append(size)
        y.append(time)
    return x, y

def draw(x, y, filename):
    """
    Draws a plot of the time taken to multiply matrices of increasing size on CPU.

    Args:
        x (list): The size of the matrices.
        y (list): The time taken to multiply the matrices.
        filename (str): The name of the file to save the plot.
    """
    plt.plot(x, y)
    plt.xlabel('Matrix size')
    plt.ylabel('Time (s)')
    plt.title('Matrix multiplication on CPU')
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    x, y = multiple_multiplications(40, 20000)
    draw(x, y, 'cpu_matrix_multiplication.png')