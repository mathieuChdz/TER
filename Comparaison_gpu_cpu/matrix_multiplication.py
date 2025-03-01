import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import os

POINTS_NUM = 40
MAX_SIZE = 20000
USE_GPU = True # Pour quand le GPU n'est pas disponible (mettre à False)

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

def gpu_matrix_multiplication(size):
    """
    Multiplies two random matrices on GPU.
    
    Args:
        size (int): The size of the square matrices to multiply.
    
    Returns:
        cp.ndarray: Result of the matrix multiplication on GPU.
    """
    A = cp.random.rand(size, size, dtype=cp.float32)
    B = cp.random.rand(size, size, dtype=cp.float32)
    
    cp.cuda.Device(0).synchronize()
    start_time = time.time()
    
    C = cp.dot(A, B)
    
    cp.cuda.Device(0).synchronize()
    end_time = time.time()

    return end_time - start_time

def multiple_cpu_multiplications(n, max_size):
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
    for size in range(0, max_size, max_size // n):
        time = cpu_matrix_multiplication(size)
        x.append(size)
        y.append(time)
    return x, y, "CPU"

def multiple_gpu_multiplications(n, max_size):
    """
    Multiplies n random matrices of increasing size on GPU.

    Args:
        n (int): The number of matrix multiplications to perform.
        max_size (int): The maximum size of the square matrices to multiply.
    
    Returns:
        tuple: Two lists containing the size of the matrices and the time taken to multiply them.
    """
    x = []
    y = []
    for size in range(0, max_size, max_size // n):
        time = gpu_matrix_multiplication(size)
        x.append(size)
        y.append(time)
    return x, y, "GPU"

def draw(points1, points2, filename):
    """
    Draws a graph comparing the time taken to multiply matrices on CPU and GPU.

    Args:
        points1 (tuple): Two lists containing the size of the matrices and the time taken to multiply them.
        points2 (tuple): Two lists containing the size of the matrices and the time taken to multiply them.
        filename (str): The name of the file to save the plot.
    """
    plt.plot(points1[0], points1[1], label=points1[2])
    if points2 is not None:
        plt.plot(points2[0], points2[1], label=points2[2])
    plt.xlabel('Matrix size')
    plt.ylabel('Time (s)')
    plt.title('Matrix multiplication on CPU and GPU')
    plt.legend()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    points1 = multiple_cpu_multiplications(POINTS_NUM, MAX_SIZE)
    if USE_GPU:
        points2 = multiple_gpu_multiplications(POINTS_NUM, MAX_SIZE)
    else:
        points2 = None
    draw(points1, points2, 'graphs/matrix_multiplication.png')