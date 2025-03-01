import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt
import time
import os
import sys

POINTS_NUM = 20
MAX_SIZE = 40
USE_GPU = True # Pour quand le GPU n'est pas disponible (mettre à False)

def Fibonacci_CPU(n):
    """
    Calculates the nth Fibonacci number.

    Args:
        n (int): The index of the Fibonacci number to calculate.

    Returns:
        int: The nth Fibonacci number.
    """
    if n < 0:
        return None
    elif n == 0:
        return 0
    elif n == 1 or n == 2:
        return 1
    else:
        return Fibonacci_CPU(n-1) + Fibonacci_CPU(n-2)
    
def compile_Fibonacci_GPU():
    """
    Compiles the CUDA code for calculating Fibonacci numbers.

    Returns:
        function: The compiled CUDA kernel function.
    """
    cuda_code = """
    __device__ int fibonacci(int n) {
        if (n <= 0) return 0;
        if (n == 1) return 1;
        return fibonacci(n - 1) + fibonacci(n - 2);
    }

    __global__ void fibonacci_kernel(int *d_result, int n) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_result = fibonacci(n);
        }
    }
    """
    mod = SourceModule(cuda_code)
    fibonacci_kernel = mod.get_function("fibonacci_kernel")
    return fibonacci_kernel

def Fibonacci_GPU(n, fibonacci_kernel):
    """
    Calculates the nth Fibonacci number using the GPU.

    Args:
        n (int): The index of the Fibonacci number to calculate.
        fibonacci_kernel (function): The compiled CUDA kernel function.
    
    Returns:
        float: The time taken to calculate the nth Fibonacci number.
    """
    d_result = cuda.mem_alloc(np.int32(0).nbytes)
    start = time.time()
    fibonacci_kernel(d_result, np.int32(n), block=(1,1,1), grid=(1,1))
    cuda.Context.synchronize()
    end = time.time()
    return end - start

    
def multiple_Fibonacci_CPU(n_number, max_size):
    """
    Calculates n Fibonacci numbers of increasing size.

    Args:
        n_number (int): The number of Fibonacci numbers to calculate.
        max_size (int): The maximum index of the Fibonacci numbers to calculate.

    Returns:
        tuple: Two lists containing the index of the Fibonacci numbers and the time taken to calculate them.
    """
    x = []
    y = []
    for size in range(0, max_size, max_size // n_number):
        start_time = time.time()
        Fibonacci_CPU(size)
        end_time = time.time()
        x.append(size)
        y.append(end_time - start_time)
    return x, y, "CPU"

def multiple_Fibonacci_GPU(n_number, max_size):
    """
    Calculates n Fibonacci numbers of increasing size using the GPU.

    Args:
        n_number (int): The number of Fibonacci numbers to calculate.
        max_size (int): The maximum index of the Fibonacci numbers to calculate.

    Returns:
        tuple: Two lists containing the index of the Fibonacci numbers and the time taken to calculate them.
    """
    fibonacci_kernel = compile_Fibonacci_GPU()
    x = []
    y = []
    for size in range(0, max_size, max_size // n_number):
        time = Fibonacci_GPU(size, fibonacci_kernel)
        x.append(size)
        y.append(time)
    return x, y, "GPU"
    
def draw(points1, points2, filename):
    """
    Draws a graph comparing the time taken to calculate Fibonacci numbers on CPU and GPU.

    Args:
        points1 (tuple): Two lists containing the index of the Fibonacci numbers and the time taken to calculate them on CPU.
        points2 (tuple): Two lists containing the index of the Fibonacci numbers and the time taken to calculate them on GPU.
        filename (str): The name of the file to save the graph to.
    """
    plt.plot(points1[0], points1[1], label=points1[2])
    if points2 is not None:
        plt.plot(points2[0], points2[1], label=points2[2])
    plt.xlabel('Fibonacci number')
    plt.ylabel('Time (s)')
    plt.title('Fibonacci calculation on CPU and GPU')
    plt.legend()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    if not os.path.exists('graphs'):
        os.makedirs('graphs')
    points1 = multiple_Fibonacci_CPU(POINTS_NUM, MAX_SIZE)
    if USE_GPU:
        points2 = multiple_Fibonacci_GPU(POINTS_NUM, MAX_SIZE)
    else :
        points2 = None
    draw(points1, points2, "graphs/fibonacci.png")