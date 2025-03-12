import numpy as np
import cupy as cp
import time
import gpu_consumption as gc

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
    
    print(f"CPU: Computation completed in {end_time - start_time:.2f} seconds")
    return C

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
    
    print(f"GPU: Computation completed in {end_time - start_time:.2f} seconds")
    return C

if __name__ == "__main__":
    size = 100000
    gc.init_NVML()
    begin_power = gc.get_total_power_usage()
    result_gpu = gpu_matrix_multiplication(size)
    end_power = gc.get_total_power_usage()
    gc.stop_NVML()
    
    result_cpu = cpu_matrix_multiplication(size)

    print(f"GPU: Power consumption: {end_power - begin_power:.2f} watts")