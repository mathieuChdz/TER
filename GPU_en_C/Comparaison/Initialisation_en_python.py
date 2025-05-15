import cupy as cp

a = cp.array([1, 2, 3], dtype=cp.float32)
b = cp.array([4, 5, 6], dtype=cp.float32)
c = a + b  # CuPy gère tout automatiquement