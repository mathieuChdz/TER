# Algorithmes à tester

## Algorithmes à forte accélération sur **GPU**  
1. **Multiplication de matrices**  
2. **Transformations de Fourier rapides (FFT)**  
   - À explorer : À quoi sert le calcul de la FFT ?  
3. **Tri massivement parallèle** (Exemple : **Bitonic Sort**)  
   - [GeeksforGeeks - Bitonic Sort](https://www.geeksforgeeks.org/bitonic-sort/)  
4. **Simulation d’automates cellulaires ou de systèmes multi-agents**  

## Algorithmes plus adaptés au **CPU**  
5. **Accès mémoire irrégulier**  
   - Exemple : Recherche d’éléments dans une grande liste chaotique (Recherche d’éléments dans une grande liste chaotique -> accès aléatoire dans un tableau).  
6. **Opérations dépendantes séquentiellement**  
   - Exemple : Algorithmes **récursifs** (ex. **suite de Fibonacci naïve**).  
7. **Branchements conditionnels nombreux**  
   - Parcours d’un graphe avec beaucoup de conditions.  
   - Problème du **Thread Divergence** sur GPU (mauvaise optimisation).
   
---

# Comparaison GPU vs CPU

| Algorithme  | Meilleur sur **GPU** | Meilleur sur **CPU** | FAIT |
|-------------|:----------------:|:----------------:| :----------------:|
| **Multiplication de matrices** | X | / | OUI |
| **FFT (Fast Fourier Transform)** | X | / | NON |
| **Bitonic Sort** | X | / | OUI |
| **Simulation d'automates cellulaires** | X | / | NON |
| **Accès mémoire irrégulier** | / | X | OUI |
| **Algorithmes récursifs (ICI Fibonacci)** | / | X | OUI |
| **Parcours d’un graphe avec beaucoup de conditions** | / | X | NON |
| **Somme de petites quantités de données** | / | X | NON |

---

# Implémentation en Python

#### Technologies utilisées

- **NumPy** : Une bibliothèque pour le calcul scientifique en Python. Elle permet de travailler avec des tableaux multidimensionnels et fournit une large collection de fonctions mathématiques.
- **CuPy** : Une bibliothèque compatible avec NumPy pour le calcul sur GPU. Elle permet d'exécuter des opérations sur GPU en utilisant une syntaxe similaire à NumPy.

#### Exemple de code pour le CPU (Multiplication de matrices)

```python
import numpy as np
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
   
    return C
```

#### Exemple de code pour le GPU (Multiplication de matrices)

```python
import cupy as cp
import time

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
    
    return C
```

```python
# Execution with DATA
size = 1000
time_taken_CPU = cpu_matrix_multiplication(size)
time_taken_GPU = gpu_matrix_multiplication(size)

print(f"CPU: {time_taken_CPU:.2f} seconds")
print(f"GPU: {time_taken_GPU:.2f} seconds")
```