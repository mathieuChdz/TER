# GPU et environnement de développement

- [Langages](#langages)
- [Python via GPU](#utilisation-gpu-en-python)
    - [Modules python](#modules-python)
    - [LLM](#llm)
- [C/C++ via GPU](#utilisation-gpu-en-c-c++)
- [RAPIDS](#rapids)

---


## Langages :  

### CUDA (Compute Unified Device Architecture)
Cuda est une technologie NVIDIA qui utilise les GPU pour traiter les gros volumes de données en parallèle bien au-delà de ce que peut faire un CPU.

On peut utiliser pour quoi ?

Pour accélèrer les calculs : en effet le GPU est plus rapide que le CPU grâce à ses milliers de cœurs de calculs souvent utilisés pour le machine learning. (exemple : un cpu en géneral possède 24 coeurs alors que notre GPU H100 NVL possède 16 896 cœurs).
Mais aussi les calculs en parallèle traitant des milliers de données en simultané contrairement aux CPU qui utilise des tâches en séquentielle. 

Cuda est utilisable via C/C++, mais aussi en python via des bibliothèques spécifiques:

- Python avec les bibliothèques comme par exemple [CuPy](#cupy).
 Facile utilisation qui simplifie l'accès au GPU. Mais moins de contrôle sur le GPU et la mémoire que d'autres langages comme C/C++ . Mais il existe des solutions comme [PyCuda](#pycuda) qui permet un accès bas niveau, mais plus complexe.
 
- C/C++ avec CUDA (NVIDIA Native) : 
Contrôle total sur le GPU, gestion manuelle de la mémoire et accès direct aux kernels CUDA mais plus complexe que python.

Outils nécessaires pour coder en Python :  
- Driver Nvidia, CUDA Toolkit, Python avec des versions compatibles et les bibliothèques à utiliser également avec une version compatible avec le reste 

Outils nécéssaires pour coder en C/C++ :  
- Driver Nvidia, CUDA Toolkit, Compiler C/C++ avec des versions compatibles  

---
## Utilisation GPU en python

### Modules python

#### CuPy

Quasiment toutes les fonctions de NumPy sont aussi sur CuPy : manipulation de matrices, de polynômes, algèbre linéaire, transformation rapide de fourrier, génération de nombres aléatoires, tests, opérations binaires, fonctions mathématiques, ...  
        
Plusieurs sous-modules de scipy sont également disponibles  
        
Propose également de gérer la mémoire GPU même si le degré de gestion n'est pas aussi avancé que d'autres bibliothèques comme PyCuda ou le C/C++ directement. Pour une utilisation "normale" de CuPy (utiliser comme NumPy ou Scipy), il n'y a normalement pas besoin de gérer sois-même la mémoire.  

Il est possible de définir des kernels mais il vaut mieux utiliser PyCuda ou C/C++ directement pour ça.

CuPy possède beaucoup de fonctions qui ne sont pas faites pour êtres utilisées directement par un utilisateur mais plutôt pour être utilisées par d'autres fonctions CuPy ou par d'autres modules  

Un des objectifs principals de CuPy pour un utilisateur est de remplacer NumPy et Scipy dans un programme sans avoir besoin de modifier le code (sauf les appels au module)

Documentation de CuPy : https://docs.cupy.dev/en/stable/overview.html

#### PyCuda
        
PyCuda est la bibliothèque à utiliser pour définir et exécuter des kernels CUDA directement en Python.  
        
Pour utiliser PyCuda, plusieurs étapes sont nécéssaires. Détaillons ces dernières via un exemple de __multiplication de matrices__ :

1.  Écrire le kernel en CUDA (fonction(s) ou programme à exécuter sur GPU)

    [Voir explication d'un kernel ici](#utilisation-de-kernels)
    ```c
    kernel_code = """
    __global__ void matmul(float *A, float *B, float *C, int N) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        float sum = 0.0;

        if (row < N && col < N) {
            for (int k = 0; k < N; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    """
    ```

2.  Créer les variables en python
    ```py
    import numpy as np
    import pycuda.autoinit
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    N = 4  # La taille de la matrice (ici des matrices de tailles N*N)
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)
    C = np.zeros((N, N), dtype=np.float32)
    ```

3. Initialiser les variables en GPU (allouer la mémoire)
    ```py
    A_gpu = cuda.mem_alloc(A.nbytes)
    B_gpu = cuda.mem_alloc(B.nbytes)
    C_gpu = cuda.mem_alloc(C.nbytes)
    ```

4. Copier les données vers le gpu 
    ```py
    cuda.memcpy_htod(A_gpu, A)
    cuda.memcpy_htod(B_gpu, B)
    ```

5. Récupérer la/ les fonction(s) voulue(s) du kernel
    ```py
    # On compile notre kernel et on récupère la fonction du kernel pour pour l'exec
    mod = SourceModule(kernel_code)
    matmul = mod.get_function("matmul")
    ```

6. Calculer le nombre de thread, la taille des blocs et la taille du grid
    ```py
    # on définit la taille des blocs (16x16 threads pour un bloc)
    block_size = (16, 16, 1)

    # On calcul la taille de la grille
    #on a : ⌈ taille_matrice / taille_bloc⌉ (arrondi supérieur)
    grid_size = (
        int(np.ceil(N / block_size[0])), 
        int(np.ceil(N / block_size[1])), 
        1 # 1 car c'est une matrice 2D, donc pas de Z
    )
    ```

7. Éxecuter la/les fonction
    ```py
    # On exec le kernel
    matmul(A_gpu, B_gpu, C_gpu, np.int32(N), block=block_size, grid=grid_size)
    ```

8. Si on en a besoin récupérer le(s) sortie(s) en les transférant sur le CPU
    ```py
    cuda.memcpy_dtoh(C, C_gpu)
    print("Résultat de la multiplication de matrices :")
    print(C)
    ```

La libération de la mémoire est gérée automatiquement par PyCuda donc il n'y a pas besoin de s'en occuper

PyCuda permet aussi de définir des GPU arrays mais, il vaut mieux utiliser CuPy sauf si on a besoin de plus de contrôle sur le GPU, la mémoire, ...
Il permet aussi de faire de la métaprogrammation (programme qui écivent des programmes)
        
Documentation : https://documen.tician.de/pycuda/driver.html#profiler-control        

### LLM

#### PyTorch
PyTroch est la bibliothèque que nous avons utilisé pour tester les llm mais elle permet de farie beaucoup d'autres choses :
| Possibilitées / Ce qui est possible |
|:--------------|
| Calcul différentiable |
| Optimisation numérique |
| Traitement de signal (image/audio) |
| Machine Learning et Deep Learning |
| Modèles séquentiels/séries temporelles |
| Modèles probabilistes |
| Calcul scientifique GPU généralisé (en tensoriel) |

Dans notre cas, nous avons utilisé la partie de PyTorch dédié au Machine Learning et Deep learning pour utiliser des llm.



__TODO__ : détailler comment utiliser PyTorch dans notre cas
Modèles pour coder : https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard



Guide : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
Documentation : https://pytorch.org/docs/stable/index.html

#### TensorFlow

__TODO__ 

<h2 id="utilisation-gpu-en-c-c++">Utilisation GPU en C/C++</h2>

### Utilisation de kernels
    
__TODO__ explication d'un kernel et aventages

#### Exemple de programmation cuda (bout de code):

```c
// Affichage d'un array
void printArray(int* arr, int size) 
{
    for (int i = 0; i < size; ++i)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}
```

```c
// Allocation de la mémoire sur GPU
int* gpuArrmerge;
int* gpuTemp;

cudaMalloc((void**)&gpuArrmerge, size * sizeof(int));
cudaMalloc((void**)&gpuTemp, size * sizeof(int));
```




## RAPIDS