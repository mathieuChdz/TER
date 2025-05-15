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
Pour les llm nous avons également eu besoin de la bibliothèque transformers.

Voici les étapes pour utiliser un llm :

1. Choisir un modèle pré entrainé (nous sommes allés sur https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard). Les modèles ne peuvent en général pas tout tout faire (code completion, infilling, instructions/chat) il faut donc bien choisir le bon llm en fonction des besoin.  
La taille des modèles est également importante, elle est exprimée en milliards de paramètres (7b : 7 milliards, 70b : 70 milliards), plus il y a de paramètre plus le modèle devrait être performant (meilleures réponses) au prix de d'une plus grande taille sur disque, d'une consommation accrue des ressources matérielles et d'un temps de réponse plus long (en fonction de la performance de la machine sur laquelle il a été lancé).


2. Charger le tokenizer : transforme le texte brut en tokens (unités compréhensible pour le modèle)

    ```py
    tokenizer = AutoTokenizer.from_pretrained(
        "gpt2",                      # id modèle
        model_max_length=1024,       # longueur max des tokens (par défaut : None)
        use_fast=True                # tokenizer rapide basé sur Rust (par défaut (Python) : False)
    )
    ```

3. Charger le modèle

    ```py
    model = AutoModelForCausalLM.from_pretrained(
        "gpt2",                       # id modèle
        torch_dtype=torch.float16,    # précision (torch.float32 : standard, torch.float16 : mixte, auto : optimal)
        device_map="auto",            # répartition automatique sur les GPU ("cpu" pour utiliser le cpu)
        low_cpu_mem_usage=True,       # réduction de l'utilisation de la mémoire CPU (désactiver : false)
        revision="main",              # pour utiliser une version spécifique (par exemple : "v1.0")
        trust_remote_code=False,      # autorise l'exécution de code distant (pour modèles personalisés)
        offload_folder=None           # répertoire de déchargement de parties du modèle sur disque
    )
    ```

4. Préparer les données d'entrée

    ```py
    inputs = tokenizer(
        "Bonjour, comment vas-tu ?",  # texte d'entrée / input ou code à compléter ou code avec partie manquante à compléter (peu nécésiter une balise en fonction du modèle, exemple : "<mask>" à l'endroit de la partie manquante)
        padding=True,                 # tool : Ajoute du padding pour toujours avoir la même taille
        truncation=True,              # Tronque si la séquence est trop longue
        max_length=128,               # Longueur max des séquences (par défaut : None)
        add_special_tokens=True,      # Ajoute les tokens spéciaux nécessaires au modèle
        stride=50,                    # Chevauchement entre les morceaux de séquences (utile pour les textes longs)
        return_tensors="pt"           # On spécifie les tenseurs que l'on retourne ("pt" pour PyTorch, "tf" pour TensorFlow, "np" pour NumPy)
    ).to("cuda")                      # Transfère les données sur le GPU
    ```

5. Générer une réponse

    ```py
    output = model.generate(
        inputs["input_ids"],          # ids tokens d'entrée
        max_new_tokens=50,            # limite la longueur de la réponse générée
        temperature=0.7,              # créativité (valeurs plus élevées = plus de diversité, 1.0 = neutre)
        top_k=50,                     # limite aux 50 tokens les plus probables (None pour désactiver)
        top_p=0.9,                    # limite aux tokens dont la probabilité cumulée est < 0.9 (1.0 pour désactiver)
        do_sample=True,               # échantillonnage aléatoire
        num_return_sequences=1,       # nb réponses retournées
        repetition_penalty=1.2,       # pénalise les répétitions (1.0 pour désactiver)
        length_penalty=1.0,           # pénalise ou favorise les réponses longues (1.0 = neutre, >1 favorise les longues, <1 favorise les courtes)
        early_stopping=True           # arrêter ou non la génération dès qu'un token de fin est généré
    )
    ```

6. Décoder la réponse

    ```py
    response = tokenizer.decode(
        output[0],                    # séquence de tokens générée
        skip_special_tokens=True,     # ignorer ou non les tokens spéciaux
        clean_up_tokenization_spaces=True  # clean ou non les espaces inutiles
    )
    ```

Tous les arguments pour ces fonction ne sont pas obligatoire, on peut utiliser l'option par défaut et donc ne pas mettre ceux dont on a pas besoin.

Voir un exemple fonctionnel avec des options réduites : [Cliquez ici](../../llm/llm.py)

Guide : https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
Documentation : https://pytorch.org/docs/stable/index.html

Il devrait être possible d'utilsier une extension vscode (par exemple continue) avec une api pour les llm installés sur le serveur pour pouvoir les utiliser directement dans vscode mais nous ne l'avons pas testé car nous avons recherché d'autres choses et que ce n'était pas notre but principal.

#### TensorFlow

__TODO__ 

<h2 id="utilisation-gpu-en-c-c++">Utilisation GPU en C/C++</h2>

### Utilisation de kernels

Un kernel est une fonction qui s'exécute sur le GPU et non sur le CPU. Ces fonctions sont écrites en C/C++ et utilisent CUDA, une plateforme de calcul parallèle.

Cela permet de paralléliser, à volonté, certaines opérations et ainsi d’accélérer l'exécution d'un programme. Pour créer un kernel, il faut utiliser le mot-clé *`__global__`*. Lors de l'exécution, le CPU est capable de déterminer ce qu'il doit envoyer au GPU.

À noter : il est nécessaire de gérer manuellement l'utilisation de la mémoire. De plus, une bonne gestion des threads, des blocs et de la grille GPU est indispensable.

(Un thread est une unité de base, représentant par exemple un élément d’un tableau ; un bloc est un groupe de threads ; une grille est un ensemble de blocs.)



#### Avantages des kernels :

- Parallèlisme très élevé
- Réduction du temps de calcul (via le parallélisme)
- Peut être avantageux niveau énergetique (consommation watts)
- Très éfficace lorque c'est bien utilisé


##### Exemple de programmation cuda (bout de codes):

```c
// Kernel d'une addition de deux tableaux
__global__ void add(int* a, int* b, int* c, int size) 
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) 
    {
        c[index] = a[index] + b[index];
    }
}
```
En **Python**, pour utiliser un kernel, on a recours à des fonctions toutes faites, comme par exemple `cupy.matmul()`.

En revanche, en **C/C++**, on peut écrire directement une fonction pour le kernel CUDA qui s’exécute sur le GPU, ce qui permet d’effectuer des optimisations précises et d’avoir un contrôle total.

> On peut également utiliser PyCUDA pour envoyer du code C/C++ (sous forme de chaîne de caractères) au compilateur `nvcc`, puis l’appeler depuis Python, si l’on souhaite écrire une fonction kernel en C/C++ tout en travaillant en Python.

Le **GPU** exécute des milliers de threads en parallèle.

En **Python**, les bibliothèques comme Cupy gèrent automatiquement les threads : un simple `cupy.add(a, b)` suffit. Cupy n’est ici qu’un exemple parmi d’autres bibliothèques capables d’exploiter les GPU.

En **C/C++**, c’est au développeur de décider combien de threads lancer et comment les organiser.

En **Python**, des bibliothèques comme Cupy gèrent directement l’exécution sur le GPU.

En **C/C++**, il faut gérer manuellement plusieurs étapes :

1. **Allouer de la mémoire avec `cudaMalloc()`**  
   Cette étape est nécessaire pour stocker les données que le kernel CUDA devra traiter.

   Exemple de syntaxe :

```c
   cudaError_t cudaMalloc(void** devPtr, size_t size);
```
- `devPtr` : pointeur vers la variable qui stockera l’adresse sur le GPU  
- `size` : taille en octets à allouer

2. **Copier les données du CPU vers le GPU  avec `cudaMemcpy()`**
(nécessaire pour envoyer les entrées au GPU et récupérer les résultats)

Ensuite, il faut libérer la mémoire avec `cudaFree()`.

##### Exemple : Comparaison entre Python et C/C++ : 
En Python :

```python
import cupy as cp

a = cp.array([1, 2, 3], dtype=cp.float32)
b = cp.array([4, 5, 6], dtype=cp.float32)
c = a + b  # CuPy gère tout automatiquement
print(c)  # Affiche le résultat sur le GPU
```
En C/C++ :
```c
#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA
__global__ void addVectors(float *a, float *b, float *c, int n) {
    int i = threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 3;
    float h_a[n] = {1, 2, 3};  // Données CPU
    float h_b[n] = {4, 5, 6};
    float h_c[n];

    // Alloue la mémoire GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // Copie CPU vers GPU
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Lance le kernel (1 bloc, n threads)
    addVectors<<<1, n>>>(d_a, d_b, d_c, n);

    // Copie GPU vers CPU
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Affiche le résultat
    for (int i = 0; i < n; i++) std::cout << h_c[i] << " ";

    // Nettoie la mémoire GPU
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}
```
Pour de l’optimisation manuelle et l’écriture de kernels spécialisés pour des cas précis, l’utilisation du C/C++ est recommandée.

### Bibliothèque C/C++ intéressantes :  

**Thrust** est une bibliothèque qui fournit des algorithmes parallèles optimisés pour GPU. Elle ressemble à la bibliothèque STL (Standard Template Library) en C++.

La STL contient essentiellement des implémentations de structures de données et d’algorithmes courants, tels que des listes, piles, tableaux, tris, recherches, etc...

Thrust permet d’utiliser ces algorithmes sur le GPU, en utilisant la syntaxe de la STL. Il est donc possible d’écrire du code C++ qui s’exécute sur le GPU sans avoir à se soucier des détails de l’architecture CUDA.

##### Exemple d’un tri tableau sur GPU :
```c
#include <thrust/sort.h>

// Données sur le GPU (déjà alloées avec cudaMalloc)
thrust::device_ptr<float> d_data = ...; 

// Tri sur GPU (équivalent de std::sort, mais sur GPU)
thrust::sort(thrust::device, d_data, d_data + N);
```
`thrust::device` indique que l’opération est effectuée sur le GPU, et `d_data` pointe vers les données présentes sur le GPU.

#### Quelques exemples algorithmes disponibles dans Thrust :

| Algorithme         | Exemple (GPU avec Thrust)                | Équivalent sur CPU (STL)   |
|--------------------|------------------------------------------|-----------------------------|
| Tri                | `thrust::sort(device, début, fin)`       | `std::sort`                 |
| Somme / Réduction  | `thrust::reduce(device, début, fin)`     | `std::accumulate`           |
| Recherche          | `thrust::find(device, début, fin)`       | `std::find`                 |
| Copie              | `thrust::copy(device, src, dest)`        | `std::copy`                 |

Il y a aussi quelques exemples dans `/TER/GPU_en_C/Thrust` où l’on peut voir des implémentations avec et sans Thrust.

Pour éviter la complexité d’écriture des kernels, cette bibliothèque est très utile pour les débutants en CUDA.

Un exemple pour compiler : 
```bash
nvcc  calcul_nombre_aleatoire_en_parallele.cu -o calcul_nombre_aleatoire_en_parallele
```
puis exécuter le programme avec : 
```bash
./calcul_nombre_aleatoire_en_parallele
```
#### Autres bibliothèques C/C++ intéressantes : cuBLAS et cuFFT : Algèbre Lineaire & FFT sur GPU.

### cuBLAS
🔗 https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/simpleCUBLAS

Les opérations mathématiques de type BLAS (somme scalaire, produit matriciel, etc.) nécessitent d'importantes ressources en calcul et en mémoire.

L'utilisation de cuBLAS permet d'exécuter ces opérations d'algèbre linéaire de manière hautement optimisée sur le GPU, ce qui améliore considérablement les performances, notamment pour les calculs intensifs.

#### Niveaux de BLAS pris en charge par cuBLAS

La bibliothèque cuBLAS couvre l’ensemble des trois niveaux du standard BLAS, chacun correspondant à un type d’opération avec une complexité différente :

| **Niveau** | **Type d’opération**                         | **Exemples de fonctions cuBLAS**       |
|------------|-----------------------------------------------|----------------------------------------|
| **1**      | Opérations vectorielles (**O(N)**)           | `cublasSaxpy`, `cublasDdot`            |
| **2**      | Multiplications matrice–vecteur (**O(N²)**)  | `cublasSgemv`, `cublasDsymv`           |
| **3**      | Multiplications matrice–matrice (**O(N³)**)  | `cublasSgemm`, `cublasDgemm`           |

####  cuFFT : Transformée de Fourier sur GPU.

🔗 https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries/simpleCUFFT

La FFT(Transformée de Fourier rapide) est utilisée pour l’analyse fréquentielle de signaux (audio, images, etc.).  
Elle est donc idéale pour le traitement du signal en temps réel, en raison de son efficacité algorithmique.
La bibliothèque cuFFT est optimisée et permettant ainsi d’accélérer les calculs FFT par rapport aux implémentations CPU.

##### Fonctions principales de cuFFT
 
Le tableau ci-dessous présente les principales fonctions utilisées pour initialiser un plan de calcul et exécuter les FFT entre données réelles et complexes.

| **Fonction**           | **Opération**                                  |
|------------------------|------------------------------------------------|
| `cufftPlan1d/2d/3d`    | Initialise un plan FFT (1D, 2D ou 3D)          |
| `cufftExecR2C`         | Transformée de Fourier : Réel → Complexe       |
| `cufftExecC2R`         | Transformée de Fourier : Complexe → Réel       |



##### Tests comparatifs : GPU vs CPU

Nous réalisons deux tests pour comparer les performances entre **GPU** et **CPU** :

- cuBLAS (Multiplication matricielle) : comparaison des temps d’exécution d’un produit matriciel sur le GPU (via cuBLAS) et sur le CPU (via BLAS ou `numpy.dot` par exemple).

- FFT (Transformée de Fourier rapide) : comparaison entre l’exécution de la FFT sur le GPU (via cuFFT ou CuPy) et sur le CPU (via NumPy ou SciPy).

Ces tests permettent de mettre en évidence les gains de performance apportés par l’accélération GPU dans des opérations intensives en calcul.

Résultat : 
```c
Benchmark Multiplication Matricielle (GEMM) 
Matrices 1024x1024

CPU GEMM: 120.543 ms
GPU GEMM: 2.189 ms
```
```c
Benchmark FFT 
Taille du signal: 1048576 points

CPU FFT: 25.672 ms
GPU FFT: 0.412 ms
```

### Autres bibliothèques utiles pour CUDA en C/C++
🔗https://github.com/NVIDIA/cuda-samples/tree/master/Samples/4_CUDA_Libraries

- **cuRAND**  
  Générateur de nombres aléatoires haute performance sur GPU (uniformes, normaux, log-normaux, etc.).

- **cuSPARSE**  
  Fournit des algorithmes optimisés pour les matrices creuses (sparse), utiles dans les grands systèmes linéaires.

- **NPP (NVIDIA Performance Primitives)**  
  Ensemble de fonctions optimisées pour le traitement d’images et de signaux (filtrage, convolution, transformation de couleurs...)

- **CUB**  
  Bas niveau mais très rapide : primitives parallèles (scan, reduce, radix sort) très efficaces et personnalisables.

- **nvGRAPH**  
  Fournit des algorithmes pour le traitement de graphes (BFS, PageRank, etc.) sur GPU.

- **TensorRT**  
  Pour l'inférence de réseaux de neurones profonds avec une optimisation sur GPU.

- **NCCL (NVIDIA Collective Communications Library)**  
  Utilisée pour la communication rapide entre plusieurs GPU (par ex. pour le Deep Learning distribué).


### 🔹 Exemples notables dans `CUDA_Features`

🔗https://github.com/NVIDIA/cuda-samples/tree/master/Samples/3_CUDA_Features

- `simpleCudaGraphs` 
  Démontre l'utilisation des CUDA Graphs, une fonctionnalité permettant de capturer et de réutiliser des séquences d'opérations GPU. Cela réduit la surcharge du CPU et améliore les performances globales.

- `cdpSimpleQuicksort` 
  Implémente un tri rapide (quicksort) en utilisant le CUDA Dynamic Parallelism, où les kernels peuvent lancer d'autres kernels. Cela permet une parallélisation récursive efficace.

- `cudaTensorCoreGemm`  
  Montre comment exploiter les Tensor Cores pour effectuer des multiplications de matrices à haute performance, en utilisant l’API WMMA (Warp Matrix Multiply Accumulate).

- `graphMemoryFootprint`  
  Explore la gestion de la mémoire dans les CUDA Graphs, et comment minimiser l’empreinte mémoire lors de l'exécution de graphes complexes.




### Multi-GPU avec CUDA

🔗https://github.com/NVIDIA/cuda-samples/tree/master/Samples/0_Introduction/simpleMultiGPU

Lorsque l’on dispose de plusieurs GPU (comme dans notre cas, avec 2 GPU H100), cette méthodologie de programmation peut être très avantageuse. Elle permet de répartir les calculs sur plusieurs cartes en parallèle, ce qui est particulièrement utile pour :

- accélérer l'entraînement de grands modèles (comme les LLM),
- traiter de gros jeux de données,
- réduire le temps de calcul global,
- gérer des modèles trop volumineux pour un seul GPU,
- et optimiser l’utilisation des ressources disponibles.

Exemple de mise en œuvre :  
`/TER/GPU_en_C/Multi_gpu/multi_gpu.cu`



## RAPIDS

C'est quoi rapids ?

RAPIDS est un ensemble de bibliothèques (open-source) développé par NVIDIA, pour accélérer les executions via GPU. Rapids se base et utilise NVIDIA CUDA ainsi que Apache Arrow pour l'accélération GPU. Plus précisément, retrouve :

## RAPIDS : Utilisation de CUDA et Apache Arrow

| **Technologie** | **Description** |
|------------------|------------------|
| **CUDA** | Plateforme de calcul parallèle développée par NVIDIA. Permet d'exploiter la puissance des GPU pour effectuer des calculs massivement parallèles (traitement de données, apprentissage machine, ...) |
| **Apache Arrow** | Framework open-source conçu pour optimiser le traitement et le partage de données en mémoire (format de données très performant). Le format est un tableau en colonnes --> réduit les coûts de sérialisation et de désérialisation.|

Cette combinaison permet d'accélérer les workflows de science des données (accéléreration des processus de traitement des données).

Il est conçu pour fonctionner de manière similaire aux outils populaires comme Pandas ou Scikit-learn, mais en exploitant la puissance des GPU au maximum pour de meilleurs performances.

### Graphs

#### cuGraph

cuGraph est une bibliothèque de graphes disponible dans RAPIDS. Elle permet d'exécuter des algorithmes de graphes sur GPU, ce qui améliore considérablement les performances par rapport aux implémentations CPU.

cuGraph est disponible en tant que backend de NetworksX (la bibliothèque de graphes Python populaire) en utilisant nx-cugraph. Il suffit de mettre la variable d'environnement NX_CUGRAPH_AUTOCONFIG à True pour utiliser cuGraph comme backend sans changer le code écrit avec NetworkX.

La liste des algorithmes disponibles dans cuGraph (assez longue) est disponible [ici](https://docs.rapids.ai/api/cugraph/stable/nx_cugraph/supported-algorithms/).

Il n'y a rien de particulièrement compliqué pour lancer ces algorithmes car tout est déjà implémenté. Il faut juste configurer les paramètres d'entrée (graph et autres) correctement.   
Des exemples de comment exécuter les algorithmes sont disponibles sur le github de cuGraph [ici](https://github.com/rapidsai/cugraph/tree/main/notebooks/algorithms).


### Machine Learning

#### cuML

cuML est une bibliothèque de machine learning disponible dans RAPIDS. On l'utilise ici pour exploiter la puissance des GPU afin d'améliorer les performances des training de modèles de machine learning. Elle s'utilise de la même facon que scikit-learn.

Avec cuML, il est possible d'exécuter rapidement des algorithmes tels que :
- Régression linéaire et logistique
- Clustering (KMEANS)
- Classification
- Autres

L'avantage de cuML est que l'entrainement des modèle peut aller jusqu'à 50 fois plus vite. Cela permet d'avoir plus de temps pour optimiser son modèle sans avoir un problème de temps d'attente entre chaque entrainement.

[Pour en savoir plus sur cuML](https://rapids.ai/cuml-accel/)

Pour tester et comparer les performances CPU vs GPU vs GPU avec cuML, des modèles de machines learning étudiés et travaillés en 2ème et 3ème année de BUT informatique ont été sélectionnés. On y retrouve :
- Régression Linéaire
- KMEANS
- Descente de gradient

Voir document de test de performance (temps d'execution) sur différents modèles de machine learning [Cliquez ici](../machine_learning_CPU_vs_GPU.md)