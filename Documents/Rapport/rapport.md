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

Un kernel est une fonction qui s'execute sur GPU et non pas sur CPU. Ces derniers sont écrits en C/C++ et utilise CUDA (plateforme de calcul parallèle). Cela nous permet de paralléliser quand on le souhaite des opérations et ainsi accélérer l'execution d'un programme.

Pour créer un kernel, il faut utiliser le mot-clé *`__global__`*. Lors de l'execution, le CPU sera capable de savoir ce qu'il faut envoyer au GPU.

À noter : Il est nécéssaire de gérer manuellement l'utilisation de la mémoire. De plus, il faut une bonne gestion des threads, des blocs et de la grille GPU.

#### Avantages des kernels :

- Parallèlisme très élevé
- Réduction du temps de calcul (via le parallélisme)
- Peut être avantageux niveau énergetique (consommation watts)
- Très éfficace lorque c'est bien utilisé

### Exemple de programmation cuda (bout de codes):

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