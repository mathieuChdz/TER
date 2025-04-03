# Programmation GPU en Python  

## CuPy :  
### Utilisation
Remplace NumPy pour le calcul GPU, avec une API très similaire.  

### Usage  
Calculs scientifiques et apprentissage automatique.  

### Documentation  
https://docs.cupy.dev/en/stable/user_guide/basic.html  
### Permet de
- manipuler des matrices pareille que NumPy mais en utilisant le GPU
- définir plusieurs types de kernel CUDA, définir un module (ensemble de fonction cuda, plusierus kernels)
- utiliser des stream et events
- transformation de Fourier rapide
- benchmarker des fonction et mesurer le temps d'éxecution sur GPU

## PyCuda :  
### Utilisation
Interface Python pour CUDA permettant d'écrire et d'exécuter directement du code CUDA. 

### Usage
Manipulation bas niveau des GPU via CUDA en Python.  

### Documentation
https://documen.tician.de/pycuda/  

### Permet d'exécuter des kernels en CUDA
On écrit un code en CUDA puis on peut l'exécuter dans le programme python. il faut aussi allouer la mémoire, transférer ce qu’on veut au kernel, récupérer la sortie, …
kernel = morceau de code qui va être exécuté plusieurs fois, en parallèle 

## PyTorch :  
### Utilisation
Deep learning et calcul matriciel optimisé sur GPU.  

### Usage
Machine learning et deep learning.  

### Site
https://pytorch.org/  

### Permet de :
- créer des modèles avec des réseaux de neurones
- de les entraîner en utilisant des ensembles de données
- de les tester pendant l'entraînement pour vérifier qu’ils s’améliorent
- de les sauvegarder et charger

## Numba :  
### Utilisation
Accélération du code Python via la compilation Just-In-Time (JIT) et utilisation de CUDA.  

### Usage
Accélération de boucles et fonctions lourdes.  

### Site
https://numba.pydata.org  

### Permet d'accélérer le code python
Particulièrement si il contient des array NumPy, fonctions et loops. Compile le code d'une fonction décorée en code machine juste avant la première exécution de celle-ci.

## RAPIDS GPU

--> suite de bibliothques open-source développée par **NVIDIA**.  
--> Fonctionne trs bien avec CUDA  

| Objectifs |
|----------------|
| Accélération du traitement des données |
| Accélération du machine learning |


### Avantages
RAPIDS permet d’exécuter des tâches massivement parallles sur GPU. Cela a pour effet d'accélérer le traitement de données et le training de modles (machines learning) par rapport aux CPU.

### Les 3 bibliothèques principales
1. **cuDF** : Manipulation de DataFrames (à la place de Panda sur CPU)
2. **cuML** : Basée sur scikit-learn
3. **cuGraph** : Manipoulation et traitement des graphes (à la place de NetworkX).

### Quelques exemples de ce qui est plus efficace et rapide via ces bibliothèques

|   bibliothèque    |   problèmes/algorithmes   |
|   ------------    |   ---------------------   |
|   cuDF   |    merge, groupBy, string UDF, numeric UDF, ...   |
|   cuML   |    KNN, Linear Regression, KMeans, Descente de gradient, ...   |
|   cuGraph   |    PageRank, Betweenness, ...    |