# Éléments de recherche

## Comment évaluer la puissance d'un LLM ?

Il y a trois facteurs principaux :

### 1. La puissance calculatoire  
Pour qu’un LLM fonctionne efficacement, il a besoin d’une bonne puissance de calcul.  
Les calculs sont effectués principalement sur GPU, car ils sont beaucoup plus efficaces que les CPU pour ce type de tâches.  
Les GPU permettent un calcul parallèle et distribué, ce qui accélère considérablement les performances.

**Objectif :**  
- Tester la puissance de GPU du serveur pour connaître sa puissance calculatoire (petaflop ?).  
- Comparer avec un calcul via CPU.  
- Tester la différence entre plusieurs LLM (gros modèles vs modèles plus petits).  

---

### 2. Le nombre de paramètres du modèle  
Plus un modèle possède de paramètres, plus il peut gérer des tâches complexes.

---

### 3. La taille du trainset  
Un dataset plus grand et plus diversifié améliore la capacité du modèle à généraliser et à fournir des réponses pertinentes.

---

## Les environnements disponibles pour le développement sur GPU

**Option principale : CUDA (Compute Unified Device Architecture)**  
- Développé par NVIDIA  
- Permet le calcul parallèle  

---

## Outils nécessaires

| Outil / Langage | Version | Commande | Installé |
|----------------|---------|----------|----------|
| Python        | 3.12.3  | `python3 --version` | OUI |
| CUDA          | 12.2    | `nvidia-smi` ou `nvcc --version` |  OUI |

---

## Programmation GPU en Python  

**CuPy** :  
- Une bibliothèque Python qui offre une syntaxe similaire à NumPy, mais optimisée pour les GPU.  
- L'installation via `pip` ne fonctionne pas sur le serveur, car la distribution est gérée par le système.  
- Les paquets ne sont pas gérés par `pip` mais par le gestionnaire de paquets de la distribution (probablement `apt`).  
- L'erreur d’installation rencontrée vise à éviter les conflits entre `pip` et `apt`.  

---

**Référence :**  
[Article sur les performances des LLM](https://www.followtribes.io/performances-llm-puissance-gpu-parametres-dataset/)
