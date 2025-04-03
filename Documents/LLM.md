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

**Référence :**  
[Article sur les performances des LLM](https://www.followtribes.io/performances-llm-puissance-gpu-parametres-dataset/)
