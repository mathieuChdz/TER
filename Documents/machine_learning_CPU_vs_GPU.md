# Comparaison des performances CPU vs GPU en Machine Learning

Dans ce rapport, nous analysons les performances des CPU et GPU pour l'exécution d'algorithmes de machine learning. Le but est d'identifier les gains en temps d'exécution offerts par l'utilisation d'un GPU par rapport à un CPU traditionnel.

## Contexte de l'expérimentation
  - **CPU** : AMD EPYC™ 9354 *4ème génération*
  - **GPU** : NVIDIA H100 NVL (x2)

## Résultats : temps d'exécution moyen

À noter :
- Le temps d'exécution moyen calculé est arrondi à la 5e décimale près
- __10 exécutions__ d'un même N sample pour constituer le temps d'exécution moyen

### 1 - KMEANS

| Paramètres / Présets fixes | Valeur |
|:----------|:----------:|
| n_features | 2 |
| centers | 3 |
| cluster_std | 4.0 |
| random_state | 42 |

| nb samples | Mean exec time (s) (CPU) | Mean exec time (s) (GPU) | Amélioration GPU (en %) | Mean exec time (cuDF) | Amélioration cuDF
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| 500 000 | 0.69098 | 0.07606 | -88.99 % | NEED TO DO |  |
| 1 000 000 | 1.28983 | 0.05432 | -95.79 % | NEED TO DO |  |
| 1 500 000 | 1.96068 | 0.06764 | -96.55 % | NEED TO DO |  |
| 2 000 000 | 2.37538 | 0.09601 | -95.96 % | NEED TO DO |  |
| 2 500 000 | 3.14407 | 0.12450 | -96.04 % | NEED TO DO |  |
| 3 000 000 | 3.84097 | 0.13251 | -96.55 % | NEED TO DO |  |
| 3 500 000 | 4.72982 | 0.20585 | -95.65 % | NEED TO DO |  |
| 4 000 000 | 5.43119 | 0.20793 | -96.17 % | NEED TO DO |  |
| 4 500 000 | 6.64960 | 0.23208 | -96.51 % | NEED TO DO |  |
| 5 000 000 | 7.75914 | 0.23860 | -96.92 % | NEED TO DO |  |


### 2 - REGRESSION LINEAIRE

| Paramètres / Présets fixes | Valeur |
|:----------|:----------:|
| n_features | 1 |
| noise | 30 |
| bias | 30 |
| random_state | 42 |
| Taille train set | 80 % |
| Taille test set | 20 % |

| nb samples | Temps moyen d'execution (s) (CPU) | Temps moyen d'execution (s) (GPU) | Amélioration GPU |
|:------------:|:------------:|:------------:|:------------:|
| 500 000 | NEED TO DO | NEED TO DO |  |
| 1 000 000 | NEED TO DO | NEED TO DO |  |
| 1 500 000 | NEED TO DO | NEED TO DO |  |
| 2 000 000 | NEED TO DO | NEED TO DO |  |
| 2 500 000 | NEED TO DO | NEED TO DO |  |
| 3 000 000 | NEED TO DO | NEED TO DO |  |
| 3 500 000 | NEED TO DO | NEED TO DO |  |
| 4 000 000 | NEED TO DO | NEED TO DO |  |
| 4 500 000 | NEED TO DO | NEED TO DO |  |


### 2 - DESCENTE DE GRADIANT

| Paramètres / Présets fixes | Valeur |
|:----------|:----------:|
| lambda | 0.01 |
| epsilon | 0.01 |

Génération du point : tuple (x, y)
| Bornes | Interface |
|:----------|:----------|
| Borne inférieur | [-0.5, -0.5] |
| Borne supérieur | [1.5, 1.5] |

| nb samples | Temps moyen d'execution (s) (CPU) | Temps moyen d'execution (s) (GPU) | Amélioration GPU |
|:------------:|:------------:|:------------:|:------------:|
| 500 000 | NEED TO DO | NEED TO DO |  |
| 1 000 000 | NEED TO DO | NEED TO DO |  |
| 1 500 000 | NEED TO DO | NEED TO DO |  |
| 2 000 000 | NEED TO DO | NEED TO DO |  |
| 2 500 000 | NEED TO DO | NEED TO DO |  |
| 3 000 000 | NEED TO DO | NEED TO DO |  |
| 3 500 000 | NEED TO DO | NEED TO DO |  |
| 4 000 000 | NEED TO DO | NEED TO DO |  |
| 4 500 000 | NEED TO DO | NEED TO DO |  |


## Annexes

- Scripts d’entraînement
- Configurations exactes
- Courbes de performances (optionnel : à inclure en image si tu veux)

---

