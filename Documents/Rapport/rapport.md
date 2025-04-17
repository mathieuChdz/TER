# GPU et environnement de développement

- [Langages](#langages)
- [Section 2](#Mettre-le-titre-ici)
- [Section 3](#Mettre-le-titre-ici)
- [Section 4](#Mettre-le-titre-ici)

---


## Langages :  
- Python avec bibliothèques, peuvent faire des choses très différentes, avec plus ou moins de contrôle direct sur le gpu  
- C/C++ avec extensions CUDA (pas encore recherché) --> plus compliqué mais plus de contrôle sur le gpu, contrôle plus précis  

Nécessaire pour coder en Python :  
driver Nvidia, CUDA Toolkit, Python avec des versions compatibles et les bibliothèques à utiliser également avec une version compatible avec le reste  

Nécessaire pour coder en C/C++ :  
driver Nvidia, CUDA Toolkit, Compiler C/C++ avec des versions compatibles  

---
- Utilisation GPU en python
Modules python :  
    - CuPy :  
        Quasiment toutes les fonctions de NumPy sont aussi sur CuPy : manipulation de matrices, de polynômes, algèbre linéaire, transformation rapide de fourrier, génération de nombres aléatoires, tests, opérations binaires, fonctions mathématiques, ...  
        Plusieurs sous-modules de scipy sont également disponibles  
        Propose également de gérer la mémoire GPU même si le degré de gestion n'est pas aussi avancé que d'autres bibliothèques comme PyCuda ou le C/C++ directement  
        Il est possible de définir des kernels mais il vaut mieux utiliser PyCuda ou C/C++ directement pour ça  

        CuPy possède beaucoup de fonctions qui ne sont pas faites pour êtres utilisées directement par un utilisateur mais plutôt pour être utilisées par d'autres fonctions CuPy ou par d'autres modules  

        L'objectif principal de CuPy pour un utilisateur est de remplacer NumPy et Scipy dans un programme sans avoir besoin de modifer le code (sauf les appels au module)

        Documentation de CuPy : https://docs.cupy.dev/en/stable/overview.html

    - PyCuda :
        Documentation ? : https://documen.tician.de/pycuda/driver.html#profiler-control        

- Utilisation GPU en C/C++ :
    
    - Kernels :
        - exemple :
        ```cuda
        //Function to print array
        void printArray(int* arr, int size) 
        {
            for (int i = 0; i < size; ++i)
                std::cout << arr[i] << " ";
            std::cout << std::endl;
        }
        ```