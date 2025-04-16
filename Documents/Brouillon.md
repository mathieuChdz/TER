Langages :  
- Python avec bibliothèques, peuvent faire des choses très différentes, avec plus ou moins de contrôle direct sur le gpu  
- C/C++ avec extensions CUDA (pas encore recherché) --> plus compliqué mais plus de contrôle sur le gpu, contrôle plus précis  

Nécessaire pour coder en Python :  
driver Nvidia, CUDA Toolkit, Python avec des versions compatibles et les bibliothèques à utiliser également avec une version compatible avec le reste  

Nécessaire pour coder en C/C++ :  
driver Nvidia, CUDA Toolkit, Compiler C/C++ avec des versions compatibles  

Modules python :  
- CuPy :  
        Manipuler des matrices (même syntaxe que NumPy en changant numpy par cupy)