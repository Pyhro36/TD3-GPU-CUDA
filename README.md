# TD3-GPU-CUDA

Introduction à la manipulation de CUDA pour GPU

Le fichier `cuda_td_1.cu` contient le code pour la Question 1.1, la fonction de noyau cuda est
`vecAdd(float *in1, float *in2, float *out, int len)`. Il est fonctionnel d'après les exemples donnés

Le fichier `cuda_td_2.cu` contient le code pour la Question 2.1, la fonction de noyau cuda est
`colorToGrayShadesKernel(float *in, float *out, int height, int width, int channels)`. Il est fonctionnel

Le fichier `cuda_td_3.cu` contient le code pour la Question 3.1, la fonction de noyau cuda est
`colorToGrayShadesKernel(float *in, float *out, int height, int width, int channels)`. Il n'est pas fonctionnel
pour le moment

Le fichier `cuda_td_4.cu` contient le code pour la Question 4.1, la fonction de noyau cuda est
`matrixMultiply(float *A, float *B, float *C, int numARows,
                                int numAColumns, int numBRows,
                                int numBColumns, int numCRows,
                                int numCColumns)`.
Il est fonctionnel d'après les exemples donnés.

Le fichier `cuda_td_5.cu` contient le code pour la Question 5.1, la fonction de noyau cuda est
`matrixMultiplyShared(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns)`.
Il est fonctionnel d'après les exemples donnés et un peu plus performant en moyenne que celui de la question 4.1,
`cuda_td_4.cu`.
                                
Le fichier `cuda_td_atomic.cu` contient une proposition de multiplication de matrice parallelisée avec des memoires
partagées a la mode des sacs de mots. Elle est fonctionnelle mais moins performante que `cuda_td_4.cu`. Sa fonction de
noyau cuda est
`matrixMultiplyShared(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns)`.
                                      
Le code peut être retrouvé sur le GitHub public : https://github.com/Pyhro36/TD3-GPU-CUDA.                                      

# TD4-GPU-CUDA

Le fichier `cuda_pattern_td_1.cu` contient le code pour la Question 1.1, la fonction de noyau cuda est
`histoKernel(unsigned int *input, unsigned int *bins, int inputLength)`. Il est fonctionnel d'après les exemples donnés

Le fichier `cuda_pattern_td_2_3d.cu` contient le code pour la Question 2.1 implémenté avec une grille et des blocs 3D,
la fonction de noyau cuda est
`stencil(float *output, float *input, int width, int height, int depth) `. Il n'est pas fonctionnel pour le moment.

Le fichier `cuda_pattern_td_3` contient le code pour la Question 3.1, la fonction de noyau cuda est
`total(float *input, float *output, int len)`. Il est fonctionnel d'après les exemples générés par le script
`generate_vector_float.sh`. 

Le code peut être retrouvé sur le GitHub public : https://github.com/Pyhro36/TD3-GPU-CUDA.