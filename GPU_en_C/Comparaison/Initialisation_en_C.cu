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