#include <iostream>
#include <cuda_runtime.h>

// Kernel CUDA (addition manuelle)
__global__ void addVectors(float *a, float *b, float *c, int n) {
    int i = threadIdx.x;  // Chaque thread traite un élément
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    int n = 3;
    float h_a[n] = {1.0, 2.0, 3.0};  // CPU
    float h_b[n] = {4.0, 5.0, 6.0};
    float h_c[n];

    // 1. Allocation GPU
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));

    // 2. Copie CPU vers GPU
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Lancement du kernel (1 bloc, 3 threads)
    addVectors<<<1, n>>>(d_a, d_b, d_c, n);

    // 4. Copie GPU vers CPU
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // 5. Affichage
    for (int i = 0; i < n; i++) std::cout << h_c[i] << " ";  // Affiche "5 7 9"

    // 6. Nettoyage
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}