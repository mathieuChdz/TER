#include <cuda_runtime.h>
#include <stdio.h>

// Somme d’un Tableau sur 2 GPU

// Kernel pour sommer une partie du tableau
__global__ void sumArray(float *array, float *result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) atomicAdd(result, array[idx]); // Somme atomique
}

int main() {
    int N = 1 << 24; // 16 millions d'éléments
    float *h_array = new float[N]; // CPU
    for (int i = 0; i < N; i++) h_array[i] = 1.0f; // Remplit avec des 1

    // 1. Initialise 2 GPU
    int numGPUs = 2;
    cudaSetDevice(0); // Active GPU 0
    float *d_array0, *d_result0;
    cudaMalloc(&d_array0, N/2 * sizeof(float));
    cudaMalloc(&d_result0, sizeof(float));

    cudaSetDevice(1); // Active GPU 1
    float *d_array1, *d_result1;
    cudaMalloc(&d_array1, N/2 * sizeof(float));
    cudaMalloc(&d_result1, sizeof(float));

    float zero = 0.0f;

    cudaSetDevice(0);
    cudaMemcpy(d_result0, &zero, sizeof(float), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMemcpy(d_result1, &zero, sizeof(float), cudaMemcpyHostToDevice);


    // 2. Copie les données (moitié sur chaque GPU)
    cudaSetDevice(0);
    cudaMemcpy(d_array0, h_array, N/2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaSetDevice(1);
    cudaMemcpy(d_array1, h_array + N/2, N/2 * sizeof(float), cudaMemcpyHostToDevice);

    // 3. Lance les kernels en parallèle
    dim3 block(256);
    dim3 grid((N/2 + block.x - 1) / block.x);

    cudaSetDevice(0);
    sumArray<<<grid, block>>>(d_array0, d_result0, N/2);

    cudaSetDevice(1);
    sumArray<<<grid, block>>>(d_array1, d_result1, N/2);

    // 4. Synchronise et récupère les résultats
    float sum0, sum1;
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaMemcpy(&sum0, d_result0, sizeof(float), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaMemcpy(&sum1, d_result1, sizeof(float), cudaMemcpyDeviceToHost);

    float totalSum = sum0 + sum1;
    printf("Somme totale: %f\n", totalSum); // Doit afficher N (16 millions)

    // 5. Nettoyage
    cudaFree(d_array0); cudaFree(d_result0);
    cudaFree(d_array1); cudaFree(d_result1);
    delete[] h_array;
}