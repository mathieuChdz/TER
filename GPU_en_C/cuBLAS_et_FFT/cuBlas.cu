#include <iostream>
#include <vector>
#include <chrono>
#include <cublas_v2.h>
#include <cblas.h>

#define TIMER_START auto start = std::chrono::high_resolution_clock::now();
#define TIMER_END(msg) \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> elapsed = end - start; \
    std::cout << msg << elapsed.count() * 1000 << " ms\n";

const int N = 1024;  // Matrices NxN

void run_gemm_test() {
    // Allocation mémoire
    std::vector<float> h_A(N*N), h_B(N*N), h_C_cpu(N*N), h_C_gpu(N*N);
    
    // Initialisation aléatoire
    for (int i = 0; i < N*N; ++i) {
        h_A[i] = rand()/(float)RAND_MAX;
        h_B[i] = rand()/(float)RAND_MAX;
    }

    // Version CPU (OpenBLAS)
    TIMER_START
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0, h_A.data(), N, h_B.data(), N, 0.0, h_C_cpu.data(), N);
    TIMER_END("CPU GEMM: ")


    // Version GPU (cuBLAS)
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N*N*sizeof(float));
    cudaMalloc(&d_B, N*N*sizeof(float));
    cudaMalloc(&d_C, N*N*sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    // Copie CPU vers GPU
    cublasSetMatrix(N, N, sizeof(float), h_A.data(), N, d_A, N);
    cublasSetMatrix(N, N, sizeof(float), h_B.data(), N, d_B, N);

    float alpha = 1.0f, beta = 0.0f;
    TIMER_START
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    cudaDeviceSynchronize();
    TIMER_END("GPU GEMM: ")

    // Vérification
    cublasGetMatrix(N, N, sizeof(float), d_C, N, h_C_gpu.data(), N);
    float max_err = 0;
    for (int i = 0; i < N*N; ++i) {
        max_err = std::max(max_err, abs(h_C_cpu[i]-h_C_gpu[i]));
    }
    std::cout << "Erreur maximale: " << max_err << "\n";

    // Nettoyage
    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    std::cout << "Benchmark Multiplication Matricielle (GEMM) \n";
    std::cout << "Matrices " << N << "x" << N << "\n\n";
    run_gemm_test();
    return 0;
}