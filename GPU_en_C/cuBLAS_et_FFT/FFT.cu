#include <iostream>
#include <vector>
#include <chrono>
#include <cufft.h>
#include <fftw3.h>

#define TIMER_START auto start = std::chrono::high_resolution_clock::now();
#define TIMER_END(msg) \
    auto end = std::chrono::high_resolution_clock::now(); \
    std::chrono::duration<double> elapsed = end - start; \
    std::cout << msg << elapsed.count() * 1000 << " ms\n";

const int FFT_SIZE = 1 << 20;  // 1 million de points

void run_fft_test() {
    // Allocation mémoire
    std::vector<float> h_signal(FFT_SIZE);
    std::vector<fftwf_complex> h_spectrum_cpu(FFT_SIZE/2 + 1);
    std::vector<cufftComplex> h_spectrum_gpu(FFT_SIZE/2 + 1);

    // Initialisation (sinusoïde)
    for (int i = 0; i < FFT_SIZE; ++i) {
        h_signal[i] = sin(2 * M_PI * i / 256);
    }
    // Version CPU (FFTW)
    fftwf_plan plan_cpu = fftwf_plan_dft_r2c_1d(
        FFT_SIZE, h_signal.data(), h_spectrum_cpu.data(), FFTW_ESTIMATE);

    TIMER_START
    fftwf_execute(plan_cpu);
    TIMER_END("CPU FFT: ")

    // Version GPU (cuFFT)
    cufftHandle plan_gpu;
    cufftPlan1d(&plan_gpu, FFT_SIZE, CUFFT_R2C, 1);

    float *d_signal;
    cufftComplex *d_spectrum;
    cudaMalloc(&d_signal, FFT_SIZE*sizeof(float));
    cudaMalloc(&d_spectrum, (FFT_SIZE/2 + 1)*sizeof(cufftComplex));

    cudaMemcpy(d_signal, h_signal.data(), FFT_SIZE*sizeof(float), cudaMemcpyHostToDevice);

    TIMER_START
    cufftExecR2C(plan_gpu, d_signal, d_spectrum);
    cudaDeviceSynchronize();
    TIMER_END("GPU FFT: ")

    // Vérification
    cudaMemcpy(h_spectrum_gpu.data(), d_spectrum,
              (FFT_SIZE/2 + 1)*sizeof(cufftComplex), cudaMemcpyDeviceToHost);

    float max_err = 0;
    for (int i = 0; i < FFT_SIZE/2 + 1; ++i) {
        float diff = abs(h_spectrum_cpu[i][0] - h_spectrum_gpu[i].x) +
                     abs(h_spectrum_cpu[i][1] - h_spectrum_gpu[i].y);
        max_err = std::max(max_err, diff);
    }
    std::cout << "Erreur maximale: " << max_err << "\n";

    // Nettoyage
    cufftDestroy(plan_gpu);
    cudaFree(d_signal); cudaFree(d_spectrum);
    fftwf_destroy_plan(plan_cpu);
}

int main() {
    std::cout << "Benchmark FFT \n";
    std::cout << "Taille du signal: " << FFT_SIZE << " points\n\n";
    run_fft_test();
    return 0;
}