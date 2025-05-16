#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include <thrust/random.h>

// montre le calcul de la somme de certains nombres aléatoires en parallèle 

int main() {
// Générer des données aléatoires en série.
  thrust::default_random_engine rng(1337);
  thrust::uniform_real_distribution<double> dist(-50.0, 50.0);
  thrust::host_vector<double> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

// Transfére et calculer la somme.
  thrust::device_vector<double> d_vec = h_vec;
  double x = thrust::reduce(d_vec.begin(), d_vec.end(), 0, thrust::plus<int>());
}