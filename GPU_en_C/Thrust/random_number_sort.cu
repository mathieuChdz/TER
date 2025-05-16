#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/random.h>

//génère des nombres aléatoires en série, puis les transfère vers un appareil parallèle où ils sont triés.

int main() {
// Générer 32 millions de nombres aléatoires en série.
  thrust::default_random_engine rng(1337);
  thrust::uniform_int_distribution<int> dist;
  thrust::host_vector<int> h_vec(32 << 20);
  thrust::generate(h_vec.begin(), h_vec.end(), [&] { return dist(rng); });

  //Transférer les données.
  thrust::device_vector<int> d_vec = h_vec;

// Trier les données
  thrust::sort(d_vec.begin(), d_vec.end());

// Transférer les données triées vers l'hôte.
  thrust::copy(d_vec.begin(), d_vec.end(), h_vec.begin());
}