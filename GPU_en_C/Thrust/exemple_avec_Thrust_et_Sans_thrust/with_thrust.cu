#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <iostream>

int main() {
    // Données CPU
    std::vector<float> h_a = {1.0, 2.0, 3.0};
    std::vector<float> h_b = {4.0, 5.0, 6.0};

    // Copie vers GPU (géré automatiquement par Thrust)
    thrust::device_vector<float> d_a = h_a;
    thrust::device_vector<float> d_b = h_b;
    thrust::device_vector<float> d_c(h_a.size());

    // Addition GPU (sans écrire de kernel)
    thrust::transform(d_a.begin(), d_a.end(), d_b.begin(), d_c.begin(), thrust::plus<float>());

    // Copie du résultat vers CPU
    thrust::copy(d_c.begin(), d_c.end(), h_a.begin());

    // Affichage
    for (float val : h_a) std::cout << val << " ";  // Affiche "5 7 9"
}