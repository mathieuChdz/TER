#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <iostream>

//effectuer un tri lexicographique sur plusieurs clés.

// Met à jour le vecteur de permutation en fonction des clés données
template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // Stockage temporaire pour les clés
    KeyVector temp(keys.size());

    // Permute les clés selon l'ordre actuel de permutation
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // Trie stablement les clés permutées et met à jour la permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}

// Applique la permutation aux clés
template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // Copie les clés dans un vecteur temporaire
    KeyVector temp(keys.begin(), keys.end());

    // Permute les clés
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

// Génère un vecteur d'entiers aléatoires
thrust::host_vector<int> random_vector(size_t N)
{
    thrust::host_vector<int> vec(N);
    static thrust::default_random_engine rng;
    static thrust::uniform_int_distribution<int> dist(0, 9);

    for (size_t i = 0; i < N; i++)
        vec[i] = dist(rng);

    return vec;
}

int main(void)
{
    size_t N = 20;

    // Génère trois tableaux de valeurs aléatoires
    thrust::device_vector<int> upper  = random_vector(N);
    thrust::device_vector<int> middle = random_vector(N);
    thrust::device_vector<int> lower  = random_vector(N);

    std::cout << "Clés non triées" << std::endl;
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "( " << upper[i] << " , " << middle[i] << " , " << lower[i] << " )" << std::endl;
    }

    // Initialise la permutation à [0, 1, 2, ... ,N-1]
    thrust::device_vector<int> permutation(N);
    thrust::sequence(permutation.begin(), permutation.end());

    // Trie des clés de la moins significative à la plus significative
    update_permutation(lower,  permutation);
    update_permutation(middle, permutation);
    update_permutation(upper,  permutation);

    // Remarque : les clés n'ont pas été modifiées
    // Remarque : la permutation mappe maintenant les clés non triées à l'ordre trié

    // Applique la permutation finale aux tableaux de clés
    apply_permutation(lower,  permutation);
    apply_permutation(middle, permutation);
    apply_permutation(upper,  permutation);

    std::cout << "Clés triées" << std::endl;
    for (size_t i = 0; i < N; i++)
    {
        std::cout << "( " << upper[i] << " , " << middle[i] << " , " << lower[i] << " )" << std::endl;
    }

    return 0;
}
