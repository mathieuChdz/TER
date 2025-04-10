import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import coo_matrix

def extract_comments(input_file):
    """
    Extrait les lignes de commentaires (commençant par '%') d'un fichier .mtx.
    """
    comments = []
    with open(input_file, "r") as f:
        next(f)
        for line in f:
            if line.startswith("%") :
                comments.append(line[1:])
            else :
                break
    comments.append(" Weights (probabilities) havee been added to this graph.")
    return "".join(comments)

def is_valid_matrix_format(matrix_file):
    first_lines = True
    with open(matrix_file, 'r') as f:
        for line in f:
            if not line.startswith('%') and first_lines == True:
                first_lines = False
            elif not line.startswith('%') and first_lines == False :
                if len(line.split()) == 3:
                    return True
                else :
                    return False

def add_probabilities(matrix_file) :
    """
    Ajoute des probabilités aux arcs sortants d'une matrice creuse au format Matrix Market.
    La matrice est lue à partir d'un fichier .mtx, normalisée et sauvegardée dans un nouveau fichier.
    """
    if is_valid_matrix_format(matrix_file) :
        print("The matrix already has the correct format.")

    else :
        # Retirer l'extension .mtx si elle est présente
        matrix_file = matrix_file.removesuffix(".mtx") if matrix_file.endswith(".mtx") else matrix_file

        # Charger la matrice creuse au format COO
        matrix = mmread(matrix_file + ".mtx").tocoo()

        # Calculer la somme des arcs sortants pour chaque sommet (ligne)
        row_sums = np.zeros(matrix.shape[0], dtype=float)
        for i, value in zip(matrix.row, matrix.data):
            row_sums[i - 1] += value  # Les indices Matrix Market commencent à 1

        # Normaliser les poids des arcs sortants
        normalized_data = []
        for i, value in zip(matrix.row, matrix.data):
            normalized_data.append(value / row_sums[i - 1])  # Normalisation manuelle

        # Créer une nouvelle matrice normalisée
        matrix_normalized = coo_matrix((normalized_data, (matrix.row, matrix.col)), shape=matrix.shape)

        # Extraire les commentaires du fichier d'entrée
        comments = extract_comments(matrix_file + ".mtx")

        # Sauvegarder la matrice normalisée dans un nouveau fichier .mtx
        mmwrite(f"{matrix_file}N.mtx", matrix_normalized, comment = comments)
        print(("Normalized matrix saved as " + matrix_file + "N.mtx"))

def main():
    matrix_file = input("Entre the file of the matrix: ")
    add_probabilities("matrices/" + matrix_file)

main()