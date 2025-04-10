import cudf
import cugraph
from scipy.io import mmread
from scipy.sparse import coo_matrix

# Lire le fichier mtx avec scipy
def read_mtx(matrix_file):
    """
    Lit un fichier Matrix Market (.mtx) et retourne un DataFrame cuDF.
    """
    # Lire la matrice au format Matrix Market
    matrix = mmread(matrix_file)
    
    # Créer un DataFrame cuDF
    df = cudf.DataFrame({
        'src': matrix.row,
        'dst': matrix.col,
        'weight': matrix.data
    })
    print(f"Nombre d'arêtes avec poids nul : {len(df[df['weight'] == 0])}")
    return df

def create_graph(edges):
    """
    Crée un graphe à partir d'un DataFrame cuDF contenant les arêtes.
    """
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')
    return G

matrix = "webbase-1M.mtx"

G = create_graph(read_mtx("matrices/" + matrix))

# Affichage des arêtes pour test
print(G.view_edge_list())

# Calcul de PageRank
# cugraph.pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)