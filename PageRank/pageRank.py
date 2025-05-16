from scipy.io import mmread

import cudf
import cugraph

import pandas as pd
import networkx as nx

import time
import cupy as cp

def create_dataframe_cpu(matrix):
    """
    Crée un DataFrame pandas à partir d'une matrice COO
    """
    df = pd.DataFrame({
        'src': matrix.row,
        'dst': matrix.col,
        'weight': matrix.data
    })
    return df

def create_dataframe_gpu(matrix):
    """
    Crée un DataFrame cuDF à partir d'une matrice COO.
    """
    df = cudf.DataFrame({
        'src': matrix.row,
        'dst': matrix.col,
        'weight': matrix.data
    })
    return df

def create_graph_cpu(edges):
    """
    Crée un graphe NetworkX à partir d'un DataFrame pandas contenant les arêtes.
    """
    G = nx.DiGraph()
    for _, row in edges.iterrows():
        G.add_edge(row['src'], row['dst'], weight=row['weight'])
    return G


def create_graph_gpu(edges):
    """
    Crée un graphe à partir d'un DataFrame cuDF contenant les arêtes.
    """
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')
    return G

def page_rank_cpu(matrix):
    G = create_graph_cpu(create_dataframe_cpu(matrix))

    start_time = time.time()
    nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)
    end_time = time.time()

    return end_time - start_time

def page_rank_gpu(matrix):
    G = create_graph_gpu(create_dataframe_gpu(matrix))

    start_time = time.time()
    cugraph.pagerank(G, alpha=0.85, max_iter=100, tol=1e-5)
    cp.cuda.Stream.null.synchronize()
    end_time = time.time()

    return end_time - start_time



if __name__ == "__main__":
    matrix_file = "matrices/webbase-1M.mtx"
    matrix = mmread(matrix_file)
    print(f'PageRank CPU Time: {page_rank_cpu(matrix)}s')
    print(f'PageRank GPU Time: {page_rank_gpu(matrix)}s')