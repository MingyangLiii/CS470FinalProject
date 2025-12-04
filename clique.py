from utils import load_csv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances



def build_similarity_graph_df(df, threshold=0.7, metric="cosine"):
    G : nx.Graph
    X = df.values  # convert DataFrame -> numpy array
    n = len(X)
    G = nx.Graph()
    G.add_nodes_from(range(n))

    if metric == "cosine":
        S = cosine_similarity(X)
    elif metric == "euclidean":
        D = pairwise_distances(X)
        S = -D  
    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")

    for i in range(n):
        for j in range(i+1, n):
            if S[i, j] >= threshold:
                G.add_edge(i, j)

    return G


def maximal_clique_clustering(G):
    return list(nx.find_cliques(G))


def clique_cover(cliques, n_nodes):
    cliques = sorted(cliques, key=lambda x: len(x), reverse=True)

    assigned = set()
    clusters = []

    for clique in cliques:
        clique = [node for node in clique if node not in assigned]
        if len(clique) > 1: 
            clusters.append(clique)
            assigned.update(clique)

    labels = np.full(n_nodes, -1)
    for idx, cluster in enumerate(clusters):
        for node in cluster:
            labels[node] = idx

    return clusters, labels


def clique(df, threshold=0.98, metric="cosine"):
    G = build_similarity_graph_df(df, threshold, metric)
    cliques = maximal_clique_clustering(G)
    clusters, labels = clique_cover(cliques, len(df))
    return clusters, labels

def main():
    path = "preprocessed_data.csv"
    df = load_csv(path)


    # view the data
    print(df.head(20))
    print("labels set: ", set(df['diagnosis']))

    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)

    clusters, labels = clique(df, 0.99, "cosine")
    print(clusters)
    print(labels)
    print(set(labels))

    df["cluster"] = labels

    df.to_csv("clique_clustering.csv", index=False)


  



if __name__ == "__main__":
    main()