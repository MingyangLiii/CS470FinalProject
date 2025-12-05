from itertools import product
from utils import load_csv, Logger
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from visualize_result import visualize_clique, visualize_clique, visualize_clique_results
from evaluate import calculate_sse
from collections import defaultdict, deque
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class CLIQUE(BaseEstimator, ClusterMixin):

    def __init__(self, xi=10, tau=0.02):
        self.xi = xi
        self.tau = tau

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values

        n_samples, n_features = X.shape
        mins = X.min(axis=0)
        maxs = X.max(axis=0)

        self.intervals_ = [
            np.linspace(mins[d], maxs[d], self.xi + 1)
            for d in range(n_features)
        ]

        def get_unit(point):
            cell = []
            for d in range(n_features):
                idx = np.searchsorted(self.intervals_[d], point[d], side='right') - 1
                idx = min(max(idx, 0), self.xi - 1)
                cell.append(idx)
            return tuple(cell)

        units = defaultdict(list)
        for idx, point in enumerate(X):
            units[get_unit(point)].append(idx)

        density_threshold = self.tau * n_samples
        dense_units = {u for u, pts in units.items() if len(pts) >= density_threshold}

        def neighbors(unit):
            for d in range(n_features):
                for off in [-1, 0, 1]:
                    nb = list(unit)
                    nb[d] += off
                    nb = tuple(nb)
                    if nb in dense_units:
                        yield nb

        cluster_id = 0
        unit_to_cluster = {}

        for u in dense_units:
            if u not in unit_to_cluster:
                queue = deque([u])
                unit_to_cluster[u] = cluster_id

                while queue:
                    cur = queue.popleft()
                    for nb in neighbors(cur):
                        if nb not in unit_to_cluster:
                            unit_to_cluster[nb] = cluster_id
                            queue.append(nb)

                cluster_id += 1

        # Assign labels
        labels = np.full(n_samples, -1)
        for unit, cid in unit_to_cluster.items():
            for idx in units[unit]:
                labels[idx] = cid

        self.labels_ = labels
        return self

    def fit_predict(self, X, y=None):
        return self.fit(X).labels_


def experiment_clique():
    path = "preprocessed_data.csv"
    df = load_csv(path)


    # view the data
    print(df.head(10))
    print("labels set: ", set(df['diagnosis']))

    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)


    min_sse = float('inf')
    best_xi = 0
    best_tau = 0
    xi = 3
    tau = 0.005


    xi_cand = [1, 2, 3, 4]
    tau_cand = [0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.021]

    search_grid = list(product(xi_cand, tau_cand))

    results_grid = []

    min_sse = float('inf')
    score = -1
    best_params = (0, 0)
    best_labels = None



    for xi, tau in search_grid:
        print(f"Testing xi={xi}, tau={tau}")
        
        clique = CLIQUE(xi=xi, tau=tau)
        labels = clique.fit_predict(df)
        
        df_temp = df.copy()
        df_temp["cluster"] = labels
        sse = calculate_sse(df_temp)
        
        print(f"Clusters: {len(set(labels))}, SSE: {sse}")
        
        if sse < min_sse and len(set(labels)) == 2:
            score = silhouette_score(df, labels)
            min_sse = sse
            best_params = (xi, tau)
            best_labels = labels

        if len(set(labels)) == 2:
            results_grid.append(
                {"tau":tau,
                "xi":xi,
                "sse": sse,
                "silhouette_score": score
                }
            )



    print(f"\nBest parameters: xi={best_params[0]}, tau={best_params[1]}")
    print(f"Best SSE: {min_sse}")
    print(f"Silhouette Score: {score}")


    clique = CLIQUE(xi=best_params[0], tau=best_params[1])
    labels = clique.fit_predict(df)
    
    df["cluster"] = labels
    print("Final labels count: ", len(set(labels)))
    print("Final SSE: ", min_sse)
    print("Silhouette Score: ", score)
    
    df.to_csv("clique_clustering.csv", index=False)
    visualize_clique()
    visualize_clique_results(results_grid, "xi")
    visualize_clique_results(results_grid, "tau")







def main():
    path = "preprocessed_data.csv"
    df = load_csv(path)


    # view the data
    print(df.head(10))
    # print("labels set: ", set(df['diagnosis']))

    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)



    clique = CLIQUE(xi=2, tau=0.018)

    labels = clique.fit_predict(df)
    print("labels count: ", len(set(labels)))

    score = silhouette_score(df, labels)
    print("Silhouette Score: ", score)
    df["cluster"] = labels
    sse = calculate_sse(df)
    print("sse: ", sse)
    visualize_clique()



  



if __name__ == "__main__":
    logger = Logger("clique_log.txt", auto_save=True)
    # main()
    experiment_clique()