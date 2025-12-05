import math
from utils import load_csv, Logger
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from visualize_result import visualize_db_scan, visualize_dbscan_results
from evaluate import calculate_sse
from sklearn.metrics import silhouette_score

def plot_k_distance(df, k):
    X = df.values

    # Fit nearest neighbors
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # Distance to the k-th neighbor
    kth_distances = np.sort(distances[:, k-1])

    # Plot
    plt.plot(kth_distances)
    plt.ylabel(f"{k}-distance")
    plt.xlabel("Points sorted by distance")
    plt.title("k-distance Graph for Choosing eps")
    plt.show()


# for any two points, the distance should be the sum of euclidean distance of PC1 vs. every other feature
def my_distance(x, y):
    dx = x[0] - y[0]
    diff = x[1:] - y[1:]
    distances = np.sqrt(dx*dx + diff*diff)

    return np.sum(distances)




# DBScan Clustering
def dbscan(df, eps, min_samples, use_scale):

    X = df.values
    if use_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


    model = DBSCAN(eps=eps, min_samples=min_samples, metric=my_distance)
    labels = model.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = labels

    return labels, model, df_out



def experiment():
    path = "preprocessed_data.csv"
    df = load_csv(path)
    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)


    eps_cand = [i for i in range(1, 6)]
    neib_cand = [i for i in range(6, 24, 2)]

    search_grid = [[(eps, neib) for neib in neib_cand] for eps in eps_cand]
    results_grid = []


    min_sse = float('inf')
    best_eps = 0
    best_neib = 0



    for _ in search_grid:
        for __ in _:
            eps = __[0]
            neib = __[1]

            labels, model, df_out = dbscan(df, eps, neib, False)
            print("eps: ", eps)
            print("neib: ", neib)
            print("labels set: ", set(labels))

            # visualize_db_scan()
            sse = calculate_sse(df_out)
            print("sse: ", sse)

            if sse < min_sse and len(set(labels)) == 2:
                min_sse = sse
                best_eps = eps
                best_neib = neib           
            
            if len(set(labels)) == 2:
                results_grid.append(
                    {"eps":eps,
                    "neighbors":neib,
                    "sse": sse,
                    "silhouette_score": silhouette_score(df, labels)
                    }
                )

    print("min_sse: ", min_sse)
    print("best_eps: ", best_eps)
    print("best_neib", best_neib)

    labels, model, df_out = dbscan(df, best_eps, best_neib, False)
    df_out.to_csv("dbscan_clustering.csv", index=False)
    visualize_db_scan()

    visualize_dbscan_results(results_grid, "eps")
    visualize_dbscan_results(results_grid, "neighbors")




def main():
    path = "preprocessed_data.csv"
    df = load_csv(path)


    # view the data
    print(df.head(10))
    print("labels set: ", set(df['diagnosis']))

    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)

    eps = 2
    neib = 20



    labels, model, df_out = dbscan(df, eps, neib, False)
    print(df_out.head(10))
    sse = calculate_sse(df_out)
    score = silhouette_score(df, labels)
    print("sse: ", sse)
    print("labels set: ", set(labels))


    df_out.to_csv("dbscan_clustering.csv", index=False)
    visualize_db_scan()
    
    

    



if __name__ == "__main__":
    logger = Logger("dbscan_log.txt", auto_save=True)
    # main()
    experiment()