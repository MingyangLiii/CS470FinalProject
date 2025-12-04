from utils import load_csv
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
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


# DBScan Clustering
def dbscan(df, eps, min_samples, use_scale):

    X = df.values
    if use_scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)


    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = labels

    return labels, model, df_out






def main():
    path = "preprocessed_data.csv"
    df = load_csv(path)


    # view the data
    print(df.head(20))
    print("labels set: ", set(df['diagnosis']))

    # drop col diagnosis
    df.drop(columns=['diagnosis'], inplace=True)


    labels, model, df_out = dbscan(df, 8, 16, False)
    print(df_out.head(20))
    print("labels set: ", set(labels))

    df_out.to_csv("dbscan_clustering.csv", index=False)
    

    



if __name__ == "__main__":
    main()