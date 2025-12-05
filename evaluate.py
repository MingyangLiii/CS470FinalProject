import pandas as pd
import numpy as np
# evaluate the sse of clustering

def calculate_sse(df, label_col="cluster"): 
    X = df.drop(columns=[label_col]).values
    labels = df[label_col].values
    
    sse = 0
    for cluster in np.unique(labels):
        cluster_points = X[labels == cluster]
        centroid = cluster_points.mean(axis=0)
        sse += ((cluster_points - centroid) ** 2).sum()
    return sse
