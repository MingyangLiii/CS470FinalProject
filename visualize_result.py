# visualize the dbscan_clustering.csv

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# scatter plot of PC1 vs. PC2 colored by cluster 
def scatter_plot(df, x, y, hue, name=""):
    sns.scatterplot(x=x, y=y, hue=hue, data=df)
    plt.title(f"{x} vs. {y} colored by {hue} using {name}")
    plt.show()


# use scatterplot to visualize each pair of feature colored by diagnosis 
# put all plots in one image
def scatter_plot_all(df, features):
    n = len(features)
    fig, axes = plt.subplots(n, n, figsize=(15, 15))
    
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            
            if i == j:
                ax.hist(df[features[i]], bins=20, alpha=0.7)
                ax.set_title(f"{features[i]} distribution")
            else:
                sns.scatterplot(x=features[j], y=features[i], hue='diagnosis', 
                              data=df, ax=ax, palette='viridis', s=30, alpha=0.7)
                ax.set_title(f"{features[i]} vs {features[j]}")
            
            if i == n - 1:
                ax.set_xlabel(features[j])
            else:
                ax.set_xlabel('')
                
            if j == 0:
                ax.set_ylabel(features[i])
            else:
                ax.set_ylabel('')
    
    plt.tight_layout()
    plt.show()



def visualize_db_scan():
    path = "dbscan_clustering.csv"
    df = pd.read_csv(path)

    scatter_plot(df, "PC1", "PC2", "cluster", "DBSCAN")
    # scatter_plot(df, "PC1", "PC3", "cluster")
    # scatter_plot(df, "PC1", "PC4", "cluster")
    # scatter_plot(df, "PC1", "PC5", "cluster")


def visualize_clique():
    path = "clique_clustering.csv"
    df = pd.read_csv(path)

    scatter_plot(df, "PC1", "PC2", "cluster", "CLIQUE")

def visualize_clique_results(results, var="tau"):
    """
    results: list of dicts like:
    {"tau": value, "xi": value, "sse": value, "silhouette_score": value}
    """
    df = pd.DataFrame(results)
    other_var = "xi" if var == "tau" else "tau"
    best_row = df.loc[df["sse"].idxmin()]
    best_other_value = best_row[other_var]
    df_plot = df[df[other_var] == best_other_value].copy()
    df_plot = df_plot.sort_values(var)

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot var vs SSE
    axes[0].plot(df_plot[var], df_plot["sse"], marker="o")
    axes[0].set_title(f"{var} vs SSE (fixed {other_var}={best_other_value}) using CLIQUE")
    axes[0].set_xlabel(var)
    axes[0].set_ylabel("SSE")
    axes[0].grid(True)

    # Plot var vs Silhouette Score
    axes[1].plot(df_plot[var], df_plot["silhouette_score"], marker="o")
    axes[1].set_title(f"{var} vs Silhouette (fixed {other_var}={best_other_value}) using CLIQUE")
    axes[1].set_xlabel(var)
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()





def visualize_dbscan_results(results, var="eps"):
    """
    results: list of dicts like:
    {"eps": value, "neighbors": value, "sse": value, "silhouette_score": value}
    """

    df = pd.DataFrame(results)
    other_var = "neighbors" if var == "eps" else "eps"
    best_row = df.loc[df["sse"].idxmin()]
    best_other_value = best_row[other_var]
    df_plot = df[df[other_var] == best_other_value].copy()
    df_plot = df_plot.sort_values(var)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- SSE plot ---
    axes[0].plot(df_plot[var], df_plot["sse"], marker="o")
    axes[0].set_title(f"{var} vs SSE (fixed {other_var}={best_other_value}) using DBSCAN")
    axes[0].set_xlabel(var)
    axes[0].set_ylabel("SSE")
    axes[0].grid(True)

    # --- Silhouette plot ---
    axes[1].plot(df_plot[var], df_plot["silhouette_score"], marker="o")
    axes[1].set_title(f"{var} vs Silhouette (fixed {other_var}={best_other_value}) using DBSCAN")
    axes[1].set_xlabel(var)
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()






    
def main():

    path = "preprocessed_data.csv"
    df = pd.read_csv(path)
    features = ["PC1", "PC2", "PC3", "PC4", "PC5"]
    scatter_plot(df, "PC1", "PC2", "diagnosis", "Label")





if __name__ == "__main__":
    main()