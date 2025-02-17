from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd


def cluster_dataframe(df, cluster_columns, max_k=20, subcluster_threshold=15):
    """
    Automatically detects the optimal number of clusters, assigns cluster labels,
    and applies secondary clustering for further refinement.

    Args:
        df (pd.DataFrame): The input DataFrame with multiple columns for clustering.
        cluster_columns (list): List of column names to use for clustering.
        max_k (int): Maximum number of clusters to evaluate.
        subcluster_threshold (int): The number of points in a cluster before applying secondary clustering.

    Returns:
        pd.DataFrame: The refined DataFrame with an updated 'cluster' column.
    """
    # Generate cluster text parameter by concatenating specified columns
    df['cluster_text_param'] = df[cluster_columns].astype(str).agg(' '.join, axis=1)

    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['cluster_text_param'])

    wcss = []
    silhouette_scores = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    # Determine the "elbow" point
    elbow_k = np.diff(wcss, 2).argmin() + 2
    best_silhouette_k = np.argmax(silhouette_scores) + 2
    optimal_k = (elbow_k + best_silhouette_k) // 2
    optimal_k = min(optimal_k, max_k)

    # Apply initial clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    df['primary_cluster'] = kmeans.fit_predict(X)
    df['sub_cluster'] = -1  # Default value

    # SECONDARY CLUSTERING FOR LARGE CLUSTERS
    for cluster_id in df['primary_cluster'].unique():
        cluster_size = len(df[df['primary_cluster'] == cluster_id])
        if cluster_size > subcluster_threshold:
            print(f"Refining cluster {cluster_id} (Size: {cluster_size})...")

            # Extract subset
            sub_df = df[df['primary_cluster'] == cluster_id].copy()

            # Reduce dimensionality for better clustering
            reduced_X = PCA(n_components=10).fit_transform(vectorizer.transform(sub_df['cluster_text_param']).toarray())

            # Apply Agglomerative Clustering for finer grouping
            sub_clusters = AgglomerativeClustering(n_clusters=3).fit_predict(reduced_X)
            df.loc[df['primary_cluster'] == cluster_id, 'sub_cluster'] = sub_clusters

    # Create a combined cluster label
    df['cluster'] = df['primary_cluster'].astype(str) + "-" + df['sub_cluster'].astype(str)
    #drop the intermediate columns
    df.drop(columns=['cluster_text_param', 'primary_cluster', 'sub_cluster'], inplace=True)

    return df