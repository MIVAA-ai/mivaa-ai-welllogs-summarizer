from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import traceback

def cluster_dataframe(df, subsection_name, cluster_columns, max_k=20, subcluster_threshold=15):
    try:
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
        if df.empty:
            print(f"The DataFrame is empty for {subsection_name}. Clustering will not be performed.")
            return df  # Return the empty DataFrame as is

        # Ensure there are enough samples to perform clustering
        if len(df) < 2:
            print(f"Not enough samples for {subsection_name} to perform clustering.")
            df["cluster"] = 0
            return df

        # Ensure only existing columns are used
        valid_columns = df.columns.intersection(cluster_columns)

        # If no valid columns exist, set a default value
        if valid_columns.empty:
            print(f"Valid columns are not present for clustering {subsection_name}.")
            df["cluster"] = 0
            return df
        else:
            if set(valid_columns) != set(cluster_columns):
                print(f"All the columns are not present for clustering {subsection_name}. Using valid columns only.")

            df['cluster_text_param'] = df[valid_columns].astype(str).agg(' '.join, axis=1)

        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(df['cluster_text_param'])

        wcss = []
        silhouette_scores = []

        # Adjust max_k dynamically based on data size
        max_k = min(max_k, len(df) - 1)

        for k in range(2, max_k + 1):
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X)

                # Ensure at least 2 unique clusters exist before computing silhouette score
                if len(set(labels)) > 1:
                    wcss.append(kmeans.inertia_)
                    silhouette_scores.append(silhouette_score(X, labels))
                else:
                    break  # Stop clustering if only one cluster is formed

            except Exception as e:
                print(f"Skipping k={k} for {subsection_name} due to error: {str(e)}")
                break

        # Handle case where no valid clustering was performed
        if not wcss:
            print(f"Clustering failed for {subsection_name}. Assigning single cluster.")
            df["cluster"] = 0
            df.drop(columns=['cluster_text_param'], inplace=True)
            return df

        # Determine the "elbow" point
        elbow_k = np.diff(wcss, 2).argmin() + 2 if len(wcss) > 2 else 2
        best_silhouette_k = np.argmax(silhouette_scores) + 2 if silhouette_scores else 2
        optimal_k = (elbow_k + best_silhouette_k) // 2
        optimal_k = max(2, min(optimal_k, max_k))  # Ensure at least 2 clusters

        # Apply initial clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        df['primary_cluster'] = kmeans.fit_predict(X)
        df['sub_cluster'] = -1  # Default value

        # SECONDARY CLUSTERING FOR LARGE CLUSTERS
        for cluster_id in df['primary_cluster'].unique():
            cluster_size = len(df[df['primary_cluster'] == cluster_id])
            if cluster_size > subcluster_threshold:
                # Extract subset
                sub_df = df[df['primary_cluster'] == cluster_id].copy()

                # Reduce dimensionality for better clustering
                reduced_X = PCA(n_components=min(10, len(sub_df) - 1)).fit_transform(
                    vectorizer.transform(sub_df['cluster_text_param']).toarray()
                )

                # Apply Agglomerative Clustering for finer grouping
                num_subclusters = min(3, cluster_size)  # Ensure valid number of clusters
                sub_clusters = AgglomerativeClustering(n_clusters=num_subclusters).fit_predict(reduced_X)
                df.loc[df['primary_cluster'] == cluster_id, 'sub_cluster'] = sub_clusters

        # Create a combined cluster label
        df['cluster'] = df['primary_cluster'].astype(str) + "-" + df['sub_cluster'].astype(str)
        #drop the intermediate columns
        df.drop(columns=['cluster_text_param', 'primary_cluster', 'sub_cluster'], inplace=True)

        return df
    except Exception as e:
        print(f"Error in cluster_dataframe for {subsection_name}: {str(e)}")
        print(traceback.format_exc())
        return pd.DataFrame()