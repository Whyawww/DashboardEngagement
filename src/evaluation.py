import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluate_clustering(data: pd.DataFrame, labels) -> dict:
    """
    Mengevaluasi hasil clustering menggunakan 3 metrik:
    - Silhouette Score
    - Davies-Bouldin Index
    - Calinski-Harabasz Score

    Args:
        data (pd.DataFrame): Data numerik setelah preprocessing.
        labels (array-like): Label hasil clustering dari KMeans.

    Returns:
        dict: Dictionary berisi hasil evaluasi clustering.
    """
    n_clusters = len(set(labels))  # KMeans tidak memiliki label -1

    if n_clusters > 1:
        silhouette = silhouette_score(data, labels)
        davies_bouldin = davies_bouldin_score(data, labels)
        calinski_harabasz = calinski_harabasz_score(data, labels)
        
        return {
            "n_clusters": n_clusters,
            "silhouette_score": round(silhouette, 4),
            "davies_bouldin_index": round(davies_bouldin, 4),
            "calinski_harabasz_score": round(calinski_harabasz, 2)
        }
    else:
        return {
            "n_clusters": n_clusters,
            "silhouette_score": None,
            "davies_bouldin_index": None,
            "calinski_harabasz_score": None,
            "note": "Clustering tidak valid (hanya 1 cluster terbentuk)."
        }
