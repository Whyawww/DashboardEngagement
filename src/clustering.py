import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def kmeans_clustering(data: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Melakukan clustering menggunakan KMeans.

    Args:
        data (pd.DataFrame): Data numerik yang sudah dinormalisasi.
        n_clusters (int): Jumlah cluster yang diinginkan.
        random_state (int): Seed untuk hasil yang reproducible.

    Returns:
        labels (np.array): Label cluster untuk setiap data point.
        model (KMeans): Model KMeans yang sudah dilatih.
    """
    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = model.fit_predict(data)
    return labels, model

def dbscan_clustering(data: pd.DataFrame, eps: float = 0.3, min_samples: int = 5):
    """
    Melakukan clustering menggunakan DBSCAN.

    Args:
        data (pd.DataFrame): Data numerik yang sudah dinormalisasi.
        eps (float): Jarak maksimum antar data untuk membentuk cluster.
        min_samples (int): Minimum jumlah poin dalam neighborhood untuk core point.

    Returns:S
        labels (np.array): Label cluster untuk setiap data point.
        model (DBSCAN): Model DBSCAN yang sudah dilatih.
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)
    return labels, model

def evaluate_clustering(data: pd.DataFrame, labels):
    """
    Mengevaluasi hasil clustering menggunakan 3 metrik: Silhouette, Davies-Bouldin, Calinski-Harabasz.

    Args:
        data (pd.DataFrame): Data numerik yang sudah dinormalisasi.
        labels (np.array): Label cluster untuk setiap data point.

    Returns:
        dict: Dictionary berisi hasil evaluasi.
    """
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Kalau cluster valid (≥ 2 cluster)
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
            "calinski_harabasz_score": None
        }
