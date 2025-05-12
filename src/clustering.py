import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def kmeans_clustering(data: pd.DataFrame, n_clusters: int = 3, random_state: int = 42):
    """
    Melakukan clustering menggunakan KMeans.

    Args:
        data (pd.DataFrame): Data numerik yang sudah dinormalisasi.
        n_clusters (int): Jumlah cluster yang diinginkan (default = 3).
        random_state (int): Seed untuk hasil yang reproducible.

    Returns:
        labels (np.array): Label cluster untuk setiap data point.
        model (KMeans): Model KMeans yang sudah dilatih.
    """
    model = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=random_state)
    labels = model.fit_predict(data)
    return labels, model

def evaluate_clustering(data: pd.DataFrame, labels):
    """
    Mengevaluasi hasil clustering menggunakan 3 metrik: Silhouette, Davies-Bouldin, Calinski-Harabasz.

    Args:
        data (pd.DataFrame): Data numerik yang sudah diproses (misal hasil PCA).
        labels (np.array): Label cluster untuk setiap data point.

    Returns:
        dict: Dictionary berisi hasil evaluasi.
    """
    n_clusters = len(set(labels))

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
            "note": "Clustering tidak valid (mungkin semua masuk dalam satu cluster)."
        }