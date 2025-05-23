import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import numpy as np

def add_engagement_rate(data: pd.DataFrame) -> pd.DataFrame:
    """
    Menambahkan fitur engagement_rate = (likes + comments) / views.

    Args:
        data (pd.DataFrame): Data numerik asli.

    Returns:
        pd.DataFrame: Data dengan kolom engagement_rate.
    """
    data = data.copy()
    data['engagement_rate'] = (data['likes'] + data['comments']) / data['views']
    data['engagement_rate'].replace([np.inf, -np.inf], np.nan, inplace=True)
    data['engagement_rate'].fillna(0, inplace=True)
    return data

def handle_outliers_iqr(data: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan data dari outlier menggunakan metode IQR (Interquartile Range).

    Args:
        data (pd.DataFrame): Data numerik.

    Returns:
        pd.DataFrame: Data setelah outlier dibuang.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    cleaned_data = data[~((data < lower_bound) | (data > upper_bound)).any(axis=1)]
    print(f"\nData setelah membersihkan outlier (sebelumnya {data.shape[0]}, sekarang {cleaned_data.shape[0]})")
    return cleaned_data

def normalize_minmax(data: pd.DataFrame) -> pd.DataFrame:
    """
    Menormalisasi data menggunakan Min-Max Scaling (skala 0-1).

    Args:
        data (pd.DataFrame): Data numerik.

    Returns:
        pd.DataFrame: Data setelah normalisasi.
    """
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    return normalized_data

def apply_pca(data: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    """
    Mereduksi dimensi data menggunakan PCA (Principal Component Analysis).

    Args:
        data (pd.DataFrame): Data numerik yang sudah dinormalisasi.
        n_components (int): Jumlah komponen utama yang diinginkan.

    Returns:
        pd.DataFrame: Data setelah transformasi PCA.
    """
    pca = PCA(n_components=n_components, random_state=42)
    pca_components = pca.fit_transform(data)
    columns = [f'PC{i+1}' for i in range(n_components)]
    return pd.DataFrame(pca_components, columns=columns)