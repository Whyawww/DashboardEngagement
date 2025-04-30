import streamlit as st
import pandas as pd

from src.fetch_data import get_videos_from_playlist, get_video_statistics
from src.preprocessing import handle_outliers_iqr, normalize_minmax, apply_pca
from src.clustering import kmeans_clustering, dbscan_clustering
from src.evaluation import evaluate_clustering
from src.visualization import plot_correlation_heatmap, plot_pca_clusters, plot_3d_scatter

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analisis | YouTube Engagement Analyzer", layout="wide")

st.title("Engagement Dashboard")
st.markdown("Analisis interaktif pola engagement audiens berdasarkan data dari channel YouTube.")

# ------------------------- SIDEBAR -------------------------
st.sidebar.title("Konfigurasi Analisis")
st.sidebar.markdown("Silakan masukkan data dan atur parameter clustering.")

channel_id = st.sidebar.text_input("Masukkan Channel ID")
playlist_id = st.sidebar.text_input("Masukkan Playlist ID")

st.sidebar.markdown("---")
model_choice = st.sidebar.selectbox("Pilih Metode Clustering", ("KMeans", "DBSCAN"))

if model_choice == "KMeans":
    n_clusters = st.sidebar.slider("Jumlah Cluster (KMeans)", 2, 10, 3)
else:
    eps = st.sidebar.slider("Nilai eps (DBSCAN)", 0.05, 1.0, 0.3)
    min_samples = st.sidebar.slider("Min Samples (DBSCAN)", 2, 10, 3)

start_button = st.sidebar.button("Mulai Analisis")

# ------------------------- LOGIKA UTAMA -------------------------
if start_button:
    if not playlist_id:
        st.error("Playlist ID harus diisi terlebih dahulu.")
    else:
        try:
            with st.spinner("Mengambil data dari YouTube API..."):
                video_ids = get_videos_from_playlist(playlist_id)
                df_videos = get_video_statistics(video_ids)

            st.success(f"Data berhasil diambil! Jumlah video: {len(df_videos)}")
            st.subheader("Sample Data")
            st.dataframe(df_videos.head())

            # ------------------------- PREPROCESSING -------------------------
            st.header("Preprocessing Data")
            features = df_videos[['views', 'likes', 'comments']]

            st.subheader("Distribusi Awal")
            st.dataframe(features.describe())

            df_cleaned = handle_outliers_iqr(features)
            df_normalized = normalize_minmax(df_cleaned)
            df_pca = apply_pca(df_normalized)

            st.markdown(f"Data setelah outlier handling: {len(df_cleaned)} video")
            st.subheader("Distribusi Setelah Preprocessing")
            st.dataframe(df_normalized.describe())

            # ------------------------- KORELASI -------------------------
            st.header("Korelasi Antar Fitur")
            fig_corr = plot_correlation_heatmap(df_cleaned)
            st.pyplot(fig_corr)

            # ------------------------- CLUSTERING -------------------------
            st.header("Clustering Data")
            if model_choice == "KMeans":
                cluster_labels, model = kmeans_clustering(df_normalized, n_clusters=n_clusters)
            else:
                cluster_labels, model = dbscan_clustering(df_normalized, eps=eps, min_samples=min_samples)

            df_videos = df_videos.loc[df_cleaned.index]  
            df_videos["Cluster"] = cluster_labels

            st.success(f"Clustering berhasil menggunakan metode **{model_choice}**")

            # ------------------------- EVALUASI -------------------------
            df_videos = df_videos.loc[df_cleaned.index].copy()

            # EVALUASI CLUSTERING berdasarkan hasil PCA
            st.header("Evaluasi Clustering")
            eval_result = evaluate_clustering(df_pca, cluster_labels)
            st.json(eval_result)

            # ------------------------- VISUALISASI -------------------------
            st.header("Visualisasi Hasil Clustering")

            st.subheader("PCA 2D Plot")
            fig_pca = plot_pca_clusters(df_pca, cluster_labels)
            st.pyplot(fig_pca)

            st.subheader("3D Plot: Views, Likes, Comments")
            fig_3d = plot_3d_scatter(df_cleaned, cluster_labels)
            st.plotly_chart(fig_3d)

            # ------------------------- INSIGHT -------------------------
            st.header("Insight per Cluster")
            cluster_summary = df_videos.groupby("Cluster")[['views', 'likes', 'comments']].mean().round(2)
            st.dataframe(cluster_summary)

            if -1 in df_videos['Cluster'].values:
                noise_count = (df_videos['Cluster'] == -1).sum()
                st.warning(f"Terdapat {noise_count} video yang dianggap noise (Cluster -1) oleh DBSCAN.")

            st.success("Proses analisis selesai!")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")