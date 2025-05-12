import streamlit as st
import pandas as pd

from src.fetch_data import get_videos_from_playlist, get_video_statistics
from src.preprocessing import add_engagement_rate, handle_outliers_iqr, normalize_minmax, apply_pca
from src.clustering import kmeans_clustering
from src.evaluation import evaluate_clustering
from src.visualization import plot_correlation_heatmap, plot_pca_clusters, plot_3d_scatter

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analisis | YouTube Engagement Analyzer", layout="wide")

st.title("Engagement Dashboard")
st.markdown("🔍 Analisis interaktif pola engagement audiens berdasarkan data dari channel YouTube.")

# ------------------------- SIDEBAR -------------------------
st.sidebar.title("Konfigurasi Analisis")
playlist_id = st.sidebar.text_input("Masukkan Playlist ID")
n_clusters = st.sidebar.slider("Jumlah Cluster (KMeans)", 2, 10, 3)
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
                df_videos = add_engagement_rate(df_videos)

            st.success(f"Data berhasil diambil! Jumlah video: {len(df_videos)}")
            st.subheader("Sample Data")
            st.dataframe(df_videos.head())

            # ------------------------- PREPROCESSING -------------------------
            st.header("Preprocessing Data")
            selected_features = ["views", "likes", "comments", "engagement_rate"]
            df_selected = df_videos[selected_features]

            st.subheader("Distribusi Awal")
            st.dataframe(df_selected.describe())

            df_cleaned = handle_outliers_iqr(df_selected)
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
            st.header("Clustering Data dengan KMeans")
            cluster_labels, model = kmeans_clustering(df_normalized, n_clusters=n_clusters)
            df_result = df_videos.loc[df_cleaned.index].copy()
            df_result["Cluster"] = cluster_labels

            st.success(f"Clustering berhasil menggunakan KMeans (k = {n_clusters})")

            # ------------------------- EVALUASI -------------------------
            eval_result = evaluate_clustering(df_normalized, cluster_labels)

            st.header("Evaluasi Clustering")
            st.json(eval_result)

            st.subheader("Distribusi Rata-rata per Cluster")
            cluster_summary = df_result.groupby("Cluster")[["views", "likes", "comments", "engagement_rate"]].mean().round(2)
            st.dataframe(cluster_summary)

            # ------------------------- VISUALISASI -------------------------
            st.header("Visualisasi Hasil Clustering")

            st.subheader("PCA 2D Plot")
            fig_pca = plot_pca_clusters(df_pca, cluster_labels)
            st.pyplot(fig_pca)

            st.subheader("3D Plot: Views, Likes, Engagement Rate")
            fig_3d = plot_3d_scatter(df_result, cluster_labels)
            st.plotly_chart(fig_3d)

            # ------------------------- UNDUH HASIL -------------------------
            st.success("Proses analisis selesai!")
            csv = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Analisis (CSV)",
                data=csv,
                file_name='hasil_clustering.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")
            st.warning("Pastikan Playlist ID yang dimasukkan benar.")
