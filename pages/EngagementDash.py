import streamlit as st
import pandas as pd
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_PATH = os.path.join(PROJECT_ROOT, 'src')
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)
from preprocessing import add_engagement_rate, handle_outliers_iqr, normalize_minmax, apply_pca
from fetch_data import get_videos_from_playlist, get_video_statistics
from clustering import kmeans_clustering
from evaluation import evaluate_clustering
from visualization import plot_correlation_heatmap, plot_pca_clusters, plot_3d_scatter

# KONFIGURASI HALAMAN
st.set_page_config(page_title="Dashboard Analisis | YouTube Engagement Analyzer", layout="wide")

st.title("Engagement Dashboard")
st.markdown("Analisis interaktif pola engagement audiens berdasarkan data dari channel YouTube.")

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

            # ------------------------- ENGAGEMENT RATE -------------------------
            df_videos = df_videos.loc[df_cleaned.index].copy()
            df_videos["Cluster"] = cluster_labels

            # Evaluasi dan tampilkan cluster summary
            eval_result = evaluate_clustering(df_normalized, cluster_labels)
            st.json(eval_result)

            # Baru Analisis Engagement Rate
            if "Cluster" in df_videos.columns:
                st.header("Analisis Engagement Rate")

                df_videos["engagement_rate"] = (
                    (df_videos["likes"] + df_videos["comments"]) / df_videos["views"]
                ).replace([float('inf'), -float('inf')], 0).fillna(0)

                engagement_summary = df_videos.groupby("Cluster")["engagement_rate"].mean().reset_index()
                engagement_summary.columns = ["Cluster", "Rata-rata Engagement Rate"]

                import matplotlib.pyplot as plt
                import seaborn as sns

                fig_engage, ax = plt.subplots(figsize=(10, 5))
                sns.barplot(data=engagement_summary, x="Cluster", y="Rata-rata Engagement Rate", palette="Blues_d", ax=ax)
                ax.set_title("Rata-rata Engagement Rate per Cluster")
                ax.set_xlabel("Cluster")
                ax.set_ylabel("Engagement Rate")
                st.pyplot(fig_engage)
            else:
                st.warning("Pastikan proses clustering berhasil sebelum menampilkan engagement rate.")


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
