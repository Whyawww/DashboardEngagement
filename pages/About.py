import streamlit as st

st.set_page_config(
    page_title="Tentang Proyek | YouTube Engagement Analyzer",
    page_icon="ℹ️",
    layout="wide"
)

st.title("ℹ️ Tentang Proyek")
st.markdown(
    """
Judul Proyek: Clustering Video Berdasarkan Pola Engagement Audiens

Tujuan:  
Mengelompokkan video YouTube berdasarkan pola engagement (views, likes, comments) untuk memberikan insight lebih mendalam tentang interaksi audiens.  
Analisis ini dapat membantu content creator, tim digital marketing, dan brand dalam memahami karakteristik audiens serta menyusun strategi konten yang tepat.

Metode yang Digunakan:
- Pengambilan Data: Menggunakan YouTube API v3  
- Preprocessing: Outlier handling dengan IQR, normalisasi dengan Min-Max Scaling, reduksi dimensi dengan PCA  
- Clustering: Implementasi KMeans dan DBSCAN  
- Evaluasi: Menggunakan Silhouette Score, Davies-Bouldin Index, dan Calinski-Harabasz Score  
- Visualisasi: Heatmap korelasi, PCA 2D plot, 3D scatter plot

Manfaat:
Dashboard ini memungkinkan content creator untuk melihat segmen audiens dengan jelas dan mengoptimasi strategi konten berbasis data.  
    """
)

st.markdown("---")
st.subheader("Kontak & Referensi")
st.markdown(
    """
Jika ada pertanyaan, silakan hubungi:  
Email: wajinusantara05@gmail.com  
GitHub: https://github.com/Whyawww
    """
)

st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)