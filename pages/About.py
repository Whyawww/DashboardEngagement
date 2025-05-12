import streamlit as st

st.set_page_config(
    page_title="Tentang Proyek | YouTube Engagement Analyzer",
    layout="wide"
)

st.title("Tentang Proyek")
st.markdown("Clustering Video Berdasarkan Pola Engagement Audiens")

st.markdown(
    """
Proyek ini bertujuan untuk **mengelompokkan video YouTube berdasarkan pola engagement** seperti views, likes, comments, dan engagement rate.

**Siapa yang akan terbantu?**
- Content creator yang ingin memahami performa kontennya
- Tim digital marketing yang ingin mengidentifikasi pola audiens
- Brand yang ingin menargetkan iklan secara tepat

**Apa manfaatnya?**
- Menyediakan insight berbasis data untuk optimasi strategi konten
- Menemukan video dengan performa tinggi atau rendah
- Visualisasi interaktif segmen audiens
"""
)

with st.expander("Metodologi Proyek (Klik untuk lihat detail)"):
    st.markdown(
        """
**Data Collection**
- Mengambil data video dari playlist YouTube menggunakan **YouTube API v3**

**Preprocessing**
- Menambahkan fitur `engagement_rate = (likes + comments) / views`
- Menghapus outlier dengan metode **IQR**
- Normalisasi menggunakan **Min-Max Scaling**
- Reduksi dimensi dengan **PCA (2D)**

**Clustering**
- Menggunakan **KMeans Clustering**
- Parameter terbaik (`k = 3`) dipilih berdasarkan evaluasi

**Evaluasi**
- Menggunakan:
  - Silhouette Score
  - Davies-Bouldin Index
  - Calinski-Harabasz Score

**Visualisasi**
- Korelasi antar fitur
- Scatter plot PCA 2D
- Scatter plot 3D dengan Plotly (Views-Likes-EngagementRate)
        """
    )

st.markdown("---")
st.subheader("Kontak & Referensi")
col1, col2 = st.columns([1, 6])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=80)

with col2:
    st.markdown(
        """
- Email: [wahji5723@gmail.com](mailto:wahji5723@gmail.com)  
- GitHub: [github.com/Whyawww](https://github.com/Whyawww)  
        """
    )

st.markdown("---")
st.info("Kembali ke halaman *Engagement Dashboard* di sidebar untuk mencoba analisis data interaktif.")
