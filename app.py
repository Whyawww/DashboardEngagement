import streamlit as st

# Konfigurasi dasar halaman
st.set_page_config(
    page_title="YouTube Engagement Analyzer",
    layout="wide",
    page_icon=""
)

# Styling Kustom
st.markdown("""
    <style>
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4E79A7;
    }
    .subtitle-text {
        font-size: 1.2rem;
        color: #444;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f4f8;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        border-left: 6px solid #4E79A7;
    }
    ul li {
        margin-bottom: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title-text">🎥 YouTube Engagement Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Dashboard Interaktif untuk Menganalisis Pola Engagement Audiens</div>', unsafe_allow_html=True)

st.markdown("---")

# Deskripsi
st.markdown("""
Dashboard ini dirancang untuk membantu **Content Creator, Digital Marketer, dan Brand** dalam:
- Mengambil data video langsung dari **YouTube API**
- Melakukan **analisis & clustering** berdasarkan `views`, `likes`, `comments`, dan `engagement_rate`
- Mengidentifikasi pola interaksi audiens menggunakan **KMeans Clustering**
- Menampilkan **visualisasi interaktif**:
    - Korelasi antar fitur
    - PCA 2D cluster plot
    - 3D scatter plot (Views-Likes-Engagement)
- Menyediakan **insight per cluster** untuk mengoptimalkan strategi konten
""")

# Info Box
st.markdown("""
<div class="info-box">
Gunakan menu di sidebar kiri untuk memulai analisis pada halaman <strong>Engagement Dashboard</strong>  
Untuk penjelasan lengkap metode dan tujuan proyek, buka halaman <strong>Tentang Proyek</strong>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
