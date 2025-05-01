import streamlit as st

# Konfigurasi dasar halaman
st.set_page_config(
    page_title="YouTube Engagement Analyzer",
    layout="wide",
    page_icon="📊"
)

# Styling
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
    }
    .info-box {
        background-color: #f0f4f8;
        padding: 1.5rem;
        border-radius: 12px;
        margin-top: 1rem;
        border-left: 6px solid #4E79A7;
    }
    </style>
""", unsafe_allow_html=True)

# Konten Halaman
st.markdown('<div class="title-text">🎥 YouTube Engagement Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-text">Dashboard Interaktif untuk Menganalisis Pola Engagement Audiens</div>', unsafe_allow_html=True)

st.markdown("---")

st.markdown("""
Dashboard ini membantu kamu:
- Mengambil data video langsung dari YouTube API
- Melakukan analisis & clustering berdasarkan views, likes, dan comments
- Mendeteksi pola engagement audiens menggunakan KMeans & DBSCAN
- Memberikan visualisasi interaktif: PCA 2D, 3D Scatter, dan korelasi antar fitur
- Menyediakan insight per cluster yang dapat digunakan content creator atau peneliti

""")

st.markdown('<div class="info-box">Gunakan menu di sidebar kiri untuk memulai analisis pada halaman <strong>Engagement Dashboard</strong>.</div>', unsafe_allow_html=True)
st.markdown("---")
