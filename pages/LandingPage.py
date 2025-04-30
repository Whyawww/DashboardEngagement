import streamlit as st

st.set_page_config(
    page_title="Beranda | YouTube Engagement Analyzer",
    layout="wide"
)

# Tambahkan ini agar sidebar muncul
with st.sidebar:
    st.title("Navigasi")
    st.markdown("Gunakan sidebar ini untuk membuka halaman Engagement Dashboard.")
    st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=60)

# Konten utama
st.title("YouTube Engagement Analyzer")
st.subheader("Analisis Pola Engagement Audiens dari Channel YouTube")
st.markdown("""
Selamat datang di YouTube Engagement Analyzer — dashboard interaktif berbasis machine learning 
yang membantu kamu memahami pola interaksi audiens dari video YouTube secara otomatis.

Dengan hanya memasukkan Playlist ID, sistem akan:
- Mengambil data `views`, `likes`, dan `comments` dari YouTube API
- Membersihkan data dan menghilangkan outlier
- Melakukan clustering otomatis (KMeans / DBSCAN)
- Memberikan **visualisasi interaktif** dan **insight per cluster**

---

Gunakan halaman _Engagement Dashboard_ di menu sidebar untuk mulai menganalisis.
""")

st.info("Gunakan sidebar kiri untuk berpindah halaman ke Dashboard Analisis.")

st.image("https://cdn-icons-png.flaticon.com/512/1384/1384060.png", width=100)