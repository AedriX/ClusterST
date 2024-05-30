import streamlit as st


def about_page():
    st.title("Tentang Aplikasi Clustering Data Meteorologi")

    st.markdown("""
    ### Deskripsi Aplikasi
    Aplikasi ini dibuat untuk menganalisis data meteorologi menggunakan metode Fuzzy C-Means Clustering. 
    Dengan aplikasi ini, pengguna dapat mengelompokkan data meteorologi dari berbagai kota berdasarkan 
    parameter-parameter seperti temperatur, kelembapan, curah hujan, lama penyinaran matahari, dan kecepatan angin.

    ### Fitur Utama
    - **Unggah File Excel**: Pengguna dapat mengunggah file Excel yang berisi data meteorologi untuk dianalisis.
    - **Preprocessing Data**: Data yang diunggah akan diproses, termasuk normalisasi dan penamaan ulang kolom sesuai deskripsi.
    - **Fuzzy C-Means Clustering**: Data akan dikelompokkan ke dalam beberapa klaster menggunakan algoritma Fuzzy C-Means.
    - **Visualisasi Klaster**: Hasil clustering akan divisualisasikan dalam bentuk scatter plot, serta grafik time-series berdasarkan rata-rata tahunan dan bulanan.
    - **Analisis Tren Kota**: Aplikasi juga menampilkan grafik tren untuk setiap kota yang dianalisis.

    ### Panduan Penggunaan
    1. **Unggah File**: Klik tombol "Unggah file Excel" untuk mengunggah file Excel yang berisi data meteorologi.
    2. **Unduh Template**: Jika Anda tidak memiliki file yang sesuai, unduh template Excel dengan mengklik tombol "Unduh Template Excel".
    3. **Pilih Fitur untuk Dibandingkan**: Pilih fitur yang ingin dibandingkan dalam analisis clustering.
    4. **Pilih Jumlah Klaster**: Gunakan slider untuk memilih jumlah klaster yang diinginkan.
    5. **Lihat Hasil Clustering**: Lihat hasil clustering dalam bentuk tabel dan visualisasi grafik.

    ### Metodologi
    - **Normalisasi Data**: Data dinormalisasi menggunakan `StandardScaler` dari scikit-learn untuk memastikan setiap fitur memiliki skala yang sama.
    - **Fuzzy C-Means Clustering**: Metode ini digunakan untuk mengelompokkan data berdasarkan kedekatan fitur. Tidak seperti K-Means, setiap data dapat memiliki keanggotaan di beberapa klaster dengan derajat tertentu.
    - **Visualisasi**: Hasil clustering divisualisasikan untuk memudahkan analisis dan pemahaman pola data.

    ### Tentang Pengembang
    Aplikasi ini dikembangkan untuk membantu analisis data meteorologi dengan pendekatan yang interaktif dan mudah digunakan. Jika Anda memiliki pertanyaan atau masukan, silakan hubungi kami melalui halaman kontak.

    ### Sumber
    - **Pustaka Python**: 
        - `pandas`
        - `numpy`
        - `seaborn`
        - `matplotlib`
        - `streamlit`
        - `fcmeans`
        - `scikit-learn`
    - **Template Excel**: Template file Excel disediakan untuk memudahkan pengguna dalam mempersiapkan data yang sesuai format.

    Kami berharap aplikasi ini dapat bermanfaat untuk analisis data meteorologi Anda!
    """)

if __name__ == "__main__":
    about_page()
