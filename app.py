import streamlit as st
from about import about_page

from clus import cluster_page
from home import home_page


# Fungsi untuk mengelola state navigasi antar-halaman
def main():
    st.sidebar.title("Navigasi")
    pages = {
        "Home": home_page,
        "Analisis Cluster": cluster_page,
        "About": about_page
    }
    selection = st.sidebar.radio("Pilih Halaman", list(pages.keys()))
    page = pages[selection]
    with st.spinner(f"Memuat {selection} ..."):
        page()

if __name__ == "__main__":
    main()
