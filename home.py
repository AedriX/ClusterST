import streamlit as st
from PIL import Image


def home_page():
    background_image = Image.open('background.jpg')
    st.image(background_image, use_column_width=True)

    # Add title and welcome text
    st.title("Selamat Datang!")
    st.write("Aplikasi ini memudahkan Anda dalam melakukan analisis cluster dengan mudah.")

    st.write("Contoh Hasil Clustering Data Meteorologi Pulau Sulawesi:")

    # Create a grid of 4x4 images
    cols = st.columns(2)
    for i in range(4):
        # Load the image dynamically based on the index
        image_path = f'{i+1}.jpg'
        try:
            img = Image.open(image_path)
            cols[i % 2].image(img, use_column_width=True)
        except FileNotFoundError:
            st.error(f"File {image_path} tidak ditemukan.")

if __name__ == "__main__":
    home_page()
