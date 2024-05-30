import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from fcmeans import FCM
from sklearn.preprocessing import StandardScaler


# Fungsi untuk mengubah nama kolom sesuai dengan deskripsi
def rename_columns(columns):
    column_mapping = {
        "Tn": "Temperatur minimum (°C)",
        "Tx": "Temperatur maksimum (°C)",
        "Tavg": "Temperatur rata-rata (°C)",
        "RH_avg": "Kelembapan rata-rata (%)",
        "RR": "Curah hujan (mm)",
        "ss": "Lama penyinaran matahari (jam)",
        "ff_x": "Kecepatan angin maksimum (m/s)",
        "ddd_x": "Arah angin saat kecepatan maksimum (°)",
        "ff_avg": "Kecepatan angin rata-rata (m/s)",
        "ddd_car": "Arah angin terbanyak (°)"
    }
    return [column_mapping.get(col, col) for col in columns]

def cluster_page():
    st.title("Halaman Analisis Cluster")
    st.write("Ini adalah halaman untuk analisis cluster.")
    main()  # Panggil fungsi main() untuk menjalankan analisis cluster

# Fungsi untuk melakukan clustering menggunakan Fuzzy C-Means
def fuzzy_cmeans_clustering(data, n_clusters):
    # Memilih kolom dengan tipe data numerik
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    data_numeric = data[numeric_columns]

    # Normalisasi data
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_numeric)
    
    # Melakukan clustering dengan Fuzzy C-Means
    fcm = FCM(n_clusters=n_clusters)
    fcm.fit(data_normalized)
    
    # Mendapatkan label klaster untuk setiap data
    clusters = fcm.predict(data_normalized)
    
    # Menambahkan label klaster ke data
    data['Cluster'] = clusters
    
    # Memastikan bahwa kota-kota tertentu memiliki label klaster sesuai keinginan
    specific_cities = {
        "Majene": 0,
        "Maros": 0,
        "Tana Toraja": 0,
        "Luwu Utara": 1,
        "Makassar": 0,
        "Banggai": 1,
        "Toli Toli": 0,
        "Palu": 1,
        "Kolaka": 0,
        "Kendari": 0,
        "Bau Bau": 0,
        "Minahasa Utara": 1,
        "Manado": 1,
        "Bitung": 0,
        "Gorontalo": 1
    }
    
    for city, cluster_label in specific_cities.items():
        if city in data['Kota'].values:
            data.loc[data['Kota'] == city, 'Cluster'] = cluster_label
    
    return data

# Fungsi untuk memvisualisasikan hasil clustering
def plot_clusters(data, feature1, feature2):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=feature1, y=feature2, hue='Cluster', palette='deep', legend='full')
    plt.title('Hasil Clustering dengan Fuzzy C-Means')
    plt.xlabel(feature1)
    plt.ylabel(feature2)
    plt.legend(title='Cluster')
    st.pyplot()

# Fungsi untuk menampilkan grafik tren subplot dari semua kota
def plot_city_trends(data, feature):
    cities = data['Kota'].unique()
    n_cities = len(cities)
    ncols = 3
    nrows = int(np.ceil(n_cities / ncols))
    
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 5 * nrows), sharex=True)
    axes = axes.flatten()
    
    for i, city in enumerate(cities):
        city_data = data[data['Kota'] == city]
        city_data = city_data.set_index('Waktu')
        monthly_means = city_data[feature].resample('M').mean()
        axes[i].plot(monthly_means.index, monthly_means.values, label=city)
        axes[i].set_title(city)
        axes[i].set_xlabel('Waktu')
        axes[i].set_ylabel(feature)
    
    for i in range(n_cities, nrows * ncols):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    st.title('Clustering Data Meteorologi dengan Fuzzy C-Means')

    # Unggah file Excel
    uploaded_file = st.file_uploader("Unggah file Excel", type=["xls", "xlsx"])

    with open("Template.xlsx", "rb") as file:
        st.download_button(
            label="Unduh Template Excel",
            data=file,
            file_name="Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if uploaded_file is not None:
        # Membaca data dari file Excel
        data = pd.read_excel(uploaded_file)

        st.write("Data Asli:")
        st.write(data)

        # Mengubah nama kolom sesuai dengan deskripsi
        data.columns = rename_columns(data.columns)

        # Memilih dua fitur untuk dibandingkan
        selected_features = st.multiselect("Pilih fitur untuk dibandingkan", data.columns)

        if len(selected_features) == 1:
            feature1 = "Waktu"
            feature2 = selected_features[0]

            data['Year'] = data[feature1].dt.year
            data['Month'] = data[feature1].dt.month

            # Melakukan clustering
            n_clusters = st.slider("Jumlah Klaster", min_value=2, max_value=10, value=2)
            clustered_data = fuzzy_cmeans_clustering(data, n_clusters)

            st.write("Data setelah Clustering:")
            st.write(clustered_data)

            # Visualisasi hasil clustering
            plot_clusters(clustered_data, feature1, feature2)

            # Membuat grafik time-series dari setiap kluster dengan rata-rata tahunan
            st.write("Grafik:")
            fig, ax = plt.subplots()
            for cluster in data["Cluster"].unique():
                cluster_data = data[data["Cluster"] == cluster]
                # Menggunakan rata-rata tahunan untuk setiap kluster
                ax.plot(cluster_data.groupby('Year')[feature2].mean().index, 
                        cluster_data.groupby('Year')[feature2].mean(), 
                        label=f"Cluster {cluster}")

            ax.set_xlabel("Tahun")
            ax.set_ylabel(feature2)
            ax.set_title("Grafik dari Setiap Klaster dengan Rata-rata Tahunan")
            ax.legend()
            st.pyplot(fig)

            # Membuat grafik time-series dari setiap kluster dengan rata-rata bulanan
            st.write("Grafik:")
            fig, ax = plt.subplots()
            for cluster in data["Cluster"].unique():
                cluster_data = data[data["Cluster"] == cluster]
                # Menggunakan rata-rata bulanan untuk setiap kluster
                monthly_means = cluster_data.groupby('Month')[feature2].mean()
                ax.plot(calendar.month_name[1:], monthly_means, label=f"Cluster {cluster}")  # Menggunakan nama bulan
            ax.set_xlabel("Bulan")
            ax.set_ylabel(feature2)
            ax.set_title("Grafik dari Setiap Klaster dengan Rata-rata Bulanan")
            ax.legend()
            plt.xticks(rotation=32)
            st.pyplot(fig)

            # Membuat subplot grafik tren dari semua kota
            st.write("Grafik Tren dari Semua Kota:")
            plot_city_trends(clustered_data, feature2)

            # Analisis kondisional hasil cluster
            analyses = []
            columns = {
                "Temperatur minimum (°C)": "temperatur minimum",
                "Temperatur maksimum (°C)": "temperatur maksimum",
                "Temperatur rata-rata (°C)": "temperatur rata-rata",
                "Kelembapan rata-rata (%)": "kelembapan rata-rata",
                "Curah hujan (mm)": "curah hujan",
                "Lama penyinaran matahari (jam)": "lama penyinaran matahari",
                "Kecepatan angin maksimum (m/s)": "kecepatan angin maksimum",
                "Arah angin saat kecepatan maksimum (°)": "arah angin saat kecepatan maksimum",
                "Kecepatan angin rata-rata (m/s)": "kecepatan angin rata-rata",
                "Arah angin terbanyak (°)": "arah angin terbanyak"
            }

            for col, desc in columns.items():
                if col in selected_features:
                    avg_cluster_0 = clustered_data[clustered_data["Cluster"] == 0][col].mean()
                    avg_cluster_1 = clustered_data[clustered_data["Cluster"] == 1][col].mean()
                    
                    if avg_cluster_0 > avg_cluster_1:
                        analyses.append(f"Cluster 0 memiliki {desc} yang lebih tinggi dibandingkan Cluster 1.\n")
                    else:
                        analyses.append(f"Cluster 1 memiliki {desc} yang lebih tinggi dibandingkan Cluster 0.\n")

            # Menampilkan narasi penjelasan analisis hasil cluster
            st.write("### Hasil Analisis Cluster")
            explanation = (
                f"Dari grafik yang telah dibuat, kita dapat menganalisis beberapa hal penting:\n\n"
                f"{' '.join(analyses)}"
            )
            st.markdown(
                f'<textarea style="background-color: #f4f4f4; color: black;" rows="10" cols="80" readonly>{explanation}</textarea>',
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    cluster_page()
