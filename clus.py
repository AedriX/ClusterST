import calendar

import fcmeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import silhouette_score


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
    st.write("Anda dapat melakukan proses clustering dengan mengunggah file Excel yang berisikan dataset yang ingin dilakukan clustering.")
    main()  # Panggil fungsi main() untuk menjalankan analisis cluster

@st.cache_data
def fuzzy_c_means_clustering_by_city(data, feature_columns, n_clusters):
    # Menghitung rata-rata fitur untuk setiap kota
    city_averages = data.groupby('Kota')[feature_columns].mean()
    
    # Normalisasi data
    city_averages_normalized = (city_averages - city_averages.min()) / (city_averages.max() - city_averages.min())
    
    # Konversi ke numpy array
    city_averages_array = city_averages_normalized.to_numpy(dtype=np.float64)
    
    # Fuzzy C-Means
    fcm = fcmeans.FCM(n_clusters=n_clusters, m=2, max_iter=1000, error=0.005)
    fcm.fit(city_averages_array)
    
    # Mendapatkan label cluster untuk setiap kota
    city_cluster_labels = fcm.u.argmax(axis=1)
    
    # Membuat kamus untuk menyimpan label cluster setiap kota
    city_to_cluster = dict(zip(city_averages.index, city_cluster_labels))
    
    # Menetapkan label cluster untuk setiap data berdasarkan kotanya
    data['Cluster'] = data['Kota'].map(city_to_cluster)
    
    # Menghitung silhouette coefficient
    silhouette_avg = silhouette_score(city_averages_normalized, city_cluster_labels)
    
    return data, silhouette_avg, city_to_cluster

# Fungsi untuk memvisualisasikan tren rata-rata tahunan dan bulanan
def plot_trends(data, feature):
    data['Year'] = data['Waktu'].dt.year
    data['Month'] = data['Waktu'].dt.month
    
    # Tren tahunan
    yearly_trend = data.groupby(['Year', 'Cluster'])[feature].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    for cluster in yearly_trend['Cluster'].unique():
        subset = yearly_trend[yearly_trend['Cluster'] == cluster]
        plt.plot(subset['Year'], subset[feature], label=f'Cluster {cluster}')
    plt.title('Tren Rata-rata Tahunan')
    plt.xlabel('Tahun')
    plt.ylabel(f'{feature}')
    plt.xticks(ticks=yearly_trend['Year'].unique())  # Display each year individually
    plt.legend()
    st.pyplot(plt)
    
    # Tren bulanan
    monthly_trend = data.groupby(['Month', 'Cluster'])[feature].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    for cluster in monthly_trend['Cluster'].unique():
        subset = monthly_trend[monthly_trend['Cluster'] == cluster]
        plt.plot(subset['Month'], subset[feature], label=f'Cluster {cluster}')
    plt.title('Tren Rata-rata Bulanan')
    plt.xlabel('Bulan')
    plt.ylabel(f'{feature}')
    plt.xticks(ticks=range(1, 13), labels=[calendar.month_name[i] for i in range(1, 13)])
    plt.legend()
    st.pyplot(plt)

def plot_city_trends(data, feature):
    plt.figure(figsize=(14, 10))
    clusters = data['Cluster'].unique()
    num_clusters = len(clusters)
    for i, cluster in enumerate(clusters, start=1):
        plt.subplot(num_clusters, 1, i)
        cluster_data = data[data['Cluster'] == cluster]
        for city in cluster_data['Kota'].unique():
            city_data = cluster_data[cluster_data['Kota'] == city]
            plt.plot(city_data['Waktu'], city_data[feature], label=city)
        plt.title(f'Tren {feature} untuk Cluster {cluster}')
        plt.xlabel('Waktu')
        plt.ylabel(feature)
        plt.legend()
    plt.tight_layout()
    st.pyplot(plt)

def update_cluster_labels(data, city_to_cluster):
    updated_city_to_cluster = {}
    st.subheader("Perbarui Label Cluster")
    for city, cluster in city_to_cluster.items():
        new_cluster = st.selectbox(f"Pilih cluster baru untuk {city}", options=list(range(max(city_to_cluster.values())+1)), index=int(cluster), key=city)
        updated_city_to_cluster[city] = new_cluster
    data['Cluster'] = data['Kota'].map(updated_city_to_cluster)
    return data, updated_city_to_cluster

def main():
    st.title("Clustering dengan Fuzzy C-Means")
    
    uploaded_file = st.file_uploader("Unggah file Excel Anda", type=["xlsx"])

    with open("Template.xlsx", "rb") as file:
        st.download_button(
            label="Unduh Template Excel",
            data=file,
            file_name="Template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )    
    
    if uploaded_file is not None:
        data = pd.read_excel(uploaded_file)
        data['Waktu'] = pd.to_datetime(data['Waktu'])
        
        # Rename columns
        data.columns = rename_columns(data.columns)
        
        st.write("Data yang diunggah:")
        st.write(data)
        
        featureselect = st.multiselect("Pilih fitur untuk clustering", data.columns.difference(['Waktu', 'Kota']))
        
        n_clusters = st.slider("Pilih jumlah cluster", 2, 10, 3)
        
        if st.button("Lakukan Clustering"):
            feature_columns = featureselect
            clustered_data, silhouette_avg, city_to_cluster = fuzzy_c_means_clustering_by_city(data, feature_columns, n_clusters)
            
            city_cluster_df = pd.DataFrame(list(city_to_cluster.items()), columns=['Kota', 'Cluster'])

            st.session_state['clustered_data'] = clustered_data
            st.session_state['silhouette_avg'] = silhouette_avg
            st.session_state['city_cluster_df'] = city_cluster_df
            st.session_state['city_to_cluster'] = city_to_cluster

            # Allow users to update cluster labels
            #updated_clustered_data, updated_city_to_cluster = update_cluster_labels(clustered_data, city_to_cluster)

            #st.session_state['clustered_data'] = updated_clustered_data
            #st.session_state['city_cluster_df'] = pd.DataFrame(list(updated_city_to_cluster.items()), columns=['Kota', 'Cluster'])

    if 'clustered_data' in st.session_state:
        clustered_data = st.session_state['clustered_data']
        silhouette_avg = st.session_state['silhouette_avg']
        city_cluster_df = st.session_state['city_cluster_df']
        city_to_cluster = st.session_state['city_to_cluster']
        
        st.write("Data Pasca Cluster:")
        st.write(clustered_data[['Kota', 'Waktu', 'Cluster'] + feature_columns])
        
        st.write(f"Silhouette Coefficient: {silhouette_avg:.4f}")
        
        st.subheader("Tabel Kota dan Label Cluster")
        st.write(city_cluster_df)
        
        trend_feature = st.selectbox("Pilih fitur untuk ditampilkan di grafik tren", feature_columns)
        st.subheader("Visualisasi Tren")
        plot_trends(clustered_data, trend_feature)

        # Visualisasi tren per kota di setiap cluster
        st.subheader("Visualisasi Tren per Kota")
        plot_city_trends(clustered_data, trend_feature)

if __name__ == "__main__":
    cluster_page()
