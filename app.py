import streamlit as st
import pandas as pd
import joblib

# -----------------
# Muat model pipeline yang sudah disimpan
# -----------------
# Pastikan file model_sepatu_svc.pkl berada di folder yang sama
try:
    model = joblib.load('model_sepatu_svc.pkl')
except FileNotFoundError:
    st.error("File model 'model_sepatu_svc.pkl' tidak ditemukan. Pastikan file berada di direktori yang sama.")
    st.stop()


# -----------------
# Fungsi & Tampilan Aplikasi
# -----------------

# Judul dan deskripsi aplikasi
st.title('Prediksi Kualitas Sepatu Pria')
st.write('Aplikasi ini memprediksi kategori kualitas sepatu berdasarkan fitur yang Anda masukkan.')
st.write('Model ini dilatih menggunakan algoritma **Support Vector Classifier (SVC)**.')

# Daftar brand yang ada di dataset (Anda bisa mengambil ini dari notebook)
# Ini penting agar OneHotEncoder berfungsi dengan benar
list_brand = [
    'ASIAN', 'Labbin', 'aadi', 'FEETEES', 'corsac', 'BRUTON', 'ASTEROID', 'Chevit',
    'Robbie jones', 'OROCHI', 'WOODLAND', 'Layasa', 'Kraasa', 'Zixer', 'HOTSTYLE',
    'RED TAPE', 'Bata', 'BERSACHE', 'Elevarse', 'BIRDE', 'ACTION', 'FOOTGRACE',
    'Deals4you', 'ROCKFIELD', 'bacca bucci', 'bluemaker', 'Bond Street By Red Tape',
    'RODDICK', 'Liberty', 'PUMA', 'Bucik', 'ATOM'
]

# Membuat form input di sidebar
st.sidebar.header('Masukkan Fitur Sepatu:')

def user_input_features():
    brand = st.sidebar.selectbox('Pilih Brand', sorted(list_brand))
    sold = st.sidebar.slider('Jumlah Terjual', 0, 5000, 100)
    price = st.sidebar.number_input('Harga Saat Ini (₹)', min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)

    # Buat DataFrame dari input
    data = {
        'Brand_Name': [brand],
        'How_Many_Sold': [sold],
        'Current_Price': [price]
    }
    features = pd.DataFrame(data)
    return features

# Ambil input dari user
input_df = user_input_features()

# Tampilkan data input dari user
st.subheader('Fitur yang Anda Masukkan:')
st.write(input_df)

# Tombol untuk melakukan prediksi
if st.button('Prediksi Kualitas'):
    # Lakukan prediksi menggunakan pipeline
    prediction = model.predict(input_df)
    prediction_proba = model.decision_function(input_df) # Menggunakan decision_function untuk SVC

    # Tampilkan hasil prediksi
    st.subheader('Hasil Prediksi:')
    kategori_kualitas = {
        0: 'Buruk (Rating ≤ 3.5)',
        1: 'Cukup (Rating 3.5 - 4.0)',
        2: 'Baik (Rating > 4.0)'
    }

    hasil_prediksi = kategori_kualitas.get(prediction[0], 'Tidak diketahui')

    # Memberikan output yang lebih menarik
    if prediction[0] == 2:
        st.success(f'**{hasil_prediksi}** 🎉')
        st.write("Sepatu ini kemungkinan besar memiliki kualitas yang baik dan rating tinggi.")
    elif prediction[0] == 1:
        st.warning(f'**{hasil_prediksi}** 🤔')
        st.write("Sepatu ini memiliki kualitas rata-rata.")
    else:
        st.error(f'**{hasil_prediksi}** 📉')
        st.write("Sepatu ini kemungkinan memiliki kualitas di bawah rata-rata.")