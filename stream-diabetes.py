import pickle
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_option_menu import option_menu
  
# =================================================================================
# FUNGSI DARI NOTEBOOK: Salin fungsi preprocessing dan prediksi dari notebook Anda
# untuk memastikan konsistensi.
# =================================================================================

def custom_standardize(data, mean_std_dict):
    """
    Normalisasi data numpy array menggunakan mean dan std yang sudah ada.
    """
    data_scaled = data.copy().astype(float)
    mean = mean_std_dict['mean']
    std = mean_std_dict['std']
    
    # Mencegah pembagian dengan nol jika standar deviasi adalah 0
    std[std == 0] = 1
    
    data_scaled = (data - mean) / std
    return data_scaled, mean_std_dict

def svm_predict(X, w, b):
    """
    Fungsi prediksi SVM manual dari notebook Anda.
    """
    linear_output = np.dot(X, w) - b
    # Mengembalikan label asli 0 atau 1
    return np.where(linear_output >= 0, 1, 0)

# =================================================================================
# MEMUAT MODEL DAN PARAMETER
# =================================================================================

try:
    with open('svm_diabetes_model_unified.pkl', 'rb') as f:
        saved_model_data = pickle.load(f)

    # Ekstrak semua komponen yang disimpan dari dictionary
    w_loaded = saved_model_data['weights']
    b_loaded = saved_model_data['bias']
    mean_std_dict_loaded = saved_model_data['mean_std_dict']
    feature_columns_loaded = saved_model_data['feature_columns']

except FileNotFoundError:
    st.error("File 'svm_diabetes_model_unified.pkl' tidak ditemukan. Pastikan Anda sudah menjalankan notebook untuk membuat file ini.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi error saat memuat model: {e}")
    st.stop()

# =================================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =================================================================================

# Page config
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")

# Inject CSS
st.markdown("""
<style>
.css-1cypcdb.e1fqkh3o3 { border: 2px solid #264653; border-radius: 12px; padding: 8px; background-color: #1a1c22; box-shadow: 0 2px 12px rgba(0,0,0,0.25); }
.nav-link { font-weight: 600; color: #f1faee !important; margin: 0 12px; padding: 8px 20px; border-radius: 12px; transition: all 0.3s ease-in-out; font-size: 16px; }
.nav-link:hover { background-color: #118ab2 !important; color: #ffffff !important; }
.nav-link.active { background-color: #ef476f !important; color: white !important; border: 2px solid #06d6a0; }
h1, h2, h4 { color: #ffffff; text-align: center; }
p { color: #d9d9d9; font-size: 15.5px; line-height: 1.6; }
a { color: #06d6a0; text-decoration: none; }
a:hover { text-decoration: underline; }
thead th { background-color: #264653; color: #ffffff; }
tbody td { background-color: #1e1e1e; color: #eeeeee; }
.st-emotion-cache-fmhvvr p{ text-align:center; }
</style>
""", unsafe_allow_html=True)
 
# Navigation menu
selected = option_menu(
      menu_title=None,
      options=["Home", "Input Manual", "Upload CSV"],
      icons=["house", "book", "upload"],
      menu_icon="cast",
      default_index=0,
      orientation="horizontal",
)
  
# Halaman Home
if selected == "Home":
    st.title('Aplikasi Prediksi Diabetes')
    st.write('''Aplikasi Prediksi Diabetes adalah sebuah aplikasi yang berguna untuk memprediksi kemungkinan seseorang 
    menderita diabetes berdasarkan beberapa fitur yang dimasukkan. Aplikasi ini menggunakan model Support Vector Machine (SVM) 
    yang dilatih pada dataset Pima Indians Diabetes dari Kaggle. Dengan memasukkan fitur yang relevan, seperti kadar gula darah, 
    tekanan darah, dan usia, aplikasi ini dapat memberikan prediksi yang cukup akurat mengenai kemungkinan seseorang menderita 
    diabetes. Aplikasi ini sangat bermanfaat untuk meningkatkan kesadaran akan risiko diabetes, sehingga dapat mendorong perbaikan 
    pola makan dan gaya hidup untuk pencegahan.''')
    st.markdown("Dataset: [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)")
    
    # Tampilkan contoh data jika file ada
    try:
        df_display = pd.read_csv('diabetes.csv')
        st.subheader("Contoh Data Asli (Sebelum Preprocessing):")
        st.write(df_display.head())
    except FileNotFoundError:
        st.warning("File 'diabetes.csv' tidak ditemukan. Tampilan data contoh tidak tersedia.")

# Halaman Input Manual
elif selected == "Input Manual":
    st.title("Input Manual untuk Prediksi Diabetes")
    
    with st.form("prediction_form"):
        nama = st.text_input('Masukkan Nama', '')
        
        
        pregnancies = st.number_input('Jumlah Kehamilan', min_value=0, max_value=20, value=1)
        glucose = st.number_input('Kadar Glukosa (mg/dL)', min_value=0, max_value=300, value=120)
        bloodpressure = st.number_input('Tekanan Darah (mm Hg)', min_value=0, max_value=150, value=72)
        skinthickness = st.number_input('Ketebalan Kulit (mm)', min_value=0, max_value=100, value=23)
        
        insulin = st.number_input('Kadar Insulin (mu U/ml)', min_value=0, max_value=900, value=30)
        bmi = st.number_input('Indeks Massa Tubuh (BMI)', min_value=0.0, max_value=70.0, value=32.0, format="%.1f")
        dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.47, format="%.3f")
        age = st.number_input('Usia (tahun)', min_value=1, max_value=120, value=22) # Diubah defaultnya agar sesuai screenshot

        submitted = st.form_submit_button("Prediksi")

    if submitted:
        # 1. Feature Engineering: Tambah 'Rentan_Umur' dan Kategori Teks
        # Logika ini disesuaikan agar cocok dengan rentang umur pada umumnya
        if 21 <= age <= 30:
            rentan_umur = 1
            kategori_text = "Dewasa Muda"
        elif 31 <= age <= 40:
            rentan_umur = 2
            kategori_text = "Dewasa"
        elif 41 <= age <= 50:
            rentan_umur = 3
            kategori_text = "Paruh Baya"
        elif 51 <= age <= 60:
            rentan_umur = 4
            kategori_text = "Pra-Lansia"
        else: # Mencakup usia < 21 dan > 60
            rentan_umur = 5
            if age < 21:
                kategori_text = "Remaja"
            else:
                kategori_text = "Lansia"
        
        # 2. Susun data menjadi array 9 fitur
        # Urutan harus sama persis dengan 'feature_columns' dari notebook
        input_data = np.array([[
            pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, dpf, age, rentan_umur
        ]]).reshape(1, -1)
        
        # 3. Normalisasi data menggunakan mean/std dari data training
        data_normalized, _ = custom_standardize(input_data, mean_std_dict_loaded)
        
        # 4. Lakukan Prediksi
        prediction = svm_predict(data_normalized, w_loaded, b_loaded)[0]

        # 5. Tampilkan hasil
        # >>> PERUBAHAN DIMULAI DI SINI <<<
        st.info(f"Berdasarkan usia {age} tahun, Anda termasuk dalam kategori: **{kategori_text}**.")

        if prediction == 0:
            st.success(f"Hai **{nama}**, hasil prediksi menunjukkan Anda **AMAN DARI DIABETES**. Tetap jaga kesehatan ya!")
        else:
            st.error(f"Hai **{nama}**, hasil prediksi menunjukkan Anda **BERISIKO TERKENA DIABETES**. Sebaiknya konsultasikan dengan dokter.")
        # >>> PERUBAHAN SELESAI DI SINI <<<

# Halaman Upload CSV
elif selected == "Upload CSV":
    st.title("Upload File CSV untuk Prediksi Diabetes")
    
    uploaded_file = st.file_uploader("Pilih sebuah file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        # Membuat salinan DataFrame untuk menampilkan hasil akhir
        df_asli = df_upload.copy()

        # Daftar fitur asli yang dibutuhkan untuk prediksi
        original_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
                             'BMI', 'DiabetesPedigreeFunction', 'Age']
        
        # Memastikan semua kolom yang diperlukan ada di file yang diunggah
        if all(col in df_upload.columns for col in original_features):
            # --- PROSES PREDIKSI ---

            # 1. Feature Engineering: Buat kolom 'Rentan_Umur'
            bins = [20, 30, 40, 50, 60, df_upload['Age'].max()]
            labels = [1, 2, 3, 4, 5]
            df_upload['Rentan_Umur'] = pd.cut(df_upload['Age'], bins=bins, labels=labels, right=True, include_lowest=True).astype(int)

            # 2. Siapkan data untuk prediksi (menggunakan 9 fitur yang sesuai dengan model)
            # Pastikan 'feature_columns_loaded' adalah variabel yang berisi 9 nama kolom dari model yang disimpan
            features_data = df_upload[feature_columns_loaded].values
            
            # 3. Lakukan normalisasi menggunakan parameter yang sudah disimpan
            features_norm, _ = custom_standardize(features_data, mean_std_dict_loaded)
            
            # 4. Lakukan prediksi
            predictions = svm_predict(features_norm, w_loaded, b_loaded)

            # --- MENAMPILKAN HASIL ---
            
            # 5. Tambahkan kolom 'Outcome' (hasil prediksi 0 atau 1) dan 'Prediksi' (teks) ke DataFrame asli
            df_asli['Outcome'] = predictions
            # Mengubah teks agar sesuai dengan screenshot Anda
            df_asli['Prediksi'] = ['Pasien terkena diabetes' if p == 1 else 'Pasien tidak terkena diabetes' for p in predictions]
            
            # Menampilkan TabeL HASIL PREDIKSI (sesuai permintaan Anda)
            st.subheader("Hasil Prediksi")
            # Perintah ini akan membuat tabel interaktif seperti pada screenshot Anda
            st.dataframe(df_asli)
            
            # Menampilkan PIE CHART setelah tabel
            st.subheader("Distribusi Hasil Prediksi")
            pie_data = df_asli['Prediksi'].value_counts().reset_index()
            pie_data.columns = ['Status', 'Jumlah']
            fig = px.pie(pie_data, names='Status', values='Jumlah', title='Distribusi Pasien Diabetes vs Tidak Diabetes',
                         # Menyesuaikan nama dan warna agar cocok dengan teks prediksi yang baru
                         color_discrete_map={'Pasien terkena diabetes':'#ef476f', 'Pasien tidak terkena diabetes':'#06d6a0'})
            st.plotly_chart(fig)

        else:
            # Menampilkan error jika ada kolom yang kurang
            st.error(f"Error: Pastikan file CSV Anda memiliki semua kolom berikut: {', '.join(original_features)}")

st.caption("Aplikasi ini dikembangkan untuk tugas Data Mining klasifikasi risiko diabetes menggunakan SVM dan Streamlit.")