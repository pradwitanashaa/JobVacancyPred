# Gunakan base image Streamlit yang sudah ada
FROM python:3.12-slim-buster # Versi Python yang sesuai, pastikan ini kompatibel dengan training Anda (misal 3.9)

# Setel direktori kerja di dalam container
WORKDIR /app

# Salin requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Buat direktori untuk data NLTK
RUN mkdir -p /app/nltk_data

# Unduh NLTK stopwords ke direktori yang baru dibuat
# Pastikan Python dan NLTK sudah terinstal
RUN python -c "import nltk; import os; nltk.data.path.append('/app/nltk_data'); nltk.download('stopwords', download_dir='/app/nltk_data')"

# Salin sisa kode aplikasi Anda
COPY . .

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "main.py"]
