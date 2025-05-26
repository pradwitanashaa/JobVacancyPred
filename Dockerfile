# Gunakan base image Python yang sesuai. Pastikan versi ini cocok dengan lingkungan training Anda.
# Jika Anda yakin model dilatih dengan Python 3.9, gunakan 3.9. Jika di 3.13, gunakan 3.13 (namun 3.13 masih sangat baru).
# Mari kita asumsikan 3.9 untuk kompatibilitas yang lebih luas.
FROM python:3.9-slim-buster

# Setel direktori kerja di dalam container
WORKDIR /app

# Buat direktori untuk data NLTK
# Gunakan -p untuk memastikan direktori induk dibuat jika belum ada
RUN mkdir -p /app/nltk_data

# Salin requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unduh NLTK stopwords dan punkt (terkadang stopwords bergantung pada punkt)
# Pastikan ini disimpan ke lokasi yang sama dengan yang diharapkan main.py
# NLTK akan mencari di nltk.data.path, yang akan kita set di main.py.
# Tapi kita juga bisa menyetel ENV NLTK_DATA sebagai fallback/default.
ENV NLTK_DATA /app/nltk_data
RUN python -m nltk.downloader -d ${NLTK_DATA} stopwords punkt

# Salin sisa kode aplikasi Anda ke dalam container
COPY . .

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "main.py"]
