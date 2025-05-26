# Gunakan base image Python yang sesuai. Sesuaikan versi Python dengan yang kamu gunakan saat training.
# Berdasarkan traceback sebelumnya, lingkungan Streamlit Cloud menggunakan Python 3.13.
# Namun, Python 3.13 mungkin terlalu baru untuk stabilitas semua library.
# Jika model dilatih di 3.9/3.10/3.11, coba gunakan itu dulu.
# Kita akan gunakan 3.9 sebagai contoh yang umum stabil.
FROM python:3.9-slim-buster

# Setel variabel lingkungan untuk NLTK data path
# Ini akan menjadi direktori tempat NLTK akan mencari/menyimpan datanya secara default
ENV NLTK_DATA /app/nltk_data

# Setel direktori kerja di dalam container
WORKDIR /app

# Buat direktori untuk data NLTK
# Gunakan -p untuk memastikan direktori induk dibuat jika belum ada
RUN mkdir -p ${NLTK_DATA}

# Salin requirements.txt dan install dependensi Python
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unduh NLTK stopwords ke direktori yang telah ditentukan
# Karena NLTK_DATA sudah disetel, nltk.download() akan otomatis menyimpannya di sana.
# Gunakan -q (quiet) untuk mengurangi output log
RUN python -m nltk.downloader -d ${NLTK_DATA} stopwords

# Salin sisa kode aplikasi Anda ke dalam container
COPY . .

# Perintah untuk menjalankan aplikasi Streamlit
CMD ["streamlit", "run", "main.py"]
