import streamlit as st
import numpy as np
import re
import pandas as pd
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import os

# --- Streamlit Dashboard ---
st.set_page_config(page_title="Prediksi Lowongan Kerja", layout="wide")

# --- Tambahkan ini di awal main.py setelah semua import ---
# Setel NLTK data path secara eksplisit
# Ini akan bekerja di lingkungan Docker karena data diunduh ke /app/nltk_data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data') # Ini akan menjadi /app/nltk_data di container
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)
# --- Akhir penambahan ---

# # Download stopwords if not already downloaded
# # Perubahan di sini: Menggunakan if/else langsung tanpa try-except untuk DownloadError
# try:
#     nltk.data.find('corpora/stopwords')
# except LookupError: # Ganti DownloadError dengan LookupError karena itu yang pertama muncul
#     nltk.download('stopwords')

# --- Fungsi Preprocessing Teks ---
port_stem = PorterStemmer()
pattern = re.compile('[^a-zA-Z]')

def stemming(content):
    stemmed_content = pattern.sub(' ', content).lower().split()
    stopwords_set = set(stopwords.words('english'))
    stemmed_content = ' '.join(word for word in stemmed_content if word not in stopwords_set)
    return stemmed_content

# --- Memuat Model dan Vectorizer yang Sudah Disimpan ---
@st.cache_resource
def load_models():
    models = {}
    vectorizers = {}
    scalers = {}
    label_encoders = {}

    # Load CountVectorizer for n-gram
    with open('count_vectorizer_ngr.pkl', 'rb') as f:
        vectorizers['count_ngr'] = pickle.load(f)

    # Load TF-IDF Vectorizer (for KNN)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizers['tfidf'] = pickle.load(f)

    # Load Scaler for SVM
    with open('scaler_svm.pkl', 'rb') as f:
        scalers['svm'] = pickle.load(f)

    # Load LabelEncoder for XGBoost
    with open('label_encoder_xgb.pkl', 'rb') as f:
        label_encoders['xgb'] = pickle.load(f)

    # Load Models
    with open('nb_classifier.pkl', 'rb') as f:
        models['Naive Bayes'] = pickle.load(f)
    with open('lr_classifier.pkl', 'rb') as f:
        models['Logistic Regression'] = pickle.load(f)
    with open('svm_classifier.pkl', 'rb') as f:
        models['SVM'] = pickle.load(f)
    with open('knn_classifier.pkl', 'rb') as f:
        models['KNN'] = pickle.load(f)
    with open('dt_classifier.pkl', 'rb') as f:
        models['Decision Tree'] = pickle.load(f)
    with open('rf_classifier.pkl', 'rb') as f:
        models['Random Forest'] = pickle.load(f)
    with open('xgb_classifier.pkl', 'rb') as f:
        models['XGBoost'] = pickle.load(f)

    return models, vectorizers, scalers, label_encoders

models, vectorizers, scalers, label_encoders = load_models()



st.title("Fake Job Vacancy Detector")
st.markdown("Fill in the job vacancy details to predict")

# Input Form
with st.form("job_post_form"):
    st.header("Job Vacancy Detail")

    title = st.text_input("Title", help="Example: Senior Software Engineer")
    company_profile = st.text_area("Company Profile", help="Short description of the company")
    description = st.text_area("Job Description", help="Detailed responsibility")
    requirements = st.text_area("Requirements", help="Qualifications")
    benefits = st.text_area("Benefit", help="Salary, insurance, etc")
    industry = st.text_input("Industry", help="Example: Information Technology")
    employment_type = st.selectbox("Job Type", ["Full-time", "Part-time", "Contract", "Temporary", "Other"])
    location = st.text_input("Location", help="Example: US, CA, San Francisco")

    submit_button = st.form_submit_button("Predict")

if submit_button:
    # Menggabungkan teks input
    input_data_raw = {
        'title': title,
        'company_profile': company_profile,
        'description': description,
        'requirements': requirements,
        'benefits': benefits,
        'industry': industry,
        'employment_type': employment_type,
        'location': location
    }

    # Fill NaN values for text columns with empty string (sesuai preprocessing)
    for col in ['title', 'company_profile', 'description', 'requirements', 'benefits', 'industry', 'employment_type', 'location']:
        if input_data_raw[col] is None: # Cek jika user tidak mengisi
            input_data_raw[col] = ''

    # Merge text columns
    merge_text = (
        input_data_raw['title'] + ' ' +
        input_data_raw['company_profile'] + ' ' +
        input_data_raw['description'] + ' ' +
        input_data_raw['requirements'] + ' ' +
        input_data_raw['benefits'] + ' ' +
        input_data_raw['industry'] + ' ' +
        input_data_raw['employment_type'] + ' ' +
        input_data_raw['location']
    )

    # Preprocessing text
    processed_text = stemming(merge_text)

    # Transformasi teks untuk setiap model
    X_input_count_ngr = vectorizers['count_ngr'].transform([processed_text])
    X_input_tfidf = vectorizers['tfidf'].transform([processed_text])

    # Transformasi untuk SVM (perlu scaling)
    X_input_svm_scaled = scalers['svm'].transform(X_input_count_ngr)

    st.subheader("Results:")

    predictions = {}

    # Predict using Naive Bayes
    nb_pred = models['Naive Bayes'].predict(X_input_count_ngr)[0]
    predictions['Naive Bayes'] = nb_pred

    # Predict using Logistic Regression
    lr_pred = models['Logistic Regression'].predict(X_input_count_ngr)[0]
    predictions['Logistic Regression'] = lr_pred

    # Predict using SVM
    svm_pred = models['SVM'].predict(X_input_svm_scaled)[0]
    predictions['SVM'] = svm_pred

    # Predict using KNN
    knn_pred = models['KNN'].predict(X_input_tfidf)[0]
    predictions['KNN'] = knn_pred

    # Predict using Decision Tree
    dt_pred = models['Decision Tree'].predict(X_input_count_ngr)[0]
    predictions['Decision Tree'] = dt_pred

    # Predict using Random Forest
    rf_pred = models['Random Forest'].predict(X_input_count_ngr)[0]
    predictions['Random Forest'] = rf_pred

    # Predict using XGBoost
    # XGBoost membutuhkan DMatrix dan label encoding
    xgb_pred_encoded = models['XGBoost'].predict(xgb.DMatrix(X_input_svm_scaled)) # Gunakan X_input_svm_scaled karena itu yang digunakan saat training
    xgb_pred = label_encoders['xgb'].inverse_transform([int(xgb_pred_encoded[0])])[0]
    predictions['XGBoost'] = xgb_pred

    # Display predictions
    for model_name, prediction in predictions.items():
        if prediction == 'fake':
            st.error(f"**{model_name}: This job posting is LIKELY FAKE**")
        else:
            st.success(f"**{model_name}: This job posting is LIKELY GENUINE**")

    st.markdown("---")
    st.write("Note: These predictions are generated by Machine Learning models. Always perform additional verification.")
