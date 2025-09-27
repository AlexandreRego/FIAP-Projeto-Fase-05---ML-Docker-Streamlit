# -*- coding: utf-8 -*-
import pandas as pd
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# ===============================
# Diretório base
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# Função de pré-processamento leve
# ===============================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)  # remove números
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # remove espaços extras
    return text

# ===============================
# Carregar dados CSV
# ===============================
applicants_path = os.path.join(BASE_DIR, "data", "applicants.csv")
vagas_path = os.path.join(BASE_DIR, "data", "vagas.csv")

if not os.path.exists(applicants_path) or not os.path.exists(vagas_path):
    raise FileNotFoundError("❌ Arquivos 'applicants.csv' ou 'vagas.csv' não encontrados na pasta 'data/'.")

applicants = pd.read_csv(applicants_path, low_memory=False)
vagas = pd.read_csv(vagas_path, low_memory=False)

# Pré-processamento dos textos
applicants['texto_completo'] = applicants['cv_pt'].fillna('').apply(preprocess_text)
vagas['texto_completo'] = (
    vagas['perfil_vaga_principais_atividades'].fillna('') + " " +
    vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')
).apply(preprocess_text)

# ===============================
# Treinar TF-IDF
# ===============================
vectorizer = TfidfVectorizer(max_features=5000)
vectorizer.fit(applicants['texto_completo'].tolist() + vagas['texto_completo'].tolist())

# ===============================
# Salvar TF-IDF
# ===============================
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
joblib.dump(vectorizer, vectorizer_path)

print("✅ Vectorizer treinado e salvo em 'model/vectorizer.pkl'")
