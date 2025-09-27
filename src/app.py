# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import unicodedata
import nltk

# ===============================
# Download recursos NLTK
# ===============================
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

stop_words = set(stopwords.words("portuguese"))
stemmer = RSLPStemmer()

# ===============================
# Configuração da Página
# ===============================
st.set_page_config(page_title="Compatibilidade Candidato vs Vaga", layout="wide")

# ===============================
# Funções de pré-processamento
# ===============================
def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Minúsculas
    text = text.lower()
    # Remove acentos
    text = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
    # Remove caracteres especiais
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Remove múltiplos espaços
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_text(text):
    text = clean_text(text)
    tokens = [
        stemmer.stem(token)
        for token in text.split()
        if token not in stop_words and len(token) > 2
    ]
    return " ".join(tokens)

# ===============================
# Carregar arquivos e modelo
# ===============================
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")

try:
    applicants = pd.read_excel(os.path.join(DATA_DIR, "applicants.xlsx"))
    vagas = pd.read_excel(os.path.join(DATA_DIR, "vagas.xlsx"))
except FileNotFoundError:
    st.error("❌ Arquivos de dados 'applicants.xlsx' ou 'vagas.xlsx' não encontrados na pasta 'data/'.")
    st.stop()

try:
    vectorizer = joblib.load(os.path.join(DATA_DIR, "vectorizer.pkl"))
except FileNotFoundError:
    st.error("❌ Arquivo 'vectorizer.pkl' não encontrado na pasta 'data/'.")
    st.stop()

# ===============================
# Aplicar pré-processamento
# ===============================
applicants["texto_completo"] = applicants["cv_pt"].fillna("").apply(preprocess_text)

vagas["texto_completo"] = (
    vagas["perfil_vaga_principais_atividades"].fillna("") + " " +
    vagas["perfil_vaga_competencia_tecnicas_e_comportamentais"].fillna("")
).apply(preprocess_text)

# ===============================
# Vetorização
# ===============================
applicants_matrix = vectorizer.transform(applicants["texto_completo"])
vagas_matrix = vectorizer.transform(vagas["texto_completo"])

# ===============================
# Interface
# ===============================
st.title("🔎 Compatibilidade entre Candidatos e Vagas")

vaga_escolhida = st.selectbox("Selecione uma vaga:", vagas["titulo_vaga"].tolist())

if vaga_escolhida:
    vaga_idx = vagas[vagas["titulo_vaga"] == vaga_escolhida].index[0]
    vaga_vector = vagas_matrix[vaga_idx]

    # Similaridade
    similarities = cosine_similarity(vaga_vector, applicants_matrix).flatten()
    applicants["similaridade"] = similarities

    # Top 5
    top_applicants = applicants.sort_values(by="similaridade", ascending=False).head(5)

    st.subheader("👥 Top 5 Candidatos Compatíveis")
    for _, row in top_applicants.iterrows():
        st.markdown(f"**{row['nome']}** — Similaridade: {row['similaridade']:.2%}")

    # Gráfico
    st.subheader("📊 Distribuição das Similaridades")
    fig, ax = plt.subplots()
    ax.hist(similarities, bins=20, color="blue", alpha=0.7)
    ax.set_title("Distribuição de Similaridade entre Candidatos e a Vaga")
    ax.set_xlabel("Similaridade")
    ax.set_ylabel("Quantidade de Candidatos")
    st.pyplot(fig)

