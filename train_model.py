# train_model_robusto_pt.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
import re
import unicodedata
import spacy

# ===============================
# Inicializar spaCy
# ===============================
try:
    nlp = spacy.load("pt_core_news_sm")
except OSError:
    print("❌ Modelo spaCy 'pt_core_news_sm' não encontrado. Rode: python -m spacy download pt_core_news_sm")
    exit(1)

# ===============================
# Funções auxiliares
# ===============================
def clean_text(text):
    """Remove acentos, caracteres especiais e normaliza texto."""
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text):
    """Aplica lemmatização e remove stopwords."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

def preprocess_text(text):
    """Limpeza + lemmatização."""
    text = clean_text(text)
    text = lemmatize_text(text)
    return text

def carregar_csv(caminho, colunas_obrigatorias=None):
    """Carrega CSV com validação de colunas."""
    if not os.path.exists(caminho):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
    df = pd.read_csv(caminho, low_memory=False)
    if colunas_obrigatorias:
        faltando = [c for c in colunas_obrigatorias if c not in df.columns]
        if faltando:
            raise ValueError(f"Colunas obrigatórias ausentes em {caminho}: {faltando}")
    return df

# ===============================
# Carregar dados
# ===============================
try:
    applicants = carregar_csv("applicants.csv", colunas_obrigatorias=['cv_pt'])
    vagas = carregar_csv("vagas.csv", colunas_obrigatorias=[
        'perfil_vaga_principais_atividades',
        'perfil_vaga_competencia_tecnicas_e_comportamentais'
    ])
except Exception as e:
    print(f"❌ Erro ao carregar arquivos: {e}")
    exit(1)

# ===============================
# Preparar textos
# ===============================
print("⏳ Processando textos dos candidatos...")
applicants['texto_completo'] = applicants['cv_pt'].fillna('').apply(preprocess_text)

print("⏳ Processando textos das vagas...")
vagas['texto_completo'] = (
    vagas['perfil_vaga_principais_atividades'].fillna('') + " " +
    vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')
).apply(preprocess_text)

# ===============================
# Combinar todos os textos
# ===============================
todos_textos = pd.concat([applicants['texto_completo'], vagas['texto_completo']], ignore_index=True)

# ===============================
# Criar TF-IDF
# ===============================
try:
    vectorizer = TfidfVectorizer(
        ngram_range=(1,2),
        min_df=2,           # ignora termos muito raros
        max_df=0.9,         # ignora termos muito comuns
        max_features=5000
    )
    vectorizer.fit(todos_textos)
except Exception as e:
    print(f"❌ Erro ao treinar TF-IDF: {e}")
    exit(1)

# ===============================
# Salvar TF-IDF
# ===============================
os.makedirs("model", exist_ok=True)
joblib.dump(vectorizer, "model/vectorizer.pkl")

print("✅ TF-IDF robusto salvo com sucesso em /model/vectorizer.pkl")
