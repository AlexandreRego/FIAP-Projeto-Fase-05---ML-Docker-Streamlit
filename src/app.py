# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import numpy as np

# ===============================
# Configuração da Página
# ===============================
st.set_page_config(page_title="Compatibilidade Candidato vs Vaga", layout="wide")

# ===============================
# Diretório base (para caminhos robustos)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# Carregar dados CSV
# ===============================
@st.cache_data
def load_data():
    applicants_path = os.path.join(BASE_DIR, "data", "applicants.csv")
    vagas_path = os.path.join(BASE_DIR, "data", "vagas.csv")
    prospects_path = os.path.join(BASE_DIR, "data", "prospects.csv")

    # Verifique se os arquivos existem antes de tentar carregá-los
    if not os.path.exists(applicants_path) or not os.path.exists(vagas_path):
        st.error("❌ Arquivos de dados 'applicants.csv' ou 'vagas.csv' não encontrados na pasta 'data/'.")
        st.stop()

    # Ler applicants
    applicants = pd.read_csv(applicants_path, low_memory=False)

    # Ler vagas (com tratamento de possíveis delimitadores e encoding)
    try:
        vagas = pd.read_csv(vagas_path, low_memory=False)
    except pd.errors.ParserError:
        vagas = pd.read_csv(vagas_path, sep=';', encoding='latin1', low_memory=False)

    # Ler prospects
    if os.path.exists(prospects_path):
        prospects = pd.read_csv(prospects_path, low_memory=False)
        prospects.columns = prospects.columns.str.strip().str.lower()
    else:
        prospects = pd.DataFrame(columns=["codigo", "titulo"])

    return applicants, vagas, prospects

applicants, vagas, prospects = load_data()

# Preparar texto completo
applicants['texto_completo'] = applicants['cv_pt'].fillna('')
vagas['texto_completo'] = vagas['perfil_vaga_principais_atividades'].fillna('') + " " + \
                         vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')

# ===============================
# Carregar TF-IDF salvo
# ===============================
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    st.error("❌ TF-IDF não encontrado em 'model/vectorizer.pkl'. Execute train_model.py primeiro.")
    st.stop()

# ===============================
# Funções de Similaridade
# ===============================
def calcular_similaridade(candidato_texto, vaga_texto):
    tfidf_matrix = vectorizer.transform([candidato_texto, vaga_texto])
    similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return similarity, tfidf_matrix

def get_top_keywords(tfidf_matrix, vectorizer, top_n=10):
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_sorting = np.argsort(tfidf_matrix.toarray()).flatten()[::-1]
    top_n_terms = feature_array[tfidf_sorting][:top_n]
    return top_n_terms

# ===============================
# Layout do App
# ===============================
st.title("🔎 Análise de Compatibilidade")

tab1, tab2 = st.tabs(["Análise Individual (Cód. Candidato Vs Cód. Vaga)", "Top 5 Candidatos para Vaga"])

# ===============================
# CSS - Plano de fundo
# ===============================
page_bg = """
<style>
[data-testid="stAppViewContainer"] { 
    background-image: url("https://www.itagroup.com/filesimages/Insights/White%20Papers/Channel_Channel%20Partner%20Ecosystems/6.%20Retention%20Channel/Insight-Channel-Ecosystem-Retention-WP-Primary-Image.jpg"); 
    background-size: cover; 
    background-position: center; 
    background-repeat: no-repeat; 
}
[data-testid="stHeader"] { background: rgba(0,0,0,0.5); }
[data-testid="stSidebar"] { background: rgba(255,255,255,0.8); }
h1, h2, h3, h4, h5, h6, p { color: #ffffff !important; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------------------
# Aba 1: Análise Individual
# -------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        id_candidato = st.text_input("Digite o ID do Candidato:")
    with col2:
        id_vaga = st.text_input("Digite o ID da Vaga:")

    if id_candidato and id_vaga:
        candidato = applicants[applicants["id_candidato"].astype(str) == id_candidato]
        vaga = vagas[vagas["ID da Vaga"].astype(str) == id_vaga]

        if not candidato.empty and not vaga.empty:
            candidato_texto = candidato['texto_completo'].values[0]
            vaga_texto = vaga['texto_completo'].values[0]
            similarity, _ = calcular_similaridade(candidato_texto, vaga_texto)

            # Informações do Candidato
            st.subheader("📌 Informações do Candidato")
            st.write(f"**Nome:** {candidato['infos_basicas_nome'].values[0]}")
            st.write(f"**Área de Atuação:** {candidato['informacoes_profissionais_area_atuacao'].values[0]}")
            st.write(f"**Nível Acadêmico:** {candidato['formacao_e_idiomas_nivel_academico'].values[0]}")
            st.write(f"**E-mail:** {candidato['informacoes_pessoais_email'].values[0]}")
            st.write(f"**Telefone:** {candidato['informacoes_pessoais_telefone_celular'].values[0]}")

            st.subheader("📄 Texto completo do Candidato")
            st.text_area("CV completo", candidato_texto, height=200)

            # Informações da Vaga
            st.subheader("📌 Informações da Vaga")
            st.write(f"**Título da Vaga:** {vaga['informacoes_basicas_titulo_vaga'].values[0]}")
            st.write(f"**Cliente:** {vaga['informacoes_basicas_cliente'].values[0]}")
            st.write(f"**Tipo de Contratação:** {vaga['informacoes_basicas_tipo_contratacao'].values[0]}")
            st.write(f"**Prazo:** {vaga['informacoes_basicas_prazo_contratacao'].values[0]}")
            st.write(f"**UF:** {vaga['perfil_vaga_estado'].values[0]}")

            st.subheader("📄 Texto completo da Vaga")
            st.text_area("Descrição completa da vaga", vaga_texto, height=200)

            # Compatibilidade %
            st.subheader("📊 Compatibilidade")
            st.markdown(f"<h2 style='color:white;'>{similarity*100:.2f}%</h2>", unsafe_allow_html=True)

            # Status de aprovação no prospects
            st.subheader("📌 Status do Processo Seletivo")
            aprovado = prospects[prospects["codigo"].astype(str) == id_candidato]

            if not aprovado.empty:
                st.success(f"✅ Candidato aprovado nas seguintes vagas:")
                for _, row in aprovado.iterrows():
                    titulo_vaga_aprov = row["titulo"]
                    vaga_info = vagas[vagas["informacoes_basicas_titulo_vaga"].astype(str) == str(titulo_vaga_aprov)]
                    if not vaga_info.empty:
                        id_vaga_aprov = vaga_info["ID da Vaga"].values[0]
                        st.success(f" - {id_vaga_aprov} - {titulo_vaga_aprov}")
                    else:
                        st.success(f" - (ID da vaga não encontrado) - {titulo_vaga_aprov}")
            else:
                st.error("❌ Não aprovado ou não seguiu na etapa de seleção")

# -------------------------------
# Aba 2: Top 5 Candidatos
# -------------------------------
with tab2:
    id_vaga_top = st.text_input("Digite o ID da Vaga para buscar os melhores candidatos:", key="top5")

    if id_vaga_top:
        vaga_top = vagas[vagas["ID da Vaga"].astype(str) == id_vaga_top]
        if not vaga_top.empty:
            vaga_texto_top = vaga_top['texto_completo'].values[0]
            st.success(f"✅ Vaga encontrada: {vaga_top['informacoes_basicas_titulo_vaga'].values[0]}")

            def buscar_top5(applicants_slice, threshold, vaga_texto):
                resultados = []
                for _, candidato in applicants_slice.iterrows():
                    similarity, tfidf_matrix = calcular_similaridade(
                        candidato['texto_completo'], vaga_texto
                    )
                    if similarity >= threshold:
                        resultados.append({
                            'id_candidato': candidato['id_candidato'],
                            'nome': candidato['infos_basicas_nome'],
                            'compatibilidade': similarity,
                            'area_atuacao': candidato['informacoes_profissionais_area_atuacao'],
                            'nivel_academico': candidato['formacao_e_idiomas_nivel_academico'],
                            'email': candidato['informacoes_pessoais_email'],
                            'telefone': candidato['informacoes_pessoais_telefone_celular'],
                            'keywords': get_top_keywords(tfidf_matrix, vectorizer, 5)
                        })
                        if len(resultados) >= 5:  # já encontrou 5
                            break
                return resultados

            # Estratégia em blocos
            top5 = []
            if len(top5) < 5:
                top5 = buscar_top5(applicants.head(1000), 0.70, vaga_texto_top)

            if len(top5) < 5:
                top5 += buscar_top5(applicants.iloc[1000:6000], 0.50, vaga_texto_top)

            if len(top5) < 5:
                top5 += buscar_top5(applicants.iloc[6000:11000], 0.30, vaga_texto_top)

            if top5:
                st.subheader("🏆 Top 5 Candidatos")
                resultados_df = pd.DataFrame(top5[:5])  # garante só 5
                resultados_df['compatibilidade'] = resultados_df['compatibilidade'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(resultados_df[['id_candidato','nome','compatibilidade','area_atuacao','nivel_academico']], use_container_width=True)
            else:
                st.warning("⚠️ Nenhum candidato encontrado com os critérios definidos.")
