# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# ===============================
# Configuração da Página
# ===============================
st.set_page_config(page_title="Compatibilidade Candidato vs Vaga", layout="wide")

# ===============================
# Diretório base (para caminhos robustos)
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===============================
# CSS - Plano de fundo (aplicado antes de qualquer render)
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
h1, h2, h3, h4, h5, h6, p, label { color: #ffffff !important; }

/* Painel semi-transparente para os containers com borda (catálogo) */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(0, 0, 0, 0.55);
    border-radius: 10px;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===============================
# Helpers
# ===============================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\d+', ' ', text)  # remove números
    text = text.translate(str.maketrans('', '', string.punctuation))  # remove pontuação
    text = re.sub(r'\s+', ' ', text).strip()  # remove espaços extras
    return text


def normalizar_id(serie: pd.Series) -> pd.Series:
    """Garante IDs como string limpa (evita '123.0' quando o pandas lê como float)."""
    return serie.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def coluna_existente(df: pd.DataFrame, opcoes: list):
    """Retorna a primeira coluna existente no DataFrame dentre as opções."""
    for c in opcoes:
        if c in df.columns:
            return c
    return None


def limpar_texto_exibicao(serie: pd.Series) -> pd.Series:
    s = serie.fillna("").astype(str).str.strip()
    return s.replace({"": "Não informado", "nan": "Não informado", "None": "Não informado"})

# ===============================
# Carregar dados CSV
# ===============================
@st.cache_data
def load_data():
    applicants_path = os.path.join(BASE_DIR, "data", "applicants.csv")
    vagas_path = os.path.join(BASE_DIR, "data", "vagas.csv")
    prospects_path = os.path.join(BASE_DIR, "data", "prospects.csv")

    if not os.path.exists(applicants_path) or not os.path.exists(vagas_path):
        st.error("❌ Arquivos 'applicants.csv' ou 'vagas.csv' não encontrados na pasta 'data/'.")
        st.stop()

    applicants = pd.read_csv(applicants_path, low_memory=False)
    try:
        vagas = pd.read_csv(vagas_path, low_memory=False)
    except pd.errors.ParserError:
        vagas = pd.read_csv(vagas_path, sep=';', encoding='latin1', low_memory=False)

    if os.path.exists(prospects_path):
        prospects = pd.read_csv(prospects_path, low_memory=False)
        prospects.columns = prospects.columns.str.strip().str.lower()
    else:
        prospects = pd.DataFrame(columns=["codigo", "titulo"])

    # Normalização de IDs
    applicants["id_candidato"] = normalizar_id(applicants["id_candidato"])
    vagas["ID da Vaga"] = normalizar_id(vagas["ID da Vaga"])
    if "codigo" in prospects.columns:
        prospects["codigo"] = normalizar_id(prospects["codigo"])

    # Pré-processamento de texto
    applicants['texto_completo'] = applicants['cv_pt'].fillna('').apply(preprocess_text)
    vagas['texto_completo'] = (
        vagas['perfil_vaga_principais_atividades'].fillna('') + " " +
        vagas['perfil_vaga_competencia_tecnicas_e_comportamentais'].fillna('')
    ).apply(preprocess_text)

    return applicants, vagas, prospects

applicants, vagas, prospects = load_data()

# ===============================
# Catálogos para exibição (candidatos e vagas)
# ===============================
@st.cache_data
def montar_catalogos(_applicants: pd.DataFrame, _vagas: pd.DataFrame):
    # ---- Candidatos ----
    mapa_cand = {
        "ID do Candidato":       ["id_candidato"],
        "Nome do Candidato":     ["infos_basicas_nome", "informacoes_pessoais_nome"],
        "Formação do Candidato": ["formacao_e_idiomas_nivel_academico",
                                  "formacao_e_idiomas_cursos"],
        "Cidade do Candidato":   ["infos_basicas_local", "informacoes_pessoais_cidade",
                                  "informacoes_pessoais_endereco"],
    }
    cat_cand = pd.DataFrame(index=_applicants.index)
    for nome, opcoes in mapa_cand.items():
        c = coluna_existente(_applicants, opcoes)
        cat_cand[nome] = limpar_texto_exibicao(_applicants[c]) if c else "Não informado"

    # ---- Vagas ----
    mapa_vaga = {
        "ID da Vaga":       ["ID da Vaga", "id_vaga"],
        "Nome da Vaga":     ["informacoes_basicas_titulo_vaga"],
        "Formação Exigida": ["perfil_vaga_nivel_academico", "perfil_vaga_areas_atuacao"],
    }
    cat_vaga = pd.DataFrame(index=_vagas.index)
    for nome, opcoes in mapa_vaga.items():
        c = coluna_existente(_vagas, opcoes)
        cat_vaga[nome] = limpar_texto_exibicao(_vagas[c]) if c else "Não informado"

    # Local da vaga = Cidade / UF (com fallback para local_trabalho)
    col_cidade = coluna_existente(_vagas, ["perfil_vaga_cidade"])
    col_uf = coluna_existente(_vagas, ["perfil_vaga_estado"])
    if col_cidade or col_uf:
        cidade = _vagas[col_cidade].fillna("").astype(str).str.strip() if col_cidade else ""
        uf = _vagas[col_uf].fillna("").astype(str).str.strip() if col_uf else ""
        local = (cidade + " / " + uf).str.strip(" /") if isinstance(cidade, pd.Series) else uf
        cat_vaga["Local da Vaga"] = limpar_texto_exibicao(local)
    else:
        c = coluna_existente(_vagas, ["perfil_vaga_local_trabalho"])
        cat_vaga["Local da Vaga"] = limpar_texto_exibicao(_vagas[c]) if c else "Não informado"

    return cat_cand.reset_index(drop=True), cat_vaga.reset_index(drop=True)

catalogo_candidatos, catalogo_vagas = montar_catalogos(applicants, vagas)


def filtrar_catalogo(df: pd.DataFrame, termo: str, col_formacao: str, formacoes: list):
    out = df
    if formacoes:
        out = out[out[col_formacao].isin(formacoes)]
    if termo:
        mask = out.apply(
            lambda col: col.str.contains(termo, case=False, na=False, regex=False)
        ).any(axis=1)
        out = out[mask]
    return out.reset_index(drop=True)

# ===============================
# Carregar ou treinar TF-IDF
# ===============================
vectorizer_path = os.path.join(BASE_DIR, "model", "vectorizer.pkl")
if os.path.exists(vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
else:
    vectorizer = TfidfVectorizer(max_features=5000)
    vectorizer.fit(applicants['texto_completo'].tolist() + vagas['texto_completo'].tolist())
    os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)

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

# -------------------------------
# Catálogo: Candidatos e Vagas disponíveis
# -------------------------------
with st.container(border=True):
    st.markdown("### 📋 Base disponível para análise")
    st.caption(
        "Pesquise e clique em uma linha para preencher automaticamente os campos de ID "
        "nas abas de análise abaixo."
    )

    col_cand, col_vaga = st.columns(2)

    # ---- Candidatos ----
    with col_cand:
        st.markdown(f"#### 👤 Candidatos ({len(catalogo_candidatos):,})".replace(",", "."))
        f1, f2 = st.columns([3, 2])
        with f1:
            busca_cand = st.text_input("Buscar (nome, cidade, formação ou ID)", key="busca_cand")
        with f2:
            opcoes_form_cand = sorted(catalogo_candidatos["Formação do Candidato"].unique())
            filtro_form_cand = st.multiselect("Formação", opcoes_form_cand, key="filtro_form_cand")

        cand_filtrado = filtrar_catalogo(
            catalogo_candidatos, busca_cand, "Formação do Candidato", filtro_form_cand
        )
        st.caption(f"{len(cand_filtrado)} candidato(s) encontrado(s)")

        ev_cand = st.dataframe(
            cand_filtrado,
            use_container_width=True,
            hide_index=True,
            height=350,
            on_select="rerun",
            selection_mode="single-row",
            key="grid_candidatos",
        )

        linhas = ev_cand.selection.rows
        if linhas and linhas[0] < len(cand_filtrado):
            id_sel = cand_filtrado.iloc[linhas[0]]["ID do Candidato"]
            # Só sobrescreve o input quando a seleção mudar (preserva digitação manual)
            if st.session_state.get("_ultimo_cand_sel") != id_sel:
                st.session_state["_ultimo_cand_sel"] = id_sel
                st.session_state["input_id_candidato"] = id_sel

    # ---- Vagas ----
    with col_vaga:
        st.markdown(f"#### 💼 Vagas ({len(catalogo_vagas):,})".replace(",", "."))
        f3, f4 = st.columns([3, 2])
        with f3:
            busca_vaga = st.text_input("Buscar (título, local, formação ou ID)", key="busca_vaga")
        with f4:
            opcoes_form_vaga = sorted(catalogo_vagas["Formação Exigida"].unique())
            filtro_form_vaga = st.multiselect("Formação exigida", opcoes_form_vaga, key="filtro_form_vaga")

        vaga_filtrada = filtrar_catalogo(
            catalogo_vagas, busca_vaga, "Formação Exigida", filtro_form_vaga
        )
        st.caption(f"{len(vaga_filtrada)} vaga(s) encontrada(s)")

        ev_vaga = st.dataframe(
            vaga_filtrada,
            use_container_width=True,
            hide_index=True,
            height=350,
            on_select="rerun",
            selection_mode="single-row",
            key="grid_vagas",
        )

        linhas = ev_vaga.selection.rows
        if linhas and linhas[0] < len(vaga_filtrada):
            id_sel = vaga_filtrada.iloc[linhas[0]]["ID da Vaga"]
            if st.session_state.get("_ultima_vaga_sel") != id_sel:
                st.session_state["_ultima_vaga_sel"] = id_sel
                st.session_state["input_id_vaga"] = id_sel   # Aba 1
                st.session_state["top5"] = id_sel            # Aba 2

tab1, tab2 = st.tabs(["Análise Individual (Cód. Candidato Vs Cód. Vaga)", "Top 5 Candidatos para Vaga"])

# -------------------------------
# Aba 1: Análise Individual
# -------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        id_candidato = st.text_input("Digite o ID do Candidato:", key="input_id_candidato").strip()
    with col2:
        id_vaga = st.text_input("Digite o ID da Vaga:", key="input_id_vaga").strip()

    if id_candidato and id_vaga:
        candidato = applicants[applicants["id_candidato"] == id_candidato]
        vaga = vagas[vagas["ID da Vaga"] == id_vaga]

        if candidato.empty:
            st.warning(f"⚠️ Candidato '{id_candidato}' não encontrado na base.")
        if vaga.empty:
            st.warning(f"⚠️ Vaga '{id_vaga}' não encontrada na base.")

        if not candidato.empty and not vaga.empty:
            candidato_texto = candidato['texto_completo'].values[0]
            vaga_texto = vaga['texto_completo'].values[0]
            similarity, _ = calcular_similaridade(candidato_texto, vaga_texto)

            st.subheader("📌 Informações do Candidato")
            st.write(f"**Nome:** {candidato['infos_basicas_nome'].values[0]}")
            st.write(f"**Área de Atuação:** {candidato['informacoes_profissionais_area_atuacao'].values[0]}")
            st.write(f"**Nível Acadêmico:** {candidato['formacao_e_idiomas_nivel_academico'].values[0]}")
            st.write(f"**E-mail:** {candidato['informacoes_pessoais_email'].values[0]}")
            st.write(f"**Telefone:** {candidato['informacoes_pessoais_telefone_celular'].values[0]}")

            st.subheader("📄 Texto completo do Candidato")
            st.text_area("CV completo", candidato_texto, height=200)

            st.subheader("📌 Informações da Vaga")
            st.write(f"**Título da Vaga:** {vaga['informacoes_basicas_titulo_vaga'].values[0]}")
            st.write(f"**Cliente:** {vaga['informacoes_basicas_cliente'].values[0]}")
            st.write(f"**Tipo de Contratação:** {vaga['informacoes_basicas_tipo_contratacao'].values[0]}")
            st.write(f"**Prazo:** {vaga['informacoes_basicas_prazo_contratacao'].values[0]}")
            st.write(f"**UF:** {vaga['perfil_vaga_estado'].values[0]}")

            st.subheader("📄 Texto completo da Vaga")
            st.text_area("Descrição completa da vaga", vaga_texto, height=200)

            st.subheader("📊 Compatibilidade")
            st.markdown(f"<h2 style='color:white;'>{similarity*100:.2f}%</h2>", unsafe_allow_html=True)

            st.subheader("📌 Status do Processo Seletivo")
            aprovado = prospects[prospects["codigo"].astype(str) == id_candidato]
            if not aprovado.empty:
                st.success("✅ Candidato aprovado nas seguintes vagas:")
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
    id_vaga_top = st.text_input(
        "Digite o ID da Vaga para buscar os melhores candidatos:", key="top5"
    ).strip()

    if id_vaga_top:
        vaga_top = vagas[vagas["ID da Vaga"] == id_vaga_top]
        if vaga_top.empty:
            st.warning(f"⚠️ Vaga '{id_vaga_top}' não encontrada na base.")
        else:
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
                        if len(resultados) >= 5:
                            break
                return resultados

            top5 = []
            if len(top5) < 5:
                top5 = buscar_top5(applicants.head(1000), 0.70, vaga_texto_top)
            if len(top5) < 5:
                top5 += buscar_top5(applicants.iloc[1000:6000], 0.50, vaga_texto_top)
            if len(top5) < 5:
                top5 += buscar_top5(applicants.iloc[6000:11000], 0.30, vaga_texto_top)

            if top5:
                st.subheader("🏆 Top 5 Candidatos")
                resultados_df = pd.DataFrame(top5[:5])
                resultados_df['compatibilidade'] = resultados_df['compatibilidade'].apply(lambda x: f"{x*100:.2f}%")
                st.dataframe(resultados_df[['id_candidato','nome','compatibilidade','area_atuacao','nivel_academico']], use_container_width=True)
            else:
                st.warning("⚠️ Nenhum candidato encontrado com os critérios definidos.")
