# Compatibilidade Candidato vs Vaga 🔎
## Descrição do Projeto 📌

Este projeto tem como objetivo analisar a compatibilidade entre candidatos e vagas de emprego utilizando técnicas de Processamento de Linguagem Natural (NLP) e similaridade de textos.

## O sistema permite:

1 - Comparar um candidato específico com uma vaga escolhida;

2 - Exibir informações detalhadas do candidato e da vaga;

3 - Calcular o percentual de compatibilidade com base no currículo (CV) e descrição da vaga;

4 - Listar o Top 10 candidatos mais compatíveis para uma vaga;

5 - Visualizar palavras-chave mais relevantes na compatibilidade (interpretabilidade).

O projeto conta com uma pipeline de pré-processamento para tratar textos em português, utilizando spaCy e TF-IDF para transformar descrições em vetores comparáveis.

___________________________________________________________________________________________________________________________

# Stack Utilizada 🛠️

Linguagem: Python 3.10+

Framework Web: Streamlit

NLP: spaCy (pt_core_news_sm)

Machine Learning: scikit-learn (TF-IDF + Similaridade do Cosseno)

Serialização: Joblib

Manipulação de Dados: Pandas, NumPy

Visualização: Matplotlib

Ambiente Virtual: venv

___________________________________________________________________________________________________________________________

# Estrutura do Projeto ⚙️

<img width="620" height="272" alt="image" src="https://github.com/user-attachments/assets/c6607854-0be2-4dab-a481-0ec085a28eb5" />

___________________________________________________________________________________________________________________________

# Como Rodar o App Localmente ▶️

## Clone o repositório:

git clone https://github.com/seuusuario/seurepositorio.git
cd seurepositorio

## Crie um ambiente virtual:

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

## Instale as dependências:

pip install -r requirements.txt

## Baixe o modelo spaCy em português:

python -m spacy download pt_core_news_sm

## Treine o modelo (apenas na primeira vez):

python train_model_robusto_pt.py

Isso irá gerar o arquivo model/vectorizer.pkl.

## Execute a aplicação Streamlit:

streamlit run app.py

## Acesse o app no navegador:

http://localhost:8501

# Como Treinar o Modelo Novamente 📚

Caso novas bases de candidatos ou vagas sejam adicionadas, o modelo deve ser re-treinado:

## Atualize os arquivos applicants.csv e vagas.csv.

Rode novamente o script de treinamento:

python train_model_robusto_pt.py

## O novo modelo TF-IDF será salvo em:

/model/vectorizer.pkl

__________________________________________________________________________________________________________________________

# Justificativa da Escolha do Modelo 📊

### TF-IDF (Term Frequency – Inverse Document Frequency):
Escolhido para transformar textos em vetores numéricos que refletem a importância de termos dentro do contexto de candidatos e vagas.

### Similaridade do Cosseno:
Métrica eficiente para comparar a proximidade semântica entre o currículo de um candidato e a descrição de uma vaga.

### Interpretabilidade:
O sistema retorna as palavras-chave mais relevantes que explicam a compatibilidade, permitindo compreender o porquê de um candidato estar bem (ou mal) ranqueado.

### Métricas de Validação:

Distribuição de similaridades avaliada para garantir separação entre perfis compatíveis e não compatíveis;

Ajuste de parâmetros do TF-IDF (ngramas, min_df, max_df, max_features) para equilibrar performance e robustez;

Pipeline de pré-processamento em português com spaCy, incluindo lematização e remoção de stopwords.

Essa abordagem garante um modelo leve, interpretável e eficiente, adequado para sistemas de RH que necessitam de explicabilidade.

__________________________________________________________________________________________________________________________

# Tratamento das Bases 📖

O notebook Tratamentos_Bases_Fase_05.ipynb contém:

Limpeza e padronização das bases de candidatos e vagas;

Normalização de colunas e tratamento de valores ausentes;

Exportação final dos datasets para o app e treinamento.
