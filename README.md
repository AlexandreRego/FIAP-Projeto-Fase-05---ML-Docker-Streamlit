# Compatibilidade Candidato vs Vaga 🔎
## Descrição do Projeto 📌

## Este projeto permite analisar a compatibilidade entre candidatos e vagas de emprego utilizando técnicas de Processamento de Linguagem Natural (NLP) e similaridade de textos.

### O sistema permite:

Comparar um candidato específico com uma vaga escolhida;

Exibir informações detalhadas do candidato e da vaga;

Calcular o percentual de compatibilidade com base no currículo (CV) e descrição da vaga;

Listar o Top 5 candidatos mais compatíveis para uma vaga;

Visualizar palavras-chave mais relevantes na compatibilidade (interpretabilidade).

O projeto utiliza uma pipeline de pré-processamento leve para tratar textos em português, transformando descrições em vetores comparáveis com TF-IDF.

___________________________________________________________________________________________________________________________

# Stack Utilizada 🛠️

Linguagem: Python 3.10+

Framework Web: Streamlit

Machine Learning: scikit-learn (TF-IDF + Similaridade do Cosseno)

Serialização: Joblib

Manipulação de Dados: Pandas, NumPy

Visualização: Matplotlib

Ambiente Virtual: venv

___________________________________________________________________________________________________________________________

# Estrutura do Projeto ⚙️

project/
│
├─ data/
│  ├─ applicants.csv
│  ├─ vagas.csv
│  └─ prospects.csv
│
├─ model/
│  └─ vectorizer.pkl   # gerado após treinar o modelo
│
├─ app.py
├─ train_model.py
├─ requirements.txt
└─ README.md


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

## Treine o modelo (apenas na primeira vez):

python train_model.py

Isso irá gerar o arquivo model/vectorizer.pkl.

## Execute a aplicação Streamlit:

streamlit run app.py

## Acesse o app no navegador:

http://localhost:8501

# Como Treinar o Modelo Novamente 📚

Caso novas bases de candidatos ou vagas sejam adicionadas, o modelo deve ser re-treinado:

## Atualize os arquivos applicants.csv e vagas.csv.

Rode novamente o script de treinamento:

python train_model.py

## O novo modelo TF-IDF será salvo em:

/model/vectorizer.pkl

__________________________________________________________________________________________________________________________

# Justificativa da Escolha do Modelo 📊

## TF-IDF (Term Frequency – Inverse Document Frequency):
Transforma textos em vetores numéricos refletindo a importância de cada termo dentro do contexto de candidatos e vagas.

### Similaridade do Cosseno:
Compara a proximidade semântica entre o currículo de um candidato e a descrição da vaga de forma eficiente.

### Interpretabilidade:
O sistema retorna as palavras-chave mais relevantes que explicam a compatibilidade, permitindo entender o porquê de um candidato estar bem (ou mal) ranqueado.

### Pipeline leve:
Pré-processamento em português usando funções simples (minúsculas, remoção de números e pontuação, remoção de espaços extras), sem SpaCy, garantindo leveza e desempenho no Streamlit Cloud.

__________________________________________________________________________________________________________________________

# Tratamento das Bases 📖

O notebook Tratamentos_Bases_Fase_05.ipynb contém:

Limpeza e padronização das bases de candidatos e vagas;

Normalização de colunas e tratamento de valores ausentes;

Exportação final dos datasets para o app e treinamento.
