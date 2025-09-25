# Use uma imagem base do Python, idealmente uma versão específica
FROM python:3.10

# Defina o diretório de trabalho dentro do container
WORKDIR /app

# Copie o arquivo de requisitos para o container
COPY requirements.txt .

# Instale as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o seu projeto para o diretório de trabalho do container
COPY . .

# Expõe a porta que o Streamlit usa por padrão
EXPOSE 8501

# Comando para rodar a aplicação quando o container iniciar
CMD ["streamlit", "run", "app.py"]