# Utiliser une image de base appropriée pour Streamlit
FROM python:3.10

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Définir la commande par défaut pour exécuter l'application Streamlit
CMD ["streamlit", "run", "app.py"]
