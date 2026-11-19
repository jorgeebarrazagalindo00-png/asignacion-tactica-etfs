# Imagen del panel. El pipeline no corre aquí: lo ejecuta GitHub Actions.
FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# El panel solo lee. Necesita SUPABASE_URL y SUPABASE_PUBLISHABLE_KEY como variables de entorno:
#   docker run -p 8501:8501 --env-file .env etf2
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
