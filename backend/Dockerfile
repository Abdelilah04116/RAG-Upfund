FROM python:3.10-slim

WORKDIR /app

# Mise à jour des packages système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Installation de PyTorch CPU d'abord (plus stable)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .

# Installation des autres packages
RUN pip install --no-cache-dir --timeout=1000 --retries=3 -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
