# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Copia só as dependências
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Cria usuário com UID não-root
RUN adduser --uid 1001 --disabled-password --gecos '' appuser && chown -R 1001 /app

# Copia o restante do app
COPY . .

# Usa UID explicitamente
USER 1001

EXPOSE 8080

CMD ["python", "app.py"]
