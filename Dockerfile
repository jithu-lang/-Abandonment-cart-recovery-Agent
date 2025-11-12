# ============================
# 1. Builder Stage (install deps)
# ============================
FROM python:3.11-slim as builder

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y build-essential \
    && pip install --upgrade pip \
    && pip install --prefix=/install -r requirements.txt

# ============================
# 2. Final Slim Image
# ============================
FROM python:3.11-slim

WORKDIR /app

# Copy installed dependencies from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . .

EXPOSE 8000
EXPOSE 8501

CMD uvicorn api:app --host 0.0.0.0 --port 8000 & \
    streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
