
FROM python:3.14-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && useradd --create-home appuser

COPY --from=builder /install /usr/local

RUN rm -rf /usr/local/lib/python3.14/site-packages/pip \
           /usr/local/lib/python3.14/site-packages/pip-*.dist-info \
           /usr/local/bin/pip \
           /usr/local/bin/pip3 \
           /usr/local/bin/pip3.14 \
           /usr/local/lib/python3.14/site-packages/setuptools \
           /usr/local/lib/python3.14/site-packages/setuptools-*.dist-info \
           /usr/local/lib/python3.14/site-packages/_distutils_hack \
           /usr/local/lib/python3.14/site-packages/pkg_resources

COPY src/ ./src/
COPY model.joblib .

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "src.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
