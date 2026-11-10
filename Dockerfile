FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app src/fifa_churn ./src/fifa_churn
COPY --chown=app:app models/churn_proxy_model.joblib ./models/churn_proxy_model.joblib

ENV MODEL_PATH=/app/models/churn_proxy_model.joblib
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "fifa_churn.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
