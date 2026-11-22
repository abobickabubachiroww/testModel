FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY api.py ./
COPY data/raw/manufacturing_defect_dataset.csv ./data/raw/manufacturing_defect_dataset.csv
COPY src/models/saved/qualitypredict_final.joblib ./src/models/saved/qualitypredict_final.joblib
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
