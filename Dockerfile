FROM python:3.12-slim
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY config.yaml ./
EXPOSE 8501
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "src/app.py"]
