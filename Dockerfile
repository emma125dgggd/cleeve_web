FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

COPY ./ .

EXPOSE 8000

HEALTHCHECK CMD curl --fail http://localhost:8000/health

ENTRYPOINT ["/app/run.sh"]

#[pipenv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
