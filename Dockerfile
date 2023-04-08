FROM python:3.8-slim

WORKDIR /app

COPY Pipfile Pipfile.lock /app/

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    pipenv \
    && rm -rf /var/lib/apt/lists/*

RUN pipenv install --system --deploy

COPY ./ .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["/app/run.sh"]
#[pipenv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
