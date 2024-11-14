FROM apache/airflow:2.8.0
USER root

COPY requirements.txt /
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        freetds-dev \
        libkrb5-dev \
        libssl-dev \
        libffi-dev \
        python3-dev \
        iputils-ping \
        vim \
        nano \
        xvfb \
        libxi6 \
        libgconf-2-4 \
        postgresql-client && \
    rm -rf /var/lib/apt/lists/*

USER airflow
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt