FROM nvidia/cuda:12.6.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHON_VERSION=3.12.11

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    ca-certificates \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    tk-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j$(nproc) && \
    make altinstall && \
    cd / && \
    rm -rf /tmp/Python-${PYTHON_VERSION}*

RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/local/bin/python3.12 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip3

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN apt-get update && apt-get install -y postgresql-client

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt