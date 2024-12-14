FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

USER root

ARG DEBIAN_FRONTEND=noninteractive

LABEL github_repo="https://github.com/moewiee/f5tts"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /workspace

RUN git clone https://github.com/moewiee/f5tts.git \
    && cd f5tts \
    && pip install -e .[eval]

ENV SHELL=/bin/bash

WORKDIR /workspace/f5tts

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    mecab \
    libmecab-dev \
    mecab-ipadic-utf8 && \
    rm -rf /var/lib/apt/lists/*

# Install the mecab-python3 package for Python
RUN pip install mecab-python3

# docker run -it --gpus all -v $(pwd)/checkpoints:/workspace/f5tts/checkpoints f5tts /bin/bash