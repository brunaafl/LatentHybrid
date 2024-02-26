FROM nvcr.io/nvidia/pytorch:23.09-py3

RUN useradd --uid 1000 -U --create-home --shell /bin/bash bruna

WORKDIR /workspace/project
COPY requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN ["mkdir","-m777", "/.mne"]
RUN ["mkdir","-m777", "/workspace/outputs"]
RUN ["mkdir","-m777", "/workspace/datasets"]

ENV WANDB_API_KEY=e7daeba620fc06b90ca2842da9f15beb57a82456
RUN ["wandb", "login"]