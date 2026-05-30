FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN useradd --uid 1007 -U --create-home --shell /bin/bash bruna

WORKDIR /workspace/project
RUN pip3 install --upgrade pip

# Pre-install CPU torchaudio to sidestep the CUDA 13.0 vs 12.8 binary conflict
COPY requirements.txt .
RUN pip3 install -r requirements.txt 
RUN ["mkdir","-m777", "/.mne"]
RUN ["mkdir","-m777", "/workspace/outputs"]
RUN ["mkdir","-m777", "/workspace/datasets"]
RUN ["mkdir","-m777", "/workspace/models"]

ENV WANDB_API_KEY=e7daeba620fc06b90ca2842da9f15beb57a82456
RUN ["wandb", "login"]
