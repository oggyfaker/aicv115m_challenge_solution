# Build a docker that run the service for predicting Covid result
# author: Tony Do 
# 2rd Sep 2021 
# Edited by: FruitAI team

ARG CUDA_VERSION=10.1
ARG PYTHON_VERSION=3.6

# base image - require nvidia GPU installed in host PC to  run this Docker
FROM nvidia/cuda:10.1-base
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.6

#CMD nvidia-smi # test GPU connected with Docker, uncommand for testing
# set the working directory
ENV HOME .
WORKDIR $HOME
COPY requirements.txt ./
CMD mkdir modules/pytorch-image-models
COPY ./modules/pytorch-image-models/ $HOME/modules/pytorch-image-models
RUN apt update && apt-get install -y --no-install-recommends apt-utils && \
	apt-get install dialog -y && \
	apt-get -y install gcc && \
	apt install ffmpeg libsndfile1 --yes --no-install-recommends && \
	apt-get install curl --yes --no-install-recommends &&\
    pip install --upgrade pip \
	&& pip install setuptools 

RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p conda && \
    rm miniconda.sh && \
    conda/bin/conda install -y python=$PYTHON_VERSION jupyter jupyterlab && \
	conda/bin/pip install torch==1.7.0+cu101 torchvision==0.8.1+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html &&\
	conda/bin/pip install -r requirements.txt && \
	conda/bin/pip install $HOME/modules/pytorch-image-models/. && \
	conda/bin/conda clean -ya

ENV PATH $HOME/conda/bin:$PATH
RUN touch $HOME/.bashrc && \
    echo "export PATH=$HOME/conda/bin:$PATH" >> $HOME/.bashrc

COPY ./ .

# Start serving Covid predicted API
CMD python3 serve.py