FROM continuumio/miniconda3

RUN mkdir -p M1_Coursework

COPY . /M1_Coursework

WORKDIR /M1_Coursework

RUN conda env update -f environment.yml --name ADSCW

RUN echo "conda activate PrincDSCW" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]