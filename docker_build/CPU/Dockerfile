FROM python:3.10
#FROM pytorch/pytorch:1.0.1-cuda10.0-cudnn7-runtime

RUN mkdir /app
WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update &&  apt-get install -y build-essential
RUN apt-get update --fix-missing
#RUN apt-get install pkg-config libsrtp2-dev libopus-dev libvpx-dev -y
#RUN apt-get install ffmpeg opus libvpx pkg-config
#RUN apt-get install poppler-utils -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
COPY ./requirements.txt /app/.
RUN pip install -r requirements.txt --ignore-installed
RUN pip install streamlit==1.10.0 streamlit-option-menu==0.3.2

CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0
