FROM python:3.9.13-slim

WORKDIR /colorization-AI
COPY requirements.txt /colorization-AI/
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /colorization-AI/