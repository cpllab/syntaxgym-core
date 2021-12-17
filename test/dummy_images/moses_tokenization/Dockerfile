FROM continuumio/miniconda3

RUN pip install sacremoses

RUN mkdir -p /opt
COPY bin /opt/bin
COPY spec.json /opt/spec.json
ENV PATH "/opt/bin:${PATH}"
ENV PYTHONIOENCODING utf-8
