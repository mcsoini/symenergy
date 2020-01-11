FROM jupyter/minimal-notebook

##  RUN pip install symenergy
USER root
COPY examples/*.ipynb /home/jovyan/work/

WORKDIR /home/jovyan/work


