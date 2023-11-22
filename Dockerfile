FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -y pandas=2.1.3 \
    altair=5.1.2
    ipykernel
    matplotlib>=3.8.0
    scikit-learn>=1.3.1    
    requests>=2.24.0
    ipython
    altair=5.1.2
    vl-convert-python  # For saving altair charts as static images 
    jinja2
    pytest


