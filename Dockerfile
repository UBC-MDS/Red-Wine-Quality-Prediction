FROM quay.io/jupyter/minimal-notebook:2023-11-19

# Install Conda packages
# Install Conda packages
RUN conda install -y python=3.11.6 \
    ipykernel \
    matplotlib \
    pandas \
    scikit-learn \
    requests \
    ipython \
    graphviz \
    python-graphviz \
    lightgbm \
    altair



