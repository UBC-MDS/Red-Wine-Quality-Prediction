FROM quay.io/jupyter/minimal-notebook:2023-11-19

# Install Conda packages
# Install Conda packages
RUN conda install -y python=3.11.6 \
    ipykernel 6.26.0 \
    matplotlib 3.8.2 \
    pandas 2.1.3 \
    scikit-learn 1.3.2 \
    requests 2.31.0 \
    ipython 8.17.2 \
    graphviz 9.0.0 \
    python-graphviz 0.20.1 \
    lightgbm 4.1.0 \
    altair 5.1.2



