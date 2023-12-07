FROM quay.io/jupyter/minimal-notebook:2023-11-19

# Install Conda packages
RUN conda install -y python=3.11.6 \
    ipykernel=6.26.0 \
    matplotlib=3.8.2 \
    pandas=2.1.3 \
    scikit-learn=1.3.2 \
    requests=2.31.0 \
    ipython=8.17.2 \
    pytest=7.4.3 \
    click=8.1.7 \
    make \
    vl-convert-python=1.1.0 \
    seaborn=0.12.0 \
    jupyter-book=0.15.1 \
    notebook=7.0.6 -c conda-forge 
RUN pip install altair==5.1.2
