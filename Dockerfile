FROM ubcdsci/jupyterlab:v0.9.0

# Install packages with specific versions from conda-forge
# Packages and versions specified in environment.yml
COPY environment.yaml .
RUN conda env update --file environment.yaml
