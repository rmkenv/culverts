# Stage 1: Build with system deps for geopandas, shapely, proj
FROM mambaorg/micromamba:1.5.8 as build

USER root
RUN micromamba create -y -n app -c conda-forge python=3.11 \
    geopandas shapely proj geos gdal \
    && micromamba clean --all --yes

SHELL ["bash", "-lc"]
WORKDIR /app
COPY requirements.txt /app/requirements.txt
# Install pip packages inside the conda env
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate app && \
    pip install --no-cache-dir -r requirements.txt

COPY culvert_analysis.py /app/culvert_analysis.py
COPY app_api.py /app/app_api.py
COPY worker.py /app/worker.py

# Stage 2: Runtime
FROM mambaorg/micromamba:1.5.8

SHELL ["bash", "-lc"]
COPY --from=build /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH
ENV CONDA_DEFAULT_ENV=app

WORKDIR /app
COPY --from=build /app /app

EXPOSE 8000

# Default: run API
CMD ["bash", "-lc", "uvicorn app_api:app --host 0.0.0.0 --port 8000 --workers 2"]
