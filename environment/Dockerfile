# hash:sha256:9ba17e9455b2117a759cc1987ae7db6dcb7fd974962efa614b36c23c3ada6e2c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG AWS_ACCESS_KEY_ID
ARG AWS_DEFAULT_REGION
ARG AWS_SECRET_ACCESS_KEY
ARG CODEOCEAN_API_TOKEN

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

ENV DOCDB_HOST=api.allenneuraldynamics.org
ENV DOCDB_DATABASE=analysis
ENV DOCDB_COLLECTION=architecture_example_v2
ENV CODEOCEAN_DOMAIN=codeocean.allenneuraldynamics.org
ENV CODEOCEAN_EMAIL=YOUR_EMAIL_HERE@alleninstitute.org
ENV ANALYSIS_BUCKET=aind-scratch-data/analysis/architecture_example_v2

RUN pip install -U --no-cache-dir \
    pydantic-settings==2.9.1 \
    s3fs==0.4.2 \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@8a69491c67b79e5dcfb84084513ecba33b0160ed#egg=analysis-pipeline-utils

COPY postInstall /
RUN /postInstall
