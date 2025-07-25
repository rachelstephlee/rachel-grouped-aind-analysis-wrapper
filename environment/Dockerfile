# hash:sha256:2f11c68a109bb38de834a8b92e1d199664331631803ec5f17891c94d8045f2ae
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
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@8722a08fc651ee0d4773927d91a5378ec8449980#egg=analysis-pipeline-utils

COPY postInstall /
RUN /postInstall
