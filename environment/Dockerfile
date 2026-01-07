# hash:sha256:795756216c2aa5c0cc26922da19e10fe5178cffa33671d1d182599bd10a4fca3
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV DOCDB_HOST=api.allenneuraldynamics.org
ENV DOCDB_DATABASE=analysis
ENV DOCDB_COLLECTION=architecture_example_v2
ENV CODEOCEAN_DOMAIN=codeocean.allenneuraldynamics.org
ENV CODEOCEAN_EMAIL=tom.chartrand@alleninstitute.org
ENV ANALYSIS_BUCKET=aind-scratch-data/analysis/architecture_example_v2

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    pydantic-settings==2.9.1 \
    s3fs==2025.12.0 \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@fffaeb1bc0596c8413ccecfce5f746ba3223c24a#egg=analysis-pipeline-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
