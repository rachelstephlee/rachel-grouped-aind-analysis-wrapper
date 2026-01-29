# hash:sha256:954b59e162e5df5340f6de8bf6ec9f86346103b378c400c688cda95ac0515ab2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV DOCDB_HOST=api.allenneuraldynamics.org
ENV DOCDB_DATABASE=analysis
ENV DOCDB_COLLECTION=architecture_example_v2
ENV CODEOCEAN_DOMAIN=codeocean.allenneuraldynamics.org
ENV ANALYSIS_BUCKET=aind-scratch-data/analysis/architecture_example_v2

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    boto3==1.42.30 \
    pydantic-settings==2.9.1 \
    s3fs==2026.1.0 \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@27bac72c93d35fbf37c20b3de06fb4cec0b4ff57#egg=analysis-pipeline-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
