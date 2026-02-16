# hash:sha256:76d217b68a97203802c3629b9da7ef696df56293b09a9e96a55cc7bf25b1ca74
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
    boto3==1.42.30 \
    pydantic-settings==2.9.1 \
    s3fs==2026.1.0 \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@38519e126458e080f44c8088d6c7dbddf17966f6#egg=analysis-pipeline-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
