# hash:sha256:af3252a5c53caad0861d39149d7520f9f989f8971952124375a0b50b470614cc
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
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@a6d93d2878bc30e1d370f0078f2f04bced1216f1#egg=analysis-pipeline-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
