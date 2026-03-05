# hash:sha256:9be9da638ad9243a3f74bc4f5992f3a4aa177e59a4f326f12be2c970cd6a4f01
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    analysis-pipeline-utils==0.2.1 \
    boto3==1.42.30 \
    pydantic-settings==2.9.1 \
    s3fs==2026.1.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
