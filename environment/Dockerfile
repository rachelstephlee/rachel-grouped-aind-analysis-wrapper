# hash:sha256:d90ef0e553ccbd5659755d4affc7c8128e5282b48e6a22c58d3ba4f01c80c1ef
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
    -e git+https://github.com/AllenNeuralDynamics/aind-analysis-results.git@9a03b4632ecb9c4fcb9b4f654e7d567b4d17e6ba#egg=aind-analysis-results

COPY postInstall /
RUN /postInstall
