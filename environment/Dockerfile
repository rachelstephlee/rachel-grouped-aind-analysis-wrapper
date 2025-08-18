# hash:sha256:8fd01c5901919cfcd71488a147e14c285f4667347b3d3b547212b6a70fc5033f
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
ENV DOCDB_COLLECTION=dynamic-foraging-nm
ENV CODEOCEAN_DOMAIN=codeocean.allenneuraldynamics.org
ENV CODEOCEAN_EMAIL=rachel.lee@alleninstitute.org
ENV ANALYSIS_BUCKET=aind-analysis-prod-o5171v/dynamic-foraging-nm

RUN pip install -U --no-cache-dir \
    aind-analysis-arch-result-access==0.7.1 \
    awscli==1.38.3 \
    boto3==1.37.3 \
    ipykernel==6.30.0 \
    ipympl==0.9.6 \
    matplotlib==3.7.5 \
    numpy==1.26.4 \
    pandas==2.3.1 \
    plotly==5.24.1 \
    psutil==7.0.0 \
    pydantic-settings==2.9.1 \
    pypdf2==3.0.1 \
    rupture==1.4.5 \
    s3fs==2025.2.0 \
    scikit-learn==1.7.1 \
    scipy==1.16.0 \
    seaborn==0.13.2 \
    statsmodels==0.14.5 \
    xarray==2025.7.1 \
    zarr==2.18.6 \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@main#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@e98970ad1c1fcb2e235fef53a0cbf7361a360236#egg=aind-dynamic-foraging-data-utils \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@8722a08fc651ee0d4773927d91a5378ec8449980#egg=analysis-pipeline-utils

COPY postInstall /
RUN /postInstall
