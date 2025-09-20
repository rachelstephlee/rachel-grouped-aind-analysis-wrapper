# hash:sha256:42f0e062e2df8c3300839ddf218f16d11c48dc29f97c23cd191a794304a92c51
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

ENV DOCDB_HOST=api.allenneuraldynamics.org
ENV DOCDB_DATABASE=analysis
ENV DOCDB_COLLECTION=dynamic-foraging-nm
ENV CODEOCEAN_DOMAIN=codeocean.allenneuraldynamics.org
ENV CODEOCEAN_EMAIL=rachel.lee@alleninstitute.org
ENV ANALYSIS_BUCKET=aind-analysis-prod-o5171v/dynamic-foraging-nm

RUN --mount=type=secret,id=git-access-token \
    GIT_ACCESS_TOKEN="$(cat /run/secrets/git-access-token)" \
    pip install -U --no-cache-dir \
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
    aind_hierarchical_bootstrap \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-basic-analysis.git@change_multi_avg_signal_output#egg=aind-dynamic-foraging-basic-analysis \
    -e git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-data-utils.git@multiple_sess_per_day_fix#egg=aind-dynamic-foraging-data-utils \
    -e git+https://github.com/AllenNeuralDynamics/analysis-pipeline-utils@ae433685788495d8cc63032d2a8fb67cb354190f#egg=analysis-pipeline-utils

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
