FROM pytorch/pytorch:latest-gpu

#FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.10.0-gpu-py39-cu112-ubuntu20.04-sagemaker

# Install scikit-learn and pandas
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg libsndfile-dev
RUN pip install pandas==1.5.1  boto3==1.20.24 sagemaker==2.116.0  matplotlib scikit-learn


WORKDIR "/opt"

# # Dependencies
# COPY poetry.lock .
# COPY pyproject.toml .
# RUN pip install poetry
# RUN poetry config virtualenvs.create false
# RUN poetry install --no-dev

COPY /src/ /opt/ml/src/

# Git
RUN echo "SOURCE_COMMIT: $SOURCE_COMMIT"

# Entrypoint (doesn't do anything)
ENTRYPOINT ["python", "-u", "src/ml/entrypoint.py"]