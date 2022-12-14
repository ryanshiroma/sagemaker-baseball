FROM python:3.8-slim-buster

# Install scikit-learn and pandas
RUN apt-get -y update
RUN apt-get -y upgrade
RUN pip install pandas==1.5.1 boto3==1.20.24 sagemaker==2.116.0 pyyaml pillow==9.3.0 scikit-learn


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
ENTRYPOINT ["python", "-u", "ml/src/entrypoint.py"]