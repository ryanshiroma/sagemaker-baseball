FROM python:3.8-slim-buster

# Install scikit-learn and pandas
RUN apt-get -y update
RUN apt-get -y upgrade
RUN apt-get install -y ffmpeg libsndfile-dev
RUN pip install pandas==1.5.1 librosa==0.9.2 boto3==1.20.24 pillow==9.3.0 sagemaker==2.116.0 bs4 noisereduce matplotlib lxml pyyaml scikit-learn


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