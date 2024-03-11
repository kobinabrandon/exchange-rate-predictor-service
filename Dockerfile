# Set Python version
FROM python:3.11.8-slim-bullseye

ENV PYTHONBUFFERED=1

ENV POETRY_VERSION=1.8.2
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Set path for poetry environment
ENV PATH="${PATH}:${POETRY_VENV}/bin"

# Set Working Directory
WORKDIR /Exchange-Rate-Predictor/

# Get list of dependencies
COPY poetry.lock pyproject.toml /Exchange-Rate-Predictor/

# Copy the scripts
COPY . /Exchange-Rate-Predictor/

# Install dependencies  
RUN poetry install 

# Start the server
CMD ["poetry", "run", "/Exchange-Rate-Predictor/src/inference_pipeline/app/main.py"]
