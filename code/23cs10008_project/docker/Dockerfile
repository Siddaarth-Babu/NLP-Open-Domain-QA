# An image is the immutable recipe and runtime environment for the QA code.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Copy dependency descriptions first. Docker can reuse this layer when only code changes.
COPY requirements-task1.txt pyproject.toml ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-task1.txt

# Source and configuration are part of the image. Data and results are mounted at runtime.
COPY code/ ./code/
COPY configs/ ./configs/
RUN python -m pip install --no-build-isolation -e .

# A container can override this command, e.g. to build the corpus or index.
CMD ["python", "code/task1/run.py"]
