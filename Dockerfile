FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04

ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && \
    apt-get install -y \
        git \
        python3 \
        python-is-python3 \
        python3-pip \
        python3-venv \
        libgl1 \
        libglib2.0-0 \
        libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3 /cache_models.py && rm /cache_models.py

# Add src files (Worker Template)
ADD src .

CMD [ "python3", "-m", "uvicorn", "handler:app", "--host", "0.0.0.0", "--port", "8083", "--limit-concurrency", "8" ]

