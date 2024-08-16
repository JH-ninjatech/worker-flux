# Base image -> https://github.com/runpod/containers/blob/main/official-templates/base/Dockerfile
# DockerHub -> https://hub.docker.com/r/runpod/base/tags
FROM runpod/base:0.6.2-cuda12.2.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python /cache_models.py && rm /cache_models.py

# Add src files (Worker Template)
ADD src .

CMD [ "python", "-u", "/handler.py" ]