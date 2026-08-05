# RunPod Serverless worker for the soccer analytics pipeline.
#
# Build and push:
#   docker build -t <dockerhub-user>/soccer-worker:v1 -f service/Dockerfile .
#   docker push <dockerhub-user>/soccer-worker:v1
#
# Build context is the REPO ROOT, not service/, because the image needs src/.

FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    ULTRALYTICS_AUTOINSTALL=false

# ffmpeg for duration probing; libgl/libglib for OpenCV's runtime deps, which the
# slim pytorch images do not carry and which fail late and confusingly without.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src/ /app/src/
COPY service/handler.py /app/handler.py

# Bake the weights into the image. Downloading them at runtime would happen on every
# cold container and be billed as GPU seconds -- and would fail entirely if the
# endpoint ever runs without egress.
ARG WEIGHTS_FILE=yolo26n.pt
COPY ${WEIGHTS_FILE} /app/${WEIGHTS_FILE}
ENV WEIGHTS=/app/${WEIGHTS_FILE}

# Fail the BUILD rather than the first job if the pipeline cannot be imported.
RUN python -c "import sys; sys.path.insert(0,'/app/src'); import pipeline; print('pipeline import OK')"

CMD ["python", "-u", "/app/handler.py"]
