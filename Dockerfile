FROM axolotlai/axolotl-cloud:main-latest

LABEL org.opencontainers.image.source=https://github.com/shikaku2/lora-trainer

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/huggingface-cache \
    TRANSFORMERS_CACHE=/workspace/huggingface-cache/hub \
    HUGGINGFACE_HUB_CACHE=/workspace/huggingface-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true \
    CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

RUN pip install --no-cache-dir runpod hf_transfer

COPY handler.py         /workspace/handler.py
COPY pod_entrypoint.py  /workspace/pod_entrypoint.py
COPY train.py           /workspace/train.py

CMD ["python3", "-u", "pod_entrypoint.py"]
