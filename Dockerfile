# much leaner, still has CUDA + PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-client \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    mistral-common \
    peft \
    bitsandbytes \
    transformers \
    datasets \
    accelerate \
    "trl==0.12.2" \
    hf_transfer

COPY handler.py    /workspace/handler.py
COPY train_lora.py /workspace/train_lora.py
COPY train_cpt.py  /workspace/train_cpt.py
COPY train_dpo.py  /workspace/train_dpo.py

CMD ["python3", "-u", "handler.py"]
