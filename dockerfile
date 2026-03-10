# much leaner, still has CUDA + PyTorch
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV PYTHONUNBUFFERED=1 \
    # Point HF cache at the RunPod network volume so the model
    # is downloaded once and reused across cold starts.
    HF_HOME=/runpod-volume/huggingface-cache \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache \
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
    hf_transfer 

COPY handler.py    /workspace/handler.py
COPY train_lora.py /workspace/train_lora.py

CMD ["python3", "-u", "handler.py"]
