# Base: official SGLang GPU image (CUDA 12.x, PyTorch pre-installed)
# We use it purely for the CUDA runtime — SGLang server is never started.
FROM lmsysorg/sglang:latest

ENV PYTHONUNBUFFERED=1 \
    # Point HF cache at the RunPod network volume so the model
    # is downloaded once and reused across cold starts.
    HF_HOME=/runpod-volume/huggingface-cache \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRUST_REMOTE_CODE=true

WORKDIR /workspace

# Training stack on top of the SGLang base
RUN pip install --no-cache-dir \
    runpod \
    mistral-common \
    peft \
    bitsandbytes \
    transformers \
    datasets \
    accelerate \
    hf_transfer \
    openssh-client

COPY handler.py    /workspace/handler.py
COPY train_lora.py /workspace/train_lora.py

CMD ["python3", "-u", "handler.py"]
