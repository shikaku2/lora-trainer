# Code-only layer — rebuilds on every push in seconds
FROM ghcr.io/shikaku2/lora-trainer-base:latest

COPY handler.py         /workspace/handler.py
COPY pod_entrypoint.py  /workspace/pod_entrypoint.py
COPY train.py           /workspace/train.py

CMD ["python3", "-u", "pod_entrypoint.py"]
