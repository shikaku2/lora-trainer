# Code-only layer — rebuilds on every push in seconds
FROM ghcr.io/shikaku2/lora-trainer-base:latest

COPY handler.py         /workspace/handler.py
COPY pod_entrypoint.py  /workspace/pod_entrypoint.py
COPY train_lora.py      /workspace/train_lora.py
COPY train_cpt.py       /workspace/train_cpt.py
COPY train_dpo.py       /workspace/train_dpo.py

CMD ["python3", "-u", "pod_entrypoint.py"]
