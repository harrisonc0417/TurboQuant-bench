"""
Gemma 4 E4B — Local Setup & Inference Guide
=============================================

Requirements
------------
- Python 3.10+
- ~8 GB disk space for weights (BF16)
- ~8 GB VRAM minimum (dense 4B model)
  - Recommended: 16 GB+ VRAM (single GPU, e.g. RTX 3090/4090)
  - Multi-GPU or CPU offload supported via device_map="auto"
- HuggingFace account with Gemma license accepted at:
  https://huggingface.co/google/gemma-4-E4B

Step 1 — Install dependencies
------------------------------
pip install -U transformers torch accelerate huggingface_hub

Step 2 — Authenticate with HuggingFace
---------------------------------------
Run this once in your terminal before executing the script:

    huggingface-cli login

Paste your HF token when prompted (generate one at https://huggingface.co/settings/tokens).
"""

# ─────────────────────────────────────────────
# STEP 1: Download weights to local disk
# ─────────────────────────────────────────────
from huggingface_hub import snapshot_download
import os

MODEL_ID = "google/gemma-4-E4B"

LOCAL_MODEL_DIR = os.path.expanduser("/home/harrison/Projects/LocalAI/models/gemma-4-e4b")

print(f"Downloading weights to: {LOCAL_MODEL_DIR}")
print("This is ~8 GB — should be quick.")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_MODEL_DIR,
    ignore_patterns=["*.msgpack", "*.h5"],  # skip non-safetensors formats
)

print("Download complete!")


# ─────────────────────────────────────────────
# STEP 2: Load the model from local disk
# ─────────────────────────────────────────────
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading model from disk...")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

print(f"Model loaded on: {next(model.parameters()).device}")


# ─────────────────────────────────────────────
# STEP 3: Text inference
# ─────────────────────────────────────────────
def run_text(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


print(run_text("Write a short joke about saving RAM."))


# ─────────────────────────────────────────────
# STEP 4: Multi-turn conversation
# ─────────────────────────────────────────────
def chat(history: list[dict], user_message: str) -> tuple[str, list[dict]]:
    history.append({"role": "user", "content": user_message})

    inputs = tokenizer.apply_chat_template(
        history,
        tokenize=True,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=1.0,
            top_p=0.95,
            top_k=64,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    history.append({"role": "assistant", "content": response})
    return response, history


history: list[dict] = []
reply, history = chat(history, "What's the capital of France?")
print(reply)
reply, history = chat(history, "And what's a famous landmark there?")
print(reply)


# ─────────────────────────────────────────────
# Tips
# ─────────────────────────────────────────────
# • E4B is a dense 4B model — very fast at inference, low VRAM.
# • If you have <8 GB VRAM, set device_map="auto" to spill layers to CPU RAM.
# • To speed up repeated runs, weights are cached at LOCAL_MODEL_DIR —
#   snapshot_download() will skip already-downloaded files on re-run.
