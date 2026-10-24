"""Acquire the independent, pinned SD1.5 Diffusers inputs used by the published run."""

import argparse
import hashlib
import json
from pathlib import Path
import requests
from huggingface_hub import snapshot_download

REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REV = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
WEIGHTS = [
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
]
FILES = WEIGHTS + [
    "model_index.json",
    "unet/config.json",
    "vae/config.json",
    "text_encoder/config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/vocab.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/special_tokens_map.json",
]


def prepare(output):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    api = requests.get(
        f"https://huggingface.co/api/models/{REPO}/revision/{REV}",
        params={"blobs": "true"},
        timeout=30,
    )
    api.raise_for_status()
    expected = {
        f["rfilename"]: f["lfs"]["sha256"]
        for f in api.json()["siblings"]
        if f["rfilename"] in WEIGHTS
    }
    if set(expected) != set(WEIGHTS):
        raise RuntimeError("Pinned model index is incomplete")
    snapshot_download(
        REPO,
        revision=REV,
        allow_patterns=FILES,
        local_dir=output,
        cache_dir=output.parent / "hf-downloads",
    )
    for name, digest in expected.items():
        with (output / name).open("rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise RuntimeError(f"Incorrect downloaded weight: {name}")
    (output / "source-manifest.json").write_text(
        json.dumps({"repo": REPO, "revision": REV, "files": expected}, indent=2)
    )
    print(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path(".cache/sd15"))
    prepare(p.parse_args().output)
