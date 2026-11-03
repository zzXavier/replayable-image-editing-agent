"""Check real Comfy HTTP upload/queue, PNG alpha semantics and download without a checkpoint."""

from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import sys
import tempfile
import time
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.comfy import Comfy
from src.data import items, prepare


def probe(url):
    client = Comfy(url, "no-weights-for-mask-contract")
    stats = client.request("GET", "/system_stats").json()
    item = items()[0]
    source, mask, _ = prepare(item)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "product-mask-contract.png"
        rgba = source.convert("RGBA")
        rgba.putalpha(mask)
        rgba.save(path)
        uploaded = client.upload(path)
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": uploaded}},
        "2": {"class_type": "MaskToImage", "inputs": {"mask": ["1", 1]}},
        "3": {
            "class_type": "SaveImage",
            "inputs": {"images": ["2", 0], "filename_prefix": "product-mask-contract"},
        },
    }
    receipt = client.request("POST", "/prompt", json={"prompt": graph}).json()
    ticket = receipt["prompt_id"]
    for _ in range(120):
        record = client.request("GET", "/history/" + ticket).json().get(ticket)
        if record:
            break
        time.sleep(0.25)
    else:
        raise TimeoutError("Mask round-trip did not complete")
    if record.get("status", {}).get("status_str") != "success":
        raise RuntimeError(f'Graph failed: {record.get("status")}')
    png = record["outputs"]["3"]["images"][0]
    payload = client.request(
        "GET", "/view", params={k: png[k] for k in ("filename", "subfolder", "type")}
    ).content
    decoded = np.asarray(Image.open(BytesIO(payload)).convert("L"))
    mismatch = int(np.count_nonzero(decoded != (255 - np.asarray(mask))))
    if mismatch:
        raise ValueError(f"{mismatch} mask pixels changed")
    return {
        "result": "passed",
        "task": "round-trip public product alpha mask through ComfyUI HTTP queue",
        "item": item["id"],
        "mismatch_pixels": mismatch,
        "size": list(source.size),
        "mask_sha256": hashlib.sha256(payload).hexdigest(),
        "comfyui_version": stats.get("system", {}).get("comfyui_version"),
        "includes_diffusion_inference": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = probe(args.url)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
