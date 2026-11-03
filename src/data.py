"""Load the bundled product images and masks offline and verify every input hash."""

import hashlib
import json
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "examples" / "products"


def items():
    return json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8-sig"))


def prepare(item):
    files = {}
    for name in ("input.png", "mask.png"):
        rel = f'{item["id"]}/{name}'
        target = BUNDLE / rel
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != item["files"][rel]:
            raise ValueError("Product input bundle hash mismatch: " + rel)
        with Image.open(target) as image:
            files[name] = image.copy()
    return (
        files["input.png"].convert("RGB"),
        files["mask.png"].convert("L"),
        item["original_sha256"],
    )
