from io import BytesIO
from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image, ImageOps
import requests
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://amazon-berkeley-objects.s3.amazonaws.com/images/small/'

def prepare(item, size=384, ref=False):
    key = 'reference' if ref else 'photo'
    path = ROOT / '.cache' / 'abo' / item[key]
    if not path.exists():
        r = requests.get(BASE + item[key], timeout=40)
        r.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(r.content)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if item.get(key + '_sha256', digest) != digest:
        raise ValueError(f"Image changed: {item[key]}")
    with Image.open(BytesIO(raw)) as inp:
        image = ImageOps.pad(inp.convert('RGB'), (size,size), color='white')
    # ABO packshots have a near-white backdrop. Only remove light pixels
    # connected to the border, preserving interior white text and labels.
    light = np.asarray(image).min(axis=-1) > 236
    borders = np.zeros_like(light)
    borders[[0,-1],:] = True
    borders[:,[0,-1]] = True
    bg = ndimage.binary_propagation(light & borders, mask=light)
    mask = ndimage.binary_fill_holes(~bg)
    labels, n = ndimage.label(mask)
    if n < 1:
        raise ValueError('No foreground found')
    mask = labels == np.bincount(labels.ravel())[1:].argmax()+1
    if not .01 < mask.mean() < .96:
        raise ValueError(f'Foreground fraction {mask.mean():.3f} outside supported domain')
    return image, Image.fromarray(mask.astype('uint8')*255), digest

def items():
    return json.loads((ROOT/'configs'/'products.json').read_text(encoding='utf-8-sig'))

