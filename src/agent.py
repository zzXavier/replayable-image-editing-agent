from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assess(source, result, mask):
    if result.size != source.size:
        raise ValueError("Result dimension mismatch")
    keep = np.asarray(mask) > 127
    x = np.asarray(source, dtype=float)
    y = np.asarray(result, dtype=float)
    diff = np.abs(x - y).mean(axis=-1)
    hue = y[:, :, 0] - np.maximum(y[:, :, 1], y[:, :, 2])
    return {
        "keep_error": float(diff[keep].mean()),
        "red_background": float((hue[~keep] > 16).mean()),
    }


def acceptable(scores):
    return scores["keep_error"] <= 18 and scores["red_background"] >= 0.35


def execute(backend, source, mask, prompt, mode, seed, out, fail_first=False):
    if mode not in ("direct", "fixed", "feedback"):
        raise ValueError(mode)
    out.mkdir(parents=True, exist_ok=True)
    source.save(out / "source.png")
    mask.save(out / "mask.png")
    trace = {
        "mode": mode,
        "prompt": prompt,
        "seed": seed,
        "max_render_calls": 2,
        "checkpoint": backend.checkpoint,
        "server_source_revision": getattr(backend, "revision", None),
        "steps": [],
        "inputs": {k: checksum(out / k) for k in ("source.png", "mask.png")},
    }

    def finish(status):
        trace["status"] = status
        (out / "trace.json").write_text(json.dumps(trace, indent=2))
        return trace

    keep = np.asarray(mask) > 127
    if source.size != mask.size or not 0.01 < keep.mean() < 0.96:
        trace["steps"].append({"tool": "validate_mask", "accepted": False})
        return finish("invalid_mask")
    trace["steps"].append(
        {"tool": "validate_mask", "accepted": True, "keep_fraction": float(keep.mean())}
    )
    for n in range(1 if mode == "direct" else 2):
        candidate = out / f"candidate-{n}.png"
        entry = {"tool": "render", "attempt": n, "seed": seed + n}
        trace["steps"].append(entry)
        try:
            if fail_first and n == 0:
                raise RuntimeError(
                    "Injected test condition: first editing tool call unavailable"
                )
            image, graph = backend.render(source, mask, prompt, seed + n, candidate)
            image.save(candidate)
            entry["image"] = candidate.name
            entry["sha256"] = checksum(candidate)
            entry["graph"] = graph
            scores = assess(source, image, mask)
            trace["steps"].append(
                {
                    "tool": "evaluate",
                    "attempt": n,
                    "scores": scores,
                    "accepted": acceptable(scores),
                }
            )
            trace["chosen_image"] = candidate.name
            trace["scores"] = scores
            if mode == "direct" or (mode == "feedback" and acceptable(scores)):
                break
        except (OSError, ValueError, RuntimeError) as error:
            entry["error"] = str(error)
            if mode == "direct":
                break
    return finish(
        "accepted"
        if "scores" in trace and acceptable(trace["scores"])
        else ("rejected" if "scores" in trace else "tool_error")
    )


def replay(path):
    path = Path(path)
    saved = json.loads(path.read_text())
    out = path.parent
    for key, digest in saved["inputs"].items():
        if checksum(out / key) != digest:
            raise ValueError("Input hash mismatch: " + key)
    for step in saved["steps"]:
        if "image" in step:
            if checksum(out / step["image"]) != step["sha256"]:
                raise ValueError("Output hash mismatch")
            i = step["attempt"]
            s = next(
                x
                for x in saved["steps"]
                if x["tool"] == "evaluate" and x["attempt"] == i
            )
            with Image.open(out / "source.png") as src, Image.open(
                out / "mask.png"
            ) as m, Image.open(out / step["image"]) as im:
                now = assess(src, im, m)
            if now != s["scores"] or acceptable(now) != s["accepted"]:
                raise ValueError("Acceptance decision diverged")
    return saved
