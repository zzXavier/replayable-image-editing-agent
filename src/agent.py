from pathlib import Path
import hashlib
import json
import re
import numpy as np
from PIL import Image
from src.comfy import workflow


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
            scores = assess(source, image, mask)
            entry["image"] = candidate.name
            entry["sha256"] = checksum(candidate)
            entry["graph"] = graph
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
            entry["error"] = str(error) or type(error).__name__
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
    if set(saved["inputs"]) != {"source.png", "mask.png"}:
        raise ValueError("Unexpected input manifest")
    for key, digest in saved["inputs"].items():
        if checksum(out / key) != digest:
            raise ValueError("Input hash mismatch: " + key)
    mode = saved["mode"]
    limit = 1 if mode == "direct" else 2
    if mode not in ("direct", "fixed", "feedback") or saved["max_render_calls"] != 2:
        raise ValueError("Render budget or policy mismatch")
    steps = iter(saved["steps"])
    last_image, last_scores = None, None
    with Image.open(out / "source.png") as source, Image.open(out / "mask.png") as mask:
        fraction = float((np.asarray(mask) > 127).mean())
        valid = source.size == mask.size and 0.01 < fraction < 0.96
        validation = {"tool": "validate_mask", "accepted": valid}
        if valid:
            validation["keep_fraction"] = fraction
        if next(steps, None) != validation:
            raise ValueError("Missing or incorrect initial mask validation")
        status = "tool_error" if valid else "invalid_mask"
        for index in range(limit if valid else 0):
            render = next(steps, None)
            seed = saved["seed"] + index
            header = {"tool": "render", "attempt": index, "seed": seed}
            if not isinstance(render, dict) or any(render.get(k) != v for k, v in header.items()):
                raise ValueError("Missing or out-of-order render attempt")
            if "error" in render:
                error = render["error"]
                if not isinstance(error, str) or not error or render != (header | {"error": error}):
                    raise ValueError("Malformed failed render attempt")
                continue
            name = f"candidate-{index}.png"
            if render.get("image") != name or "sha256" not in render or "graph" not in render:
                raise ValueError("Render attempt lacks a result or recorded error")
            if checksum(out / name) != render["sha256"]:
                raise ValueError("Output hash mismatch")
            graph = render["graph"]
            try:
                upload = graph["2"]["inputs"]["image"]
                expected = workflow(saved["checkpoint"], upload, saved["prompt"], seed)
            except (KeyError, TypeError, AttributeError) as e:
                raise ValueError("Missing workflow/request parameters") from e
            if not isinstance(upload, str) or not upload or graph != expected:
                raise ValueError("Workflow differs from recorded request")
            packed = out / f"candidate-{index}-input.png"
            if packed.exists() or saved.get("server_source_revision"):
                # ComfyUI appends " (n)" to colliding uploads on the server.
                if not re.fullmatch(rf"candidate-{index}-input(?: \(\d+\))?\.png", upload):
                    raise ValueError("Graph refers to an unrelated uploaded image")
                if not packed.is_file():
                    raise ValueError("Missing recorded ComfyUI input image")
                with Image.open(packed) as rgba:
                    if (
                        rgba.mode != "RGBA"
                        or not np.array_equal(rgba.convert("RGB"), source)
                        or not np.array_equal(rgba.getchannel("A"), mask)
                    ):
                        raise ValueError("Uploaded pixels differ from source or mask")
            with Image.open(out / name) as image:
                now = assess(source, image, mask)
            passed = acceptable(now)
            evaluation = {"tool": "evaluate", "attempt": index, "scores": now, "accepted": passed}
            if next(steps, None) != evaluation:
                raise ValueError("Missing, out-of-order or incorrect evaluation")
            last_image, last_scores = name, now
            status = "accepted" if passed else "rejected"
            if mode == "feedback" and passed:
                break
    if any(True for _ in steps):
        raise ValueError("Unexpected steps after the policy stopped")
    if (
        saved["status"] != status
        or saved.get("chosen_image") != last_image
        or saved.get("scores") != last_scores
    ):
        raise ValueError("Final decision differs from the replayed calls")
    return saved
