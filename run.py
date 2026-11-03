import argparse
import csv
import json
import os
from pathlib import Path
from PIL import Image
from src.data import ROOT, items, prepare
from src.comfy import Comfy
from src.agent import execute, replay


def evaluate(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Task execution incomplete")
    rows = []
    for trace in sorted(out.glob("*/*/trace.json")):
        saved = replay(trace)
        rows.append(
            {
                "case": trace.parents[1].name,
                "mode": saved["mode"],
                "status": saved["status"],
                "render_calls": sum(s["tool"] == "render" for s in saved["steps"]),
                **saved.get("scores", {}),
            }
        )
    if len(rows) != manifest["expected_traces"]:
        raise ValueError("Task trace count mismatch")
    if not any(r["case"].endswith("-normal") and "keep_error" in r for r in rows):
        raise ValueError("No real render succeeded for any normal task")
    if not rows:
        raise ValueError("No finished task traces in output directory")
    with (out / "scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "mode",
                "status",
                "render_calls",
                "keep_error",
                "red_background",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    summary = {}
    for mode in ("direct", "fixed", "feedback"):
        scenarios = {}
        for suffix in ("normal", "bad-mask", "tool-fault"):
            group = [
                r
                for r in rows
                if r["mode"] == mode and r["case"].endswith("-" + suffix)
            ]
            if group:
                scenarios[suffix] = {
                    "tasks": len(group),
                    "accepted": sum(r["status"] == "accepted" for r in group),
                    "invalid_masks": sum(r["status"] == "invalid_mask" for r in group),
                    "render_attempts": sum(r["render_calls"] for r in group),
                    "with_prediction": sum("keep_error" in r for r in group),
                }
        summary[mode] = scenarios
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    from src.panel import panel

    for folder in sorted(out.glob("*-normal")):
        panel(folder, out / "comparison.jpg")
        break
    for folder in sorted(out.glob("*-tool-fault")):
        panel(folder, out / "injected-failure.jpg")
        break
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["demo", "reproduce", "evaluate", "replay"])
    p.add_argument("--url", default="http://127.0.0.1:8188")
    p.add_argument("--checkpoint", default=os.getenv("COMFY_INPAINT_CHECKPOINT"))
    p.add_argument("--output", type=Path)
    p.add_argument("--trace", type=Path)
    p.add_argument("--server-revision", help="Verified source commit if known")
    a = p.parse_args()
    out = a.output or ROOT / "results" / ("demo" if a.action == "demo" else "formal")
    if a.action == "replay":
        print(json.dumps(replay(a.trace), indent=2))
    elif a.action == "evaluate":
        evaluate(out)
    else:
        if not a.checkpoint:
            p.error(
                "Specify --checkpoint using the exact ComfyUI inpainting checkpoint filename"
            )
        engine = Comfy(
            a.url,
            a.checkpoint,
            deadline=int(os.getenv("COMFY_DEADLINE_SECONDS", "600")),
        )
        engine.revision = a.server_revision
        engine.request("GET", "/system_stats")
        folder = a.checkpoint.startswith("diffusers:")
        node, field = (
            ("DiffusersLoader", "model_path")
            if folder
            else ("CheckpointLoaderSimple", "ckpt_name")
        )
        options = engine.request("GET", "/object_info/" + node).json()[node]["input"][
            "required"
        ][field][0]
        if (a.checkpoint[10:] if folder else a.checkpoint) not in options:
            raise ValueError("Checkpoint missing from local ComfyUI configuration")
        examples = [
            i for i in items() if i["split"] == "test" and i["id"] != "B083V9C8D2"
        ]
        out.mkdir(parents=True, exist_ok=True)
        manifest = {
            "status": "in_progress",
            "expected_traces": 3 if a.action == "demo" else len(examples) * 9,
        }
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        for item in examples[:1] if a.action == "demo" else examples:
            im, mask, _ = prepare(item)
            for failure in (
                ["normal"]
                if a.action == "demo"
                else ["normal", "bad-mask", "tool-fault"]
            ):
                m = Image.new("L", mask.size) if failure == "bad-mask" else mask
                for mode in ("direct", "fixed", "feedback"):
                    execute(
                        engine,
                        im,
                        m,
                        "A studio photo of "
                        + item["name"]
                        + " with a plain red backdrop",
                        mode,
                        13,
                        out / (item["id"] + "-" + failure) / mode,
                        fail_first=failure == "tool-fault",
                    )
        manifest["status"] = "completed"
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        evaluate(out)
