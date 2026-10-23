"""Run one bounded ComfyUI experiment and stop its local server on exit."""

import argparse
from pathlib import Path
import os
import socket
import subprocess
import sys
import tempfile
import time
import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
PIN = "ab08f7b2caa6d4007ecde3762116c6193613f43b"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["demo", "reproduce"])
    parser.add_argument(
        "--source", type=Path, required=True, help="Local checkout of pinned ComfyUI"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="One local Diffusers inpainting snapshot (for example the P01 Yahoo snapshot)",
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    source = a.source.resolve()
    model = a.model_dir.resolve()
    if not (model / "model_index.json").is_file():
        raise ValueError("Missing Diffusers snapshot")
    rev = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if rev != PIN:
        raise ValueError(f"Checkout {PIN}, found {rev}")
    with socket.socket() as s:
        if not s.connect_ex(("127.0.0.1", a.port)):
            raise RuntimeError("Port already occupied")
    results = a.output or ROOT / "results" / (
        "demo" if a.action == "demo" else "formal"
    )
    results.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        cfg = tmp / "model_paths.yaml"
        cfg.write_text(yaml.safe_dump({"experiment": {"diffusers": str(model.parent)}}))
        log = results / "comfy-start.log"
        with log.open("wb") as stream:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            cmd = [
                sys.executable,
                str(source / "main.py"),
                "--listen",
                "127.0.0.1",
                "--port",
                str(a.port),
                "--disable-auto-launch",
                "--offline",
                "--disable-all-custom-nodes",
                "--lowvram",
                "--extra-model-paths-config",
                str(cfg),
                "--input-directory",
                str(tmp / "input"),
                "--output-directory",
                str(tmp / "output"),
                "--user-directory",
                str(tmp / "user"),
            ]
            for sub in ("input", "output", "user"):
                (tmp / sub).mkdir()
            server = subprocess.Popen(
                cmd,
                cwd=source,
                stdout=stream,
                stderr=subprocess.STDOUT,
                creationflags=flags,
            )
        base = f"http://127.0.0.1:{a.port}"
        try:
            for _ in range(120):
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited; inspect {log}")
                try:
                    if requests.get(base + "/system_stats", timeout=2).ok:
                        break
                except requests.ConnectionError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("ComfyUI startup timed out")
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "run.py"),
                    a.action,
                    "--url",
                    base,
                    "--checkpoint",
                    "diffusers:" + model.name,
                    "--output",
                    str(results),
                    "--server-revision",
                    rev,
                ],
                check=True,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
