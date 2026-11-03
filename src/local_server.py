"""Run one bounded ComfyUI experiment and stop its local server on exit."""

import argparse
import hashlib
import json
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
PIN = "55add502206ed5511a04215db4ab8f1cfa3d99ae"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["probe", "demo", "reproduce"])
    parser.add_argument(
        "--source", type=Path, required=True, help="Local checkout of pinned ComfyUI"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        help="One local Diffusers inpainting snapshot (for example the P01 Yahoo snapshot)",
    )
    parser.add_argument("--port", type=int, default=8188)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output", type=Path)
    a = parser.parse_args()
    source = a.source.resolve()
    model = a.model_dir.resolve() if a.model_dir else None
    if a.action != "probe" and (
        not model or not (model / "model_index.json").is_file()
    ):
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
        "formal" if a.action == "reproduce" else a.action
    )
    results.mkdir(parents=True, exist_ok=True)
    if model:
        info = {
            "folder": model.name,
            "engine": rev,
            "device": a.device,
            "weights": {},
            "omp_threads": os.getenv("OMP_NUM_THREADS"),
            "mkl_threads": os.getenv("MKL_NUM_THREADS"),
        }
        source_manifest = model / "source-manifest.json"
        if source_manifest.exists():
            source_data = json.loads(source_manifest.read_text(encoding="utf-8-sig"))
            info.update({k: source_data[k] for k in ("repo", "revision")})
        for filename in sorted(model.glob("*/*.safetensors")):
            with filename.open("rb") as weight:
                info["weights"][filename.relative_to(model).as_posix()] = (
                    hashlib.file_digest(weight, "sha256").hexdigest()
                )
        (results / "model-manifest.json").write_text(json.dumps(info, indent=2))
    with tempfile.TemporaryDirectory() as temp:
        tmp = Path(temp)
        cfg = tmp / "model_paths.yaml"
        cfg.write_text(
            yaml.safe_dump(
                {"experiment": {"diffusers": str(model.parent if model else tmp)}}
            )
        )
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
                "--disable-all-custom-nodes",
                "--cache-lru",
                "32",
                "--cpu" if a.action == "probe" or a.device == "cpu" else "--lowvram",
                "--extra-model-paths-config",
                str(cfg),
                "--input-directory",
                str(tmp / "input"),
                "--output-directory",
                str(tmp / "output"),
                "--user-directory",
                str(tmp / "user"),
            ]
            for sub in ("input", "output", "user", "frontend"):
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
                except (requests.ConnectionError, requests.Timeout):
                    pass
                time.sleep(1)
            else:
                raise TimeoutError("ComfyUI startup timed out")
            if a.action == "probe":
                subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "tests/check_live_server.py"),
                        "--url",
                        base,
                        "--output",
                        str(results / "protocol.json"),
                    ],
                    check=True,
                )
                return
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
