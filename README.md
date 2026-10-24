# Replayable image editing agent

Run a bounded image-editing workflow against a local ComfyUI server and audit every generated candidate. Fixed order, direct inpainting and feedback have the same cap of two model calls. Direct editing uses one; fixed always attempts two; feedback stops when foreground error and red-background thresholds pass. Both score and actual calls are reported.

Start the pinned ComfyUI server with a compatible native Stable Diffusion inpainting checkpoint loaded under models/checkpoints. Install this repo's Python requirements, then run:

```sh
python run.py demo --checkpoint CHECKPOINT.safetensors
python run.py reproduce --checkpoint CHECKPOINT.safetensors
python run.py evaluate
python run.py replay --trace results/formal/B074H6ZXT5-normal/feedback/trace.json
```

`src/comfy.py` uploads a masked source, submits an inpaint graph and retrieves its PNG. `src/agent.py` checks masks, limits retries, records input/output hashes, prompt, seed, checkpoint name, graph and decisions. Replay checks saved images and recalculates scores without GPU access. To rerun diffusion the recorded graph and checkpoint must be available on the server. `src/data.py` uses the exact pinned public catalog inputs and mask semantics used by P01. The optional fault is injected and clearly labeled. This report only evaluates the predeclared request for a red background.

The pinned server also supports a Diffusers directory via --checkpoint diffusers:FOLDER. Add its parent as a diffusers search directory in ComfyUI extra_model_paths.yaml. The P01 Yahoo snapshot can then supply the SD2 inpainting backbone directly.
For an isolated launch with the pinned server already installed in the active environment, run python src/local_server.py demo --source PATH_TO_COMFYUI --model-dir PATH_TO_YAHOO_SNAPSHOT. This binds only to localhost, checks the server revision, writes a log, and stops the server after the run.
The 407 KB P01 input bundle is shipped under examples/products with per-PNG SHA-256 values. Every image and mask is verified locally before submitting a graph, so the agent input path works offline.

## Tested local HTTP environment

Use Python 3.11, install requirements-comfy-api.txt, then clone the official server with `git clone --depth 1 --branch v0.3.12 https://github.com/Comfy-Org/ComfyUI.git .cache/ComfyUI`. The exact source commit is validated by the launcher. A headless run does not require video template packages or custom nodes.

- `python src/local_server.py probe --source .cache/ComfyUI` exercises real HTTP alpha-mask upload, queue execution and PNG download. The recorded 384×384 transfer had zero mismatched pixels. See [the measured probe](results/probe/protocol.json).
- `python src/fetch_sd15.py --output .cache/sd15` fetches and hashes the pinned base model. Its CreativeML Open RAIL-M terms are available on the [official model card](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Weights stay in the local cache.
- `python src/local_server.py demo --source .cache/ComfyUI --model-dir .cache/sd15 --device cpu` runs the actual latent inpainting graph. On constrained CPUs set OMP_NUM_THREADS and MKL_NUM_THREADS to four.
- Change `demo` to `reproduce` for the full held-out experiment. Each run writes the ComfyUI source revision, execution device, model ID and SHA-256 weight digests.
The server caches the last 32 node outputs so that submitting identical seeds and images across policies can reuse identical predictions. Call counts in the table measure attempted editing-tool invocations, including explicitly injected failures, and do not imply repeated uncached denoising work.
