# Handoff

- Real local service: ComfyUI v0.3.12, commit 55add502206ed5511a04215db4ab8f1cfa3d99ae. The newer candidate commit had a comfy-kitchen / Torch 2.4 import incompatibility. The selected stable release passed a live byte-level HTTP mask check.
- Run `results/formal` completed on the pinned SD1.5 base, CPU, 2 held-out subjects × 3 conditions × 3 policies. P01 input pixels and masks are vendored as a separate CC BY 4.0 bundle. Original predeclared acceptance rules are preserved.
- `python -m unittest discover -s tests` passed four tests, and `python run.py evaluate --output results/formal` rebuilds per-case results and checks saved decisions without a model.
- Failure analysis: fixed and feedback policies recovered a prediction after both injected faults. Their normal edits still missed the red-background or foreground acceptance criteria. Consult results/formal/scores.csv and the traces before extending the dataset or policy.
- Next: run formal reproduction once more from the private remote's clean clone and compare traces, image hashes and scores before public release.

