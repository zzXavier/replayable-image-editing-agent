# Handoff

- Real local service: ComfyUI v0.3.12, commit 55add502206ed5511a04215db4ab8f1cfa3d99ae. The newer candidate commit had a comfy-kitchen / Torch 2.4 import incompatibility. The selected stable release passed a live byte-level HTTP mask check.
- Run `results/formal` completed on the pinned SD1.5 base, CPU, 2 held-out subjects × 3 conditions × 3 policies. P01 input pixels and masks are vendored as a separate CC BY 4.0 bundle. Original predeclared acceptance rules are preserved.
- `python -m unittest discover -s tests` passed four tests, and `python run.py evaluate --output results/formal` rebuilds per-case results and checks saved decisions without a model.
- Failure analysis: fixed and feedback policies recovered a prediction after both injected faults. Their normal edits still missed the red-background or foreground acceptance criteria. Consult results/formal/scores.csv and the traces before extending the dataset or policy.
- The independent GitHub clean clone completed all 18 formal tasks, a 3-workflow demo and 4 tests. All terminal decisions and retry counts matched, with maximum per-image RGB MAE 0.013724 on the 0–255 scale. Input and output hashes verify exactly within each saved run. See results/clean-clone.json for the audited commits, shared tree and numerical comparison.

## Replay integrity check

After the generation run, three new policy/graph regression tests and one uploaded-input test were added. All eight tests pass. The offline verifier checks chronological mask validation, each render attempt and error, the evaluation following each successful render, the mode-specific stopping rule, the ComfyUI node graph against the requested checkpoint, prompt and seed, and the saved upload's RGB and mask alpha. A renamed upload that ComfyUI numbered after a collision is accepted. A missing step, mismatched request, altered graph or changed upload raises an error.

All 18 archived formal traces match the original per-task CSV status, render count and available scores. This includes 18 mask validations, 20 render attempts, 14 successful image/evaluation pairs and six failed tool attempts. The terminal counts remain six invalid masks, ten rejected edits and two terminal tool errors. No inference was repeated for this verifier-only check. Next command: `python run.py replay --trace results/formal/B074H6ZXT5-normal/feedback/trace.json`.
