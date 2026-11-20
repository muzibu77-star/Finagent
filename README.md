# Finagent asset backup

Files that `.gitignore` keeps out of `main` and that cannot be regenerated for free:
experiment outputs and API caches (`artifacts/stage*`, `/test/Finagent/runs`, `/test/Finagent/tmp`),
the frozen evaluation sets (`data/staged/eval_v1`, `eval_v2`, `seed_claims_v*.jsonl`),
evidence screenshots for `eval_v2_labels`, the extraction SFT data, and the trained LoRA
adapter `runs/stage4_extract_v1/step_0072` (adapter weights only; optimizer `state.pt`
files were left out, so training cannot be resumed from them).

Not included (re-downloadable or rebuildable): `data/raw/` (see `configs/cninfo_sources.json`,
`configs/policy_sources.json`), `library_v*`, `cninfo_parsed_v1`, base models, venvs.

Restore (paths inside the archive are absolute minus the leading `/`):

    sha256sum -c SHA256SUMS
    cat finagent_assets.tar.gz.part-* | tar -xzf - -C /
