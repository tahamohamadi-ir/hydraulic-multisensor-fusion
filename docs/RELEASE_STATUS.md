# Release status (2026-10-23)

- Scope: reproducible research software for a controlled **learning/benchmarking** exercise.
- Verification: unit tests pass on **synthetic fixtures**; this is not scientific evaluation.
- UCI datasets are not included in repository, and no real-data accuracy scores are claimed.
- Run `python scripts/get_data.py`, then the run command from the README using the original UCI dataset.
- Inspect artifacts in the ignored `results/` directory before choosing any to make public.
- No private company code, enterprise database extracts, API keys, or participant data are included.
- The project is independent and not formally associated with ADaMS Lab or any university.

## Prior-to-CV checklist

1. Reproduce the run on original UCI data and save metadata including data checksum.
2. Verify data provenance, classes, train/test separation, and numerical results.
3. Interpret at least one failed/negative result and the limitations.
4. Publish only selected redacted results with accurately stated denominators.
5. Review the code personally before claiming your own contributions.
