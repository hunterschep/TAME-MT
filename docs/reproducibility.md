# Reproducibility

TAME-MT reports are designed to be reproducible experiment records, not only
terminal summaries.

## Always Keep

For paper tables or benchmark dashboards, keep:

- the JSON report;
- the full TAME-MT signature;
- the command line;
- TAME-MT, Python, Rust, and SacreBLEU versions;
- retrieval mode and backend;
- whether the run used an index bundle or cached segment diagnostics;
- the `performance` object or `--profile-json` artifact;
- corpus sizes and any sampling limits;
- hardware and operating-system details for timing claims.

## Signature

The report signature records metric-affecting choices:

- TAME-MT version;
- normalization and n-gram settings;
- retrieval mode, backend, and approximation flag;
- TM zero policy;
- bin and leak thresholds;
- pair candidate `top-k`;
- fast-mode candidate limits;
- selected metrics and SacreBLEU settings;
- metric-affecting dependency versions.

If any of those change, compare reports as different metric configurations.

## Cached Diagnostics

`score-cached` and `score-cached-batch` reuse segment diagnostics. They validate
metadata by default, including reference hashes and TM-hypothesis hashes, so a
different reference file of the same length does not silently reuse stale
target/pair exposure.

Use `--allow-reference-hash-mismatch` only for an explicit expert workflow. The
resulting target and pair exposure may be stale.

## Benchmark Claims

Do not quote local timing numbers without the machine profile and command.
Performance docs and public-corpus summaries should state that timings are
smoke measurements, not universal speed guarantees.
