# TAME-MT Method Specification

This document defines the public semantics of TAME-MT. The short version:
TAME-MT contextualizes BLEU and chrF with train-test exposure. It does not prove
memorization, replace human evaluation, or guarantee generalization.

## Inputs

TAME-MT expects aligned UTF-8 text files:

- `train.src`: training sources.
- `train.tgt`: paired training targets.
- `test.src`: test sources.
- `ref`: one or more references.
- `hyp`: one system output for scoring mode.

Audit mode can run without `hyp`. Source-only audit can run without `train.tgt`
or references, but target, pair, TM, and delta-over-TM metrics require the
corresponding files.

## Normalization

By default, text is normalized with Unicode NFKC, stripped at the ends, and
collapsed whitespace. Optional settings can lowercase, strip diacritics, and
normalize punctuation.

All exposure metrics are computed after normalization. Exact overlap therefore
means exact equality after the configured normalization.

## Character N-Grams

For a normalized string `a`, `G(a)` is the set of character n-grams for the
configured orders. The default orders are 3, 4, and 5. If a non-empty string is
shorter than the smallest configured order, the whole string is used as one
gram. Empty strings have an empty gram set.

## Jaccard Similarity

The similarity between normalized strings `a` and `b` is:

$$
\mathrm{sim}(a,b) =
\frac{|G(a) \cap G(b)|}{|G(a) \cup G(b)|}
$$

If both strings are empty, similarity is defined as `1.0`. If only one string is
empty, similarity is `0.0`.

## Source Exposure

For test source `x_i`, exact source exposure is:

$$
E_i^{src} =
\max_{1 \le j \le m}\mathrm{sim}(x_i, u_j)
$$

where `u_j` is training source `j`. Ties are broken by the lowest training
index. `SourceExposure` in reports means this exact maximum unless the report
retrieval block says `source_exposure_mode` is `approx`.

## Target Exposure

For reference `r_i`, exact target exposure is:

$$
E_i^{tgt} =
\max_{1 \le j \le m}\mathrm{sim}(r_i, v_j)
$$

where `v_j` is training target `j`. With multiple references, TAME-MT uses the
best target exposure across references and records the reference index in
segment diagnostics.

## Pair Exposure

For a single test source/reference pair `(x_i, r_i)` and a training pair
`(u_j, v_j)`, pair similarity is:

$$
P_{ij} =
\min(\mathrm{sim}(x_i,u_j), \mathrm{sim}(r_i,v_j))
$$

TAME-MT currently computes pair exposure by reranking a top-k candidate set from
source and target retrieval. This is not an all-pairs exact threshold search.
Reports therefore label threshold rates as `PairLeakTopK@t`.

If `exact_pair_thresholds` is enabled, TAME-MT also computes
`PairLeakExact@t` for the configured leak thresholds. For each test segment,
the exact flag is true if there exists a training index `j` and reference `r`
such that both `sim(x_i,u_j) >= t` and `sim(r_i,v_j) >= t`. This has no false
negatives for that threshold and is reported separately from top-k pair
exposure because it may be substantially slower.

Exact pair overlap is separate and exact: it checks whether the normalized
source/reference pair appears exactly as a normalized training source/target
pair.

## TM Baseline

The translation-memory baseline retrieves the nearest training source for each
test source and outputs that training row's paired target:

$$
\mathrm{TMHyp}_i = v_{j^*}
$$

where:

$$
j^* = \arg\max_j \mathrm{sim}(x_i,u_j)
$$

`TM-BLEU` and `TM-chrF` are standard corpus metrics computed on these TM
hypotheses. Reports include `tm_retrieval_exact` so approximate TM baselines are
not confused with exact nearest-neighbor baselines.

## Delta Over TM

For metric `M`, delta over TM is:

$$
\Delta\mathrm{TM}_M = M(\mathrm{hyp},\mathrm{ref}) -
M(\mathrm{TMHyp},\mathrm{ref})
$$

A high delta over TM means the system outperforms nearest-neighbor training-set
reuse under the same metric. It does not prove broad generalization by itself.

## Source Bins

Segments are assigned by exact source overlap first, then source exposure:

- `source_exact`: normalized test source exactly appears in training.
- `near`: source exposure is at least the near threshold.
- `medium`: source exposure is at least the far threshold and below near.
- `far`: source exposure is below the far threshold.

Default thresholds are `far = 0.30` and `near = 0.70`.

## GenGap

For metric `M`, generalization gap is:

$$
\mathrm{GenGap}_M = M_{\mathrm{near}} - M_{\mathrm{far}}
$$

It is undefined when either bin is empty.

## Retrieval Regimes

### exact

Exact retrieval computes exact nearest-neighbor source and target exposure.
`TM-BLEU` uses exact source nearest neighbors. Pair threshold rates are still
top-k candidate limited and are labeled `PairLeakTopK`.

### guarded

Guarded retrieval is reserved for threshold-safe large-corpus reporting. It must
provide no-false-negative threshold flags for official source bins and leak
thresholds. If a run reports only threshold-safe bounds instead of exact maximum
scores, the report must say so in `source_exposure_mode`.

The current implementation exposes exact threshold APIs on `native_exact`:
`batch_best_above`, `batch_threshold_flags`, and `batch_source_bins_exact`.
These derive threshold decisions from exact nearest-neighbor scores, so they
have no false negatives. They are conservative correctness APIs, not a separate
optimized guarded retrieval mode yet; paper-facing defaults still use exact
retrieval for reported source-exposure scores.

### approx

Approximate retrieval uses bounded candidate generation and exact reranking of
that candidate set. It is useful for exploratory work, but it can miss true
nearest neighbors. Reports set `approximate: true`, `tm_retrieval_exact: false`,
and label pair thresholds as `PairLeakTopK`.

Approximate mode must be requested explicitly with `--retrieval approx
--allow-approximate`.

## Signature Fields

Every report signature includes:

- TAME-MT version.
- normalization.
- similarity and n-gram orders.
- retrieval mode and approximation flag.
- requested index mode and resolved backend.
- TM zero policy.
- source bin and leak thresholds.
- pair top-k.
- fast-mode candidate limits.
- BLEU/chrF settings.
- metric-affecting dependency versions.

## Limitations

TAME-MT is surface-form based. It can miss semantic paraphrases. High exposure
does not prove bad behavior; it narrows the claim supported by a benchmark.
Low exposure does not guarantee out-of-training-distribution generalization.
Approximate retrieval should be validated before paper-critical use.
