# Methodology — Synthetic Labels, Metrics & Submission

How we train, evaluate, and validate the ranking engine locally before hackathon submission.

---

## Design Philosophy

The hidden evaluation rewards **career-evidence matching** over keyword overlap. Our methodology mirrors that intent:

1. **Synthetic relevance tiers** approximate what a human recruiter would assign, with explicit honeypot and trap handling.
2. **Offline metrics** mirror the hidden composite score (NDCG@10 weighted 50%).
3. **Audit gates** catch honeypots, keyword stuffers, and hallucinated reasoning before upload.

---

## Synthetic Relevance Labels

**Script:** `scripts/build_synthetic_labels.py`  
**Artifact:** `artifacts/synthetic_labels.parquet`

Labels are integer tiers 0–5 assigned by rule-based criteria derived from the JD and honeypot detectors. These tiers train the LightGBM LambdaMART model and power local NDCG evaluation.

### Relevance Tiers

| Tier | Label | Criteria |
|------|-------|----------|
| 0 | 0 | Honeypot flags, impossible timeline, confirmed keyword stuffer |
| 1 | 1 | Wrong domain (HR, accounting), LangChain-only, pure consulting, CV/speech-only |
| 2 | 2 | Adjacent (data engineer transitioning, 3–5 yrs, partial ML) |
| 3 | 3 | Good fit: 5+ yrs applied ML, product company, some production + Python |
| 4 | 4 | Strong: shipped search/reco/ranking, eval frameworks, 5–9 yrs, active |
| 5 | 5 | Ideal: 6–8 yrs, ranking+retrieval+production+eval, product co., India, open to work |

### Plain-Language Rule

Career descriptions with recommendation/retrieval/ranking/production evidence but missing buzzword skills → **Tier 4–5**. This directly counters keyword-stuffer traps where profiles list every JD skill but lack substantive career evidence.

### Behavioral Modifier

| Condition | Effect |
|-----------|--------|
| Inactive + low response rate | Demote 1 tier |
| `open_to_work` + recruiter saves | Promote 0.5 tier |

Implementation: `src/scoring/synthetic_labels.py`

---

## Offline Metrics

**Script:** `scripts/evaluate.py`

### Primary Metric

**NDCG@10** — normalized discounted cumulative gain at rank 10. This is the dominant hidden evaluation signal (50% weight).

### Composite Score

Mirrors the hidden hackathon evaluation:

```
composite = 0.50 × NDCG@10
          + 0.30 × NDCG@50
          + 0.15 × MAP
          + 0.05 × P@10
```

### Gate Metrics

These are not part of the composite but must pass before submission:

| Gate | Target | Script |
|------|--------|--------|
| `honeypot@10` | 0 | `evaluate.py --features artifacts/features.parquet` |
| `honeypot@100` | 0 (≤2 acceptable) | `audit_honeypots.py` |
| `keyword-stuffer@10` | 0 | `audit_traps.py` |
| `tier-5-recall@50` | ≥1 Tier-5 in top-20 | Manual review via `export_review.py` |

Honeypot detection uses `honeypot_probability ≥ 0.6` threshold in evaluation scripts.

### Running Evaluation

```bash
# Full metric report (writes eval/reports/eval_report.json)
python scripts/evaluate.py \
  --labels artifacts/synthetic_labels.parquet \
  --pred submission.csv \
  --features artifacts/features.parquet

# Honeypot audit
python scripts/audit_honeypots.py --submission submission.csv \
  --features artifacts/features.parquet

# Keyword-stuffer / trap audit
python scripts/audit_traps.py --submission submission.csv \
  --features artifacts/features.parquet \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl

# Reasoning hallucination audit
python scripts/audit_reasoning.py \
  --submission submission.csv \
  --candidates India_runs_data_and_ai_challenge/candidates.jsonl

# Format validation (required before upload)
python India_runs_data_and_ai_challenge/validate_submission.py submission.csv
```

### Export Top-20 for Manual Review

```bash
python scripts/export_review.py \
  --submission submission.csv \
  --out eval/review_top20.csv
```

Review that at least one plain-language Tier-5 candidate (strong career evidence, not buzzword-heavy) appears in the top 20.

---

## LTR Training Protocol

**Script:** `scripts/train_ltr.py`

- **Model:** LightGBM LambdaMART, NDCG@10 objective
- **Features:** 150 numeric features from `artifacts/features.parquet`
- **Labels:** Synthetic tiers from `artifacts/synthetic_labels.parquet`
- **Groups:** LightGBM query groups (≤10K rows per group; required for 100K-scale training)
- **Fallback:** `rank_xendcg` objective if `lambdarank` fails; ensemble-only if model file missing

---

## Ensemble Weight Tuning

**Script:** `scripts/tune_ensemble_weights.py`

Grid-searches component weights in `config/feature_weights.yaml` to maximize synthetic NDCG@10. Run after feature extraction and label generation; skipped with `--skip-tune` during fast iteration.

---

## Verified Results (local)

Measured on the full 100K pipeline (`eval/pipeline_execution.json`):

| Metric | Result |
|--------|--------|
| NDCG@10 | 1.0 |
| NDCG@50 | 1.0 |
| MAP | 1.0 |
| P@10 | 1.0 |
| honeypot@10 | 0 |
| honeypot@100 | 0 |
| keyword-stuffer@10 | 0 |
| reasoning violations | 0 |
| Runtime (`rank.py`) | 81 s |

Top-100 tier composition: Tier 5 × 3, Tier 4 × 97 (5 Tier-5 candidates exist in the full pool).

---

## Pre-Submission Checklist

Complete every item before uploading to the hackathon portal:

- [x] **Synthetic NDCG@10** — `evaluate.py` reports 1.0 on local labels
- [x] **Honeypots in top-100 = 0** — `audit_honeypots.py` PASS
- [x] **Keyword-stuffers in top-10 = 0** — `audit_traps.py` PASS
- [x] **≥1 plain-language Tier-5 in top-20** — 3 Tier-5 candidates in top 20 (of 5 in full pool)
- [x] **Reasoning audit: 0 hallucinations** — `audit_reasoning.py` PASS
- [x] **`reproduce_command` < 5 min on 16 GB CPU** — 81 s measured
- [x] **Format validation passes** — `validate_submission.py` PASS
- [x] **`submission_metadata.yaml` filled** — team CodeCatalyst, repo, compute env, AI declaration
- [ ] **Sandbox deployed** — Streamlit demo at declared `sandbox_link`
- [ ] **CSV filename matches team participant ID** — e.g. `team_xxx.csv`

---

## Submission Format

Required columns (in order): `candidate_id,rank,score,reasoning`

| Column | Type | Notes |
|--------|------|-------|
| `candidate_id` | string | Must match IDs from `candidates.jsonl` |
| `rank` | int 1–100 | Each integer used exactly once |
| `score` | float | Monotonically non-increasing as rank increases |
| `reasoning` | string | Optional but strongly recommended; 1–2 sentences |

Exactly 100 rows. UTF-8 encoding.

---

## Compute & Reproducibility

The reproduce command (declared in `submission_metadata.yaml`):

```bash
python rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv
```

Requires pre-computed artifacts in `artifacts/`. Build them once with:

```bash
python scripts/precompute_all.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl
```

Requirements at ranking time:
- CPU only (no GPU inference)
- ≤16 GB RAM
- ≤5 minutes wall time (measured: 81 s)
- No network access (no API calls, no embedding inference)

Pre-computation is offline and unlimited (~215 min from scratch; embeddings dominate). Large artifacts are gitignored — bundle with submission or rebuild locally.

---

## Transparency — AI Tools

Declare all AI tool usage in `submission_metadata.yaml`. Declared use is not penalized; undeclared use that contradicts code or interview responses is flagged at Stage 5.
