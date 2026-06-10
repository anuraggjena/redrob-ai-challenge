# Presentation Outline — 10 Slides

Outline for the Redrob hackathon presentation deck. Each slide includes suggested content and talking points drawn from the design spec.

---

## Slide 1: Title & Team

**Title:** Redrob Candidate Intelligence & Ranking Engine

**Content:**
- Team name and member roles
- Hackathon: Intelligent Candidate Discovery & Ranking Challenge
- Role: Senior AI Engineer — Founding Team
- One-line pitch: "Evidence-based ranking that finds who the JD *means*, not who keyword-matches what it *lists*"

**Talking points:**
- Series A AI-native talent intelligence platform
- 100K candidates → top 100 under strict compute constraints

---

## Slide 2: Problem & Traps

**Content:**
- Task: rank 100,000 profiles against a nuanced, anti-pattern JD
- Hidden eval weights: NDCG@10 (50%), NDCG@50 (30%), MAP (15%), P@10 (5%)
- Explicit traps the dataset contains:

| Trap Type | Example | Consequence |
|-----------|---------|-------------|
| Honeypots | Impossible timelines, fake seniority | >10% in top-100 = Stage 3 DQ |
| Keyword stuffers | Marketing Manager + 15 AI skills | Ranked high by naive overlap |
| Plain-language gems | "Built recommendation system" without RAG keyword | Missed by buzzword rankers |
| Behavioral ghosts | Perfect profile, zero engagement | Looks great on paper |

**Talking points:**
- Most teams will fail on honeypots or keyword traps
- The JD explicitly warns against checklist matching

---

## Slide 3: Key Insight — JD Means vs JD Lists

**Content:**
- Core insight from the JD: match **career evidence** to **role intent**
- Example contrast:

| Signal | Keyword Ranker | Our Approach |
|--------|----------------|--------------|
| Skills list | 15/15 JD skills → rank 5 | Weighted by proficiency × duration × endorsements |
| Career text | Ignored | "Shipped ranking system with NDCG eval" → Tier 5 |
| Title | Ignored | "Software Engineer" + expert LLM skills → honeypot flag |

**Talking points:**
- Responsibility phrases in role descriptions are the strongest signal
- Skills without duration/evidence are discounted

---

## Slide 4: 8-Stage Architecture

**Content:**
- Pipeline diagram (offline pre-compute → runtime 8 stages)
- Stage summary table:

```
1. Candidate Understanding  → CandidateProfile
2. Feature Extraction       → 150 features + evidence graph
3. Hybrid Retrieval         → BM25 + FAISS RRF → ~5K pool
4. Ensemble Scoring         → 7-component weighted score
5. Honeypot Detection       → 7 detectors, noisy-OR fusion
6. LTR Re-ranking           → LightGBM on top ~2K → top 100
7. Reasoning Generation     → Evidence-graph templates
8. Submission CSV           → Validated output
```

**Talking points:**
- Offline/online split: heavy work pre-computed; ranking in 81 sec (measured)
- Single entrypoint: `python rank.py --candidates India_runs_data_and_ai_challenge/candidates.jsonl --out ./submission.csv`

---

## Slide 5: Honeypot Detection

**Content:**
- Seven specialized detectors:

| Detector | Catches |
|----------|---------|
| Timeline impossibility | Work before company founded |
| Skill inflation | Expert + 0 months duration |
| Keyword stuffer | High overlap, no responsibility evidence |
| Fake seniority | Senior title, <3 yrs total |
| Behavioral outlier | Perfect profile, zero engagement |
| Education anomaly | Future degree dates |
| Role–skill mismatch | Non-tech title + dense AI skills |

- Fusion: noisy-OR → `honeypot_probability`
- Hard gate: probability > 0.7 excluded from top-100
- Target: 0 in top-10, ≤2 in top-100

**Talking points:**
- Honeypots are ~5% of the pool but dominate disqualification risk
- Each detector is independently testable (see test suite)

---

## Slide 6: Hybrid Retrieval

**Content:**
- Dense-first hybrid: BGE-small-en-v1.5 (384-d) + BM25
- Document: headline + summary + titles + career descriptions + skills
- Query: parsed JD — full embedding + skill/responsibility terms
- Fusion: Reciprocal Rank Fusion (k=60) of FAISS top-10K + BM25 top-10K
- Feature-gated boost for ranking/retrieval experience evidence
- Output: ~5,000 recall pool (not final rank)

**Talking points:**
- Retrieval is recall-oriented; precision comes from ensemble + LTR
- No embedding inference at ranking time — indexes pre-built

---

## Slide 7: Ensemble + LTR Scoring

**Content:**
- **Coarse (ensemble):** 7 components with tuned weights

```
0.30 technical + 0.20 semantic + 0.15 experience
+ 0.10 behavioral + 0.10 recruiter + 0.10 availability + 0.05 trust
− honeypot_penalty − keyword_stuffer_penalty
```

- **Hard gates:** consulting-only cap, wrong-domain cap, availability crush
- **Fine (LTR):** LightGBM LambdaMART, NDCG@10 objective, top ~2K → top 100
- Trained on synthetic tiers; 80/20 hash split with early stopping

**Talking points:**
- Ensemble handles interpretability and gates; LTR handles fine ordering
- Fallback to ensemble-only if LTR underperforms

---

## Slide 8: Evidence Graph Reasoning

**Content:**
- Graph nodes: Skills, Experience, Projects, Signals, Behavior, Availability, Requirements
- Edge types: supports, contradicts, risk, strong, weak
- Template fill only — no LLM at runtime:

```
"{relevant_years} yrs {primary_domain}; {strength_1} and {strength_2}; {concern_if_any}."
```

- Rules:
  - Every claim maps to a verified graph node
  - Omit concerns if no evidence (never fabricate)
  - Hallucination guard pre-validates output

**Talking points:**
- Recruiters trust reasoning tied to evidence, not generic LLM prose
- `audit_reasoning.py` validates before submission

---

## Slide 9: Offline Results & Benchmarks

**Content:**
- Local evaluation metrics (verified on full 100K pipeline):

| Metric | Result | Gate |
|--------|--------|------|
| NDCG@10 | 1.0 | Maximize |
| Composite | 1.0 | Mirror hidden eval |
| honeypot@10 | 0 | 0 |
| honeypot@100 | 0 | 0 |
| keyword-stuffer@10 | 0 | 0 |
| Runtime (100K) | 81 s | <5 min |

- Top-100: Tier 5 × 3, Tier 4 × 97; top-10 honeypot_prob = 0.0
- All audit scripts PASS (format, honeypots, traps, reasoning)

**Talking points:**
- Synthetic labels approximate hidden eval; gates catch disqualification risks
- Runtime uses 27% of 5-min budget — 74% headroom for judge reproduction

---

## Slide 10: Future — Online A/B & Recruiter Feedback Loop

**Content:**
- **Online A/B:** Deploy ranker variants; measure recruiter shortlist rate, time-to-shortlist, offer rate
- **Recruiter feedback loop:**
  - Capture thumbs-up/down on rankings with reason codes
  - Retrain LTR on implicit labels (shortlisted = relevant)
  - Update honeypot detectors from confirmed bad hires
- **Product extensions:**
  - Real-time re-ranking as new candidates enter pool
  - Explainability dashboard (evidence graph visualization)
  - Multi-JD support with shared feature backbone

**Talking points:**
- Synthetic labels bootstrap the model; production needs recruiter signal
- Evidence graph scales to interactive UI for hiring managers
- Architecture already separates offline training from runtime inference

---

## Appendix (Optional Backup Slides)

- Memory budget breakdown (16 GB)
- Feature group inventory (150 features)
- Test suite coverage (honeypots, reasoning, format)
- Sandbox demo walkthrough
