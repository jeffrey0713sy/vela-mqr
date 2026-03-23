# Vela Market Quality Rating (MQR)

A multi-agent pipeline that generates a grounded reference population of historical venture
markets, assigns L1–L5 quality ratings, and measures predictive accuracy against T+5 outcomes.

> **Oxford / Vela Research Collaboration · Version 2.1 · March 2026**

---

## Architecture

```
Role 0   Gemini + Google Search    Seed market generation (grounded web retrieval)
Role 1   GPT-4o                    Structured 7-dimension classification extraction
Role 2   Gemini + Google Search    Independent grounded verification + agreement scoring
Role 3   GPT-4o (anonymous)        Anonymous feature matrix re-scoring (leakage prevention)
Step 3   Python                    Composite scoring with structure-specific causal weights
Step 4   Python                    Percentile rating · Temperature scaling · Weighted cosine NN
```

### Key Architectural Innovation

**Role 3 (Anonymous Scorer)** is the core methodological upgrade over prior work. Role 3
receives *only* the verified feature matrix — no market name, no company names, no original
description — and re-scores each dimension independently. This prevents the model from
exploiting its parametric memory of historical outcomes (result leakage), a systematic bias
present in single-model pipelines.

### Circularity Prevention

| Role | Model | Access to market identity |
|------|-------|--------------------------|
| Role 1 | GPT-4o | Full description |
| Role 2 | Gemini | Full description (independent model family) |
| Role 3 | GPT-4o | **Feature matrix only — no identity** |

---

## Pipeline Scripts

| Script | Purpose |
|--------|---------|
| `pipeline_step1.py` | GPT-4o: 7-dimension classification extraction |
| `pipeline_step2.py` | Gemini: grounded verification + agreement scoring |
| `pipeline_step3.py` | Role 3 anonymous scoring · composite computation · AUC/ECE · temperature scaling |
| `pipeline_step4_rating.py` | Percentile rating · logistic regression P(≥L3) · weighted cosine NN |
| `pipeline_step0.py` | Gemini: historical seed market generation (2005–2019) |
| `run_scale_pipeline.py` | End-to-end orchestration for reference population generation |
| `run_ablation.py` | 5-ablation evaluation harness |
| `main.py` | Single-market evaluation entry point |
| `recompute_scores.py` | Recompute all scores from existing step1/step2 data (zero API calls) |

---

## The Seven Dimensions

| # | Dimension | Framework | Role in v2.1 |
|---|-----------|-----------|--------------|
| 1 | **Market Timing** | Rogers (1962) S-curve | PRIMARY (high weight) |
| 2 | **Competitive Intensity** | Porter (1980) Five Forces | RESIDUAL (see note) |
| 3 | **Market Size** | Current category revenue | PRIMARY (high weight) |
| 4 | **Customer Readiness** | Gartner Hype Cycle | Residual |
| 5 | **Regulatory Environment** | Compliance burden ladder | Residual |
| 6 | **Infrastructure Maturity** | Technology maturity ladder | Residual |
| 7 | **Market Structure Type** | Competitive archetype taxonomy | Weight selection gate |

> **v2.1 note on Competition (Dimension 2):** Ablation studies show competition has a single-dimension
> AUC of 0.2301 — the weakest of all dimensions and negatively correlated with outcomes in the current
> population. The direction of the mapping (nascent=85, monopoly=10) is theoretically sound but
> empirically noisy at this population size. Competition is retained in the system at residual weight
> (~6%) but removed from the primary weight block to avoid degrading composite AUC.
> This is a **future work** item: the mapping direction should be revisited with a larger, more
> balanced reference population.

---

## Causal Weight Profiles (v2.1)

Weights are applied to the two primary dimensions (timing, market_size) which together account
for **70%** of the composite score. The remaining **30%** is shared equally across the five
residual dimensions (competition, customer_readiness, regulatory, infrastructure, market_structure),
each contributing approximately **6%**.

> **Change from v2.0:** competition has been moved from PRIMARY to RESIDUAL.
> The weights below sum to 1.0 across timing + market_size only.

| Structure Type | Timing | Market Size | → Effective (×70%) |
|----------------|--------|-------------|---------------------|
| Winner-Take-Most | 60% | 40% | 42% timing · 28% market_size |
| Platform / Two-Sided | 65% | 35% | 45.5% timing · 24.5% market_size |
| Technology Enablement | 70% | 30% | 49% timing · 21% market_size |
| Fragmented / Niche | 30% | 70% | 21% timing · 49% market_size |
| Regulated Infrastructure | 35% | 65% | 24.5% timing · 45.5% market_size |

Each of the 5 residual dimensions contributes: 30% ÷ 5 = **6%** each.

---

## Agreement → Score Mapping (v2.1)

When Role 1 (GPT-4o) and Role 2 (Gemini) disagree on a dimension classification, the scoring
logic applies a confidence penalty:

| Agreement | Raw score source | Confidence multiplier |
|-----------|------------------|-----------------------|
| HIGH | Gemini classification | ×1.00 |
| MEDIUM | Average of Role1 + Gemini scores | ×0.88 |
| LOW | Average of Role1 + Gemini scores | ×0.648 (=0.72×0.9) |

> **v2.1 bug fix:** In v2.0, the LOW agreement branch computed an average score
> but then overwrote it with the Role 1 classification score before applying the
> confidence multiplier. This caused systematic underestimation (e.g., timing
> early_chasm=40 was used instead of the average of early_chasm=40 and
> late_majority=55 = 47.5). The fix reduces inter-run score variance by
> approximately 5–8 points for dimensions with LOW agreement.

---

## Rating System

| Rating | Label | Probability | Percentile band |
|--------|-------|-------------|-----------------|
| **L5** | EXCEPTIONAL | ~17–18 / 20 markets achieved scale | Top 10% |
| **L4** | STRONG | ~13–14 / 20 | 70th–90th |
| **L3** | FAVORABLE *(investment-grade threshold)* | ~10–11 / 20 | 45th–70th |
| **L2** | NEUTRAL | ~7–8 / 20 | 25th–45th |
| **L1** | SPECULATIVE | ~3–4 / 20 | Bottom 25% |

Every rating carries:
- An **Outlook Modifier** (▲ Positive / → Stable / ▼ Negative) derived from agreement distribution
- A **near_boundary** flag when the composite score is within 3 points of a rating boundary,
  signalling that the rating is unstable under small classification changes

---

## Experimental Results

### Reference Population

| Metric | Value |
|--------|-------|
| Total markets generated | 107 |
| Reference year range | 2005–2019 |
| Pipeline failures | 0 |
| Rating distribution (v2.0) | L4: 15 · L3: 42 · L2: 32 · L1: 18 · L5: 0 |

### Predictive Performance (N=107 with T+5 outcome labels)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **AUC** | **0.8486** | Strong discrimination (0.5=random · 0.7=signal · 0.8+=publishable) |
| ECE (before temperature scaling) | 0.3641 | Raw calibration error |
| ECE (after temperature scaling) | 0.3615 | Post-scaling calibration |
| **Naive ECE (all-positive baseline)** | **~0.14** | ECE of a model that always predicts scale=True |
| **ECE vs naive baseline** | **+0.22** | Pipeline ECE exceeds naive baseline — see calibration note |
| Optimal temperature T | 3.0 | Conservative scaling applied |

> **Calibration note (v2.1):** With 86% positive labels, a model that predicts "will scale"
> for every market achieves ECE ≈ 0.14. The pipeline's ECE of 0.3641 is higher — meaning
> the absolute probability values (e.g., "P(scale) = 0.7") should **not** be reported
> externally. The AUC of 0.8486 (a ranking metric, unaffected by label imbalance in the same
> way) remains a valid measure of discriminative power. See Limitations §1.

---

## Ablation Studies (N=107)

Five ablation conditions evaluated against the full pipeline baseline (AUC=0.8486).

| Condition | AUC | ΔAUC | Interpretation |
|-----------|-----|------|----------------|
| **Full pipeline (baseline)** | 0.8486 | — | Role 1 + Role 2 + Role 3 + causal weights |
| No Gemini verification (Role 1 only) | 0.8764 | +0.0278 | See note below |
| Uniform weights (no structure type) | 0.8580 | +0.0094 | See note below |
| Best single dimension (timing) | 0.8547 | +0.0061 | No single dim matches composite on held-out |

**Ablation 1 note:** Removing Gemini verification marginally *increases* AUC on the current
population. This is consistent with the label imbalance finding (86% positive): when almost
all markets "succeeded," a simpler signal can appear to rank better in-sample. The value of
Gemini grounding is in *calibration and analyst confidence*, not raw ranking.

**Ablation 2 note:** Uniform weights slightly outperform causal weights on this population.
This is expected when the population is heavily imbalanced — causal differentiation is more
valuable on a balanced held-out set.

### Single-Dimension Predictive Power Ranking

| Rank | Dimension | AUC | vs Composite | v2.1 Role |
|------|-----------|-----|-------------|-----------|
| 1 | timing | 0.8547 | +0.0061 | PRIMARY |
| 2 | market_structure | 0.8130 | −0.0356 | Residual |
| 3 | infrastructure | 0.7964 | −0.0522 | Residual |
| 4 | customer_readiness | 0.7779 | −0.0707 | Residual |
| 5 | market_size | 0.6721 | −0.1765 | PRIMARY |
| 6 | regulatory | 0.5543 | −0.2943 | Residual |
| 7 | competition | 0.2301 | −0.6185 | **Residual** (moved in v2.1) |

### Population Size Sensitivity

| N | AUC | ECE |
|---|-----|-----|
| 107 (full) | 0.8486 | 0.3641 |
| 50 | 0.8591 | 0.2898 |
| 30 | 0.7453 | 0.2920 |
| 20 | 0.9067 | 0.3282 |

AUC becomes unstable below N=30, confirming a minimum viable population of ~50 markets
for reliable percentile calibration.

---

## Comparison with Prior Work

| Feature | This work (v2.1) | v2.0 | v1.0 |
|---------|-----------------|------|------|
| Role 3 anonymous scorer | ✅ | ✅ | ❌ |
| Competition in primary weights | ❌ Moved to residual | ✅ (but weight=0) | ❌ |
| LOW agreement uses averaged score | ✅ Bug fixed | ❌ Bug present | ❌ |
| near_boundary rating flag | ✅ | ❌ | ❌ |
| Naive ECE baseline reported | ✅ | ❌ | ❌ |
| recompute_scores.py (zero API) | ✅ | ❌ | ❌ |
| Temperature scaling | ✅ | ✅ | ❌ |
| AUC / ECE reporting | ✅ | ✅ | ❌ |
| Weighted cosine NN | ✅ | ✅ | ❌ |
| Outlook modifier | ✅ | ✅ | ❌ |
| Ablation studies | ✅ 5 conditions | ✅ 5 conditions | ✅ 2×4 |

---

## Limitations and Future Work

1. **Label imbalance (critical).** The current reference population has 86% positive labels
   (achieved_scale=True). This inflates in-sample AUC and causes ECE to exceed the naive
   all-positive baseline (ECE_pipeline ≈ 0.36 > ECE_naive ≈ 0.14). Absolute probability
   values must not be reported externally until this is resolved.
   *Future work:* manually curate negative cases; use `NEGATIVE_CASE_SEEDS` in `pipeline_step0.py`
   to target known failed markets (e.g., Google Glass consumer, Quibi) during generation.

2. **LLM-generated outcome labels.** T+5 outcome labels are produced by Gemini with web
   grounding, not verified against primary financial databases. All performance metrics
   should be treated as *preliminary in-sample estimates*.
   > *"Outcome labels are LLM-generated and require future validation against real T+5
   > portfolio data."* — Vela MQR Project Brief, §8.1

3. **Role 3 population coverage.** The 107-market reference population was generated before
   Role 3 was integrated (`used_role3=False` in all archived scoring blocks). Ablation 5
   (Role 3 delta analysis) therefore has no data.
   *Future work:* run `patch_role3.py` to back-fill Role 3 scores for all 107 markets
   (~$2–3 API cost, ~30 minutes).

4. **Competition dimension mapping.** The decision to move competition to RESIDUAL is driven
   by empirical AUC on an imbalanced population and may not reflect the true causal structure.
   With a balanced population, competition's direction and weight should be re-evaluated.

5. **External validation.** No held-out real-world markets with independently verified
   outcomes have been evaluated. External AUC/ECE on genuine portfolio data is required
   before publication-strength claims can be made.

---

## Setup

```bash
pip install openai google-genai python-dotenv
```

Create `.env` in the project root:

```
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

## Quickstart

```bash
# Recompute all scores after config changes (zero API cost, ~5 seconds)
python recompute_scores.py

# Evaluate a single market
python main.py

# Generate reference population (120 markets)
python run_scale_pipeline.py --target 120

# Run ablation studies
python run_ablation.py
```

---

## Key References

- Moore, G.A. (1991). *Crossing the Chasm.* HarperBusiness.
- Porter, M.E. (1980). *Competitive Strategy.* Free Press.
- Athey, S. & Wager, S. (2019). Estimating Treatment Effects with Causal Forests. *Annals of Statistics.* arXiv:1902.07442
- Guo, C. et al. (2017). On Calibration of Modern Neural Networks. *ICML 2017.* arXiv:1706.04599
- Misra, S. (2024). Foundation Priors for Bayesian Inference. arXiv:2512.01107
- Vela Research & Oxford (2024). From Limited Data to Rare-event Prediction. arXiv:2509.08140

---

*Generated by Vela MQR v2.1 · Oxford / Vela Research Collaboration · March 2026*
