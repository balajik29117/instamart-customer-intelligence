# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements a four-part, reproducible customer-intelligence and support-copilot workflow using deterministic synthetic data. It covers data quality and SQLite analytics (A), RFM segmentation and leakage-controlled lapse prediction (B), privacy-safe ticket triage (C), and TF-IDF policy retrieval plus a resumable refund agent (D). All model calls use the supplied offline `mock_llm.py`; no API key, paid service, or network call is used. All amounts are INR.

## Reproduce

Run from the repository root with Python 3.11 or newer. The pinned dependency versions in `requirements.txt` were captured from the project environment with `pip freeze`. These commands create an isolated environment and regenerate all raw, clean, database, model and output artifacts from scratch.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows PowerShell: `.venv\Scripts\Activate.ps1`)
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python run_all.py`

Equivalent copy-paste commands:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_all.py
```

`run_all.py` runs `generate_data.py`, Part A, the independent `src.validate` checkpoint script, then Parts B, C and D in dependency order. The generator must end with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Part A prints `ALL CHECKPOINTS PASSED`; the full pipeline prints `PROJECT PIPELINE COMPLETED`.

## Part A

### Raw-data audit

| Raw table | Shape |
|---|---:|
| customers.csv | 1200 × 5 |
| products.csv | 40 × 4 |
| orders.csv | 13084 × 9 |
| tickets.csv | 164 × 5 |

The audit below treats both pandas nulls and empty CSV fields as missing. Dtypes are the pandas dtypes immediately after loading the raw export.

| Table | Column | dtype | Missing count | Missing % |
|---|---|---|---:|---:|
| customers | customer_id | object | 0 | 0.00% |
| customers | city | object | 0 | 0.00% |
| customers | signup_date | object | 0 | 0.00% |
| customers | age | object | 61 | 5.08% |
| customers | gender | object | 30 | 2.50% |
| products | product_id | object | 0 | 0.00% |
| products | product_name | object | 0 | 0.00% |
| products | category | object | 0 | 0.00% |
| products | price_inr | int64 | 0 | 0.00% |
| orders | order_id | object | 0 | 0.00% |
| orders | customer_id | object | 0 | 0.00% |
| orders | product_id | object | 0 | 0.00% |
| orders | quantity | int64 | 0 | 0.00% |
| orders | order_ts | object | 0 | 0.00% |
| orders | status | object | 0 | 0.00% |
| orders | discount_pct | int64 | 0 | 0.00% |
| orders | order_amount | object | 0 | 0.00% |
| orders | delivery_minutes | object | 2541 | 19.42% |
| tickets | ticket_id | object | 0 | 0.00% |
| tickets | customer_id | object | 0 | 0.00% |
| tickets | order_id | object | 8 | 4.88% |
| tickets | text | object | 4 | 2.44% |
| tickets | true_category | object | 0 | 0.00% |

### Specific audit figures

| Audit item | Count |
|---|---:|
| Raw order rows | 13084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12309 |
| Non-numeric order amounts | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and validation

- Customers: trimmed city strings and applied title case (24 raw spellings → 6 cities); normalised gender variants to `male`, `female` or `unknown` (30 blanks → unknown); treated 6 out-of-range ages plus 61 blanks as missing, then filled using the median age within each city via `groupby` + `transform`; parsed `signup_date` as datetime.
- Orders: parsed timestamps; removed 119 exact duplicates; retained the latest timestamp per order ID (656 older pending records removed); converted amounts to numeric and filled the 112 invalid amounts remaining after de-duplication using product price × quantity × (1 − discount/100); changed negative delivery times to missing and imputed missing completed-order delivery times with the completed-order median for that product category; cancelled orders retain missing delivery time.
- Products: converted `price_inr` to numeric. Clean tables are saved as CSV and loaded into SQLite tables `customers`, `products`, `orders`.

| Cleaned customer values | Result |
|---|---:|
| Pune | 227 |
| Delhi | 210 |
| Bengaluru | 197 |
| Chennai | 191 |
| Hyderabad | 191 |
| Mumbai | 184 |
| gender: female | 610 |
| gender: male | 560 |
| gender: unknown | 30 |
| Customers with missing age | 0 |
| Clean orders | 12309 |
| Completed orders | 11110 |
| Cancelled orders | 1199 |
| Completed orders with missing delivery | 0 |
| Cancelled orders with missing delivery | 1199 |
| Completed-order revenue | INR 2,998,668.80 |

`src/validate.py` independently checks the clean CSVs and SQLite tables (10+ assertions) and prints exactly `ALL CHECKPOINTS PASSED`; Part A also runs its own assertions. The clean data contains 12,309 unique order IDs, no pending records, no missing completed-order amounts or delivery times, and 1,199 cancelled orders with no delivery duration.

### SQL analysis

| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 city × product-category revenue | 30 cells; grand total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 products above average catalogue price | 11 products |
| Q4 customers without a completed order | 25 customers |
| Q5 spend above average | 550 customers; mean customer spend INR 2,552.06 |
| Q6 city ranking windows | 1,175 customers with completed spend; 8 rows have RANK different from ROW_NUMBER |
| Q7 daily revenue running total | 183 days; final cumulative revenue INR 2,998,668.80 |

Q6 city leaders: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Ties share a rank, so later row numbers can advance while RANK leaves gaps; this is why some RANK values differ from ROW_NUMBER. The seven executable queries are in `sql/queries.sql` and are run from Python with `pandas.read_sql_query`.

### pandas cross-check and exploration

Pivot-table grand total: INR 2,998,668.80, equal to the SQL total.

| City | Completed revenue (INR) | Mean order (INR) | Count |
|---|---:|---:|---:|
| Bengaluru | 523,643.40 | 276.47 | 1894 |
| Chennai | 463,629.15 | 272.72 | 1700 |
| Delhi | 502,829.80 | 243.62 | 2064 |
| Hyderabad | 528,589.90 | 284.95 | 1855 |
| Mumbai | 479,058.50 | 297.00 | 1613 |
| Pune | 500,918.05 | 252.48 | 1984 |

| Completed order amount statistic | Value |
|---|---:|
| Mean | INR 269.91 |
| Median | INR 135.00 |
| Skewness | 4.61 |
| Q1 | INR 64.80 |
| Q3 | INR 297.00 |
| Lower 1.5×IQR fence | INR -283.50 |
| Upper 1.5×IQR fence | INR 645.30 |
| Outliers below / above fences | 0 / 897 |

The data describes synthetic completed grocery and convenience orders across six cities and 40 products. The mean (INR 269.91) is much higher than the median (INR 135.00), and skewness 4.61 shows a strongly right-skewed spend distribution. The 1.5×IQR rule flags 897 high-value outliers and none below the lower fence; these can reflect larger baskets or electronics purchases and should be investigated rather than automatically removed. The strongest absolute correlation is order amount versus discount percentage (−0.095), followed by order amount versus quantity (+0.087); these are weak relationships, and correlation of this size does not establish causation.

## Part B

### RFM and segmentation

RFM covers 1175 customers: recency median 19 days (min 0, max 180), frequency mean 9.46 (max 33), monetary mean INR 2,552.06 (max INR 14,766.55). Snapshot: 2025-10-01 00:00:00.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 was selected: it has the strongest silhouette in this run (0.526), a clear elbow, and yields four actionable groups. Inertia always falls as K increases because adding clusters can only reduce within-cluster distances; that monotonic decrease alone does not identify a useful business segmentation.

| Segment ID (arbitrary label) | Business name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Use replenishment reminders and basket offers |
| 1 | Premium | 138 | 24.12 | 6.77 | 7,129.70 | VIP retention, early access and high-value bundles |
| 2 | Frequent Loyalists | 419 | 8.77 | 19.40 | 3,410.02 | Reward repeat ordering and cross-sell staples |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 | Win-back offers and service recovery |

Premium is segment 1, the segment with the highest mean monetary value. Cluster IDs are arbitrary labels, not an ordinal ranking. PCA explained-variance ratios: 0.634 and 0.218 (total 0.852). `outputs/customer_segments.csv` contains the required columns.

### Lapse definition, split and preprocessing

The cohort contains 1156 customers with a completed order before 2025-09-01; 443 lapsed (38.32%). A customer is lapsed when they have no completed order from 2025-09-01 through 2025-09-30. The stratified 80/20 split contains 924 training and 232 test rows (random_state=42). Features use only pre-cutoff completed/final-status orders plus customer attributes: recency, frequency, monetary, average discount, cancellation rate, average delivery, favourite product category, city, age, gender and tenure. No order dated 2025-09-01 or later entered the features, and Part B segment labels were not used as features.

The always-not-lapsed baseline has accuracy 61.6% and lapsed-class F1 0. The fitted training-only scaler's frequency mean is 8.115, versus 8.029 for the full cohort; they differ because the training split is a stratified sample rather than the entire cohort. One-hot encoding and scaling live inside the cross-validated pipeline, so every fold fits preprocessing only on its training rows; the test set is held untouched until final evaluation. A code assertion confirms there are no missing feature values.

### Model tuning and test results

Logistic regression grid: C ∈ {0.01, 0.1, 1, 10, 100}; `max_iter=1000`, 5-fold CV, F1 scoring. Random forest grid: `max_depth` ∈ {3, 5, 8}, `min_samples_leaf` ∈ {5, 10, 20}; 100 trees, random_state=42, 5-fold CV (45 fits).

| Model | Best parameters | Best CV F1 | Confusion matrix [[TN, FP], [FN, TP]] | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=100 | 0.6506 | [[111, 32], [31, 58]] | 0.6444 | 0.6517 | 0.6480 |
| Random forest | max_depth=8, min_samples_leaf=10 | 0.6811 | [[113, 30], [31, 58]] | 0.6591 | 0.6517 | 0.6554 |

Selected model: **random_forest** (test F1 0.6554). The forest detects lapsing customers with test precision 0.659 and recall 0.652; missing a likely lapse can mean losing a customer, while a false positive spends retention incentives on someone who would have stayed. The selected model clears the required 0.55 lapsed-class F1.

Class-balance experiment (training rows only): logistic regression precision 0.5891, recall 0.8539, F1 0.6972; compared with unbalanced logistic regression (0.6444, 0.6517, 0.6480), recall moves most. This makes the retention team catch more potential lapses, at the cost of more false-positive outreach.

Top five random-forest importances:
- `num__frequency`: 0.27960
- `num__monetary`: 0.27122
- `num__recency`: 0.11524
- `num__avg_delivery`: 0.06120
- `num__cancellation_rate`: 0.05247

Three largest absolute logistic-regression coefficients:
- `num__frequency`: -1.69733
- `cat__fav_category_Electronics`: -1.13506
- `cat__fav_category_Grocery`: -0.44923

The selected model is saved to `outputs/lapse_model.joblib`, reloaded, and reproduces test predictions exactly: True. The Premium test-customer retention list contains 10 rows, of which 1 truly lapsed. The Premium flag is a post-hoc whole-window RFM label that includes the September label window; therefore this retention list is illustrative and not a leakage-free evaluation.

## Part C

### Input guardrails, redaction and prompts

Tickets are processed in input-file order. Empty/whitespace tickets are marked `skipped_empty`; tickets with `est_tokens(text) > 400` are marked `flagged_long`; neither reaches the model. The reference data contains 4 empty and 4 long tickets (734 estimated tokens each).

| PII pattern | Substitutions |
|---|---:|
| Email addresses | 55 |
| Phone numbers | 91 |
| Order IDs | 132 |
| **Total** | **278** |

Redaction order and exact replacements: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. The code re-applies all three regexes and asserts that no match remains before sending the user message; the model user message is exactly the redacted ticket text. Zero PII matches remain after redaction. The zero-shot and few-shot system prompts both contain `CLASSIFY`, list billing/delivery/technical/fraud, and require JSON-only output; only the few-shot prompt contains `EXAMPLES:` and labelled examples including fraud.

### Resilience and output validation

Each full run resets `mock_llm` immediately before processing. All 19 transient failures in each run were recovered; retry waits use BASE_DELAY 0.05 × 2ⁿ × uniform(0.9, 1.15), and one retry line is printed per retry. Valid JSON categories are accepted as `classified_json`; otherwise lowercase regex word extraction accepts exactly one allowed category as `classified_keyword`; all other outputs, including JSON with an unknown category, become `flagged_invalid_output` and go to human review.

| Run | Classified | Accuracy on classified | Total tokens | Tokens / classified | Status counts (JSON / keyword / invalid / empty / long) |
|---|---:|---:|---:|---:|---|
| Zero-shot | 147 | 85.71% | 10492 | 71.37 | 135 / 12 / 9 / 4 / 4 |
| Few-shot | 147 | 100.00% | 15640 | 106.39 | 135 / 12 / 9 / 4 / 4 |

Zero-shot confusion matrix (rows = actual, columns = predicted; order billing, delivery, technical, fraud):

| Actual \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

Few-shot costs 35.02 additional tokens per classified ticket (about 5148 extra tokens across the run), while accuracy rises from 85.71% to 100.00%. In this deterministic evaluation the additional prompt cost is worthwhile because it removes the zero-shot billing misclassification errors on fraud and technical tickets.

Budget demo: TOKEN_BUDGET = 5000; 81 tickets ended as `budget_exhausted` after the running total crossed the threshold. Final few-shot run uses TOKEN_BUDGET = 100000 and produced 164 rows in `outputs/triage_results.csv`. Invalid/empty/long/budget-exhausted tickets route to human_review (NORMAL); valid fraud tickets route to security_team (HIGH); other Premium customers route to premium_desk (HIGH); all remaining valid tickets route to support_queue (NORMAL). Rule-validated outputs are the accepted JSON/keyword classifications; flagged and skipped cases go to human review.

## Part D

### Policy RAG

The five policy files are chunked into 40-word chunks with 10-word overlap, yielding 15 chunks: refund 4, delivery 3, payment 3, privacy 2, escalation 3. `TfidfVectorizer()` is fit on chunk text, and cosine similarity ranks retrieval results. Term frequency (TF) measures how often a term occurs in a chunk; inverse document frequency (IDF) downweights terms that occur in many chunks. TF-IDF is lexical, so semantically equivalent wording with different vocabulary may not match well.

| Evaluation | Expected document in top 3 (12 answerable) | Expected fact in answer (12) | Fallback (2 unanswerable) |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the refund-eligibility question (“What delivery time makes a completed order eligible for a full refund?”) and the failed-payment return-time question do not contain the expected fact in the answer. Their relevant sentences are split across non-overlapping chunks; overlap preserves sentence context. Unanswerable questions return the required “I could not find” fallback.

### Refund tools and ReAct agent

Five tools are declared in `TOOLS_SPEC` and mapped by `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, and `escalate_to_human`. The order lookup reads SQLite; refund calculation applies the completed-order / >45-minute rule; positive refunds above INR 1,000 are escalated rather than automatically issued. Simulated issue/escalation actions append timestamped JSON lines to `outputs/refund_ledger.jsonl`.

| Scenario | Status | Expected / calculated refund (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 2162 |
| S2 | REFUND | 135.00 | 5 | 2162 |
| S3 | ESCALATE | 1999.00 | 5 | 2149 |
| S4 | DENIED | 0.00 | 4 | 1617 |
| S5 | DENIED | 0.00 | 3 | 1096 |
| S6 | NOT_FOUND | 0.00 | 2 | 578 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 4518 |

Stop-safely demonstrations: S1 with token_limit=200 → `FALLBACK_TOKEN_LIMIT` (1 model call); time_limit_seconds=0 → `FALLBACK_TIMEOUT` (0 model calls). Under the default 6,000-token limit, S7 completes six model calls and exits as `FALLBACK_MAX_ITERATIONS`. Every fallback includes the completed steps and a human-handoff summary.

Crash-and-resume demonstration: crash after the second tool observation was checkpointed, then `resume_agent(run_id)` continued to `REFUND` with final answer `REFUND: INR 128.25 issued.`. Tool executions across the crash and resume, with none repeated: get_order → search_policy → calculate_refund → issue_refund. Checkpoints are stored under `state/`.

### ReAct versus plan-and-execute (under 250 words)

ReAct makes one model decision per observation, so the measured S1/S5/S6 calls are 5/3/2 and each decision can adapt to NOT_FOUND or a cancelled order. A plan-and-execute design could reduce normal model calls by requesting the whole sequence once, but it must replan when a tool returns an unexpected result and risks executing a stale plan. For this refund workflow I prefer ReAct because correctness and safe stopping matter more than saving a few local mock calls; every refund decision is checked against the database and policy before money is issued.

## Limitations and Next Steps

All records and policies are synthetic and do not represent real Instamart customers or operational policy. The classifier and agent use a deterministic offline mock rather than a production language model; real deployment would require monitoring, access controls, human review, and policy governance. TF-IDF retrieval can miss semantic paraphrases, and the RFM segmentation is descriptive rather than causal. The lapse model is trained on a single synthetic cohort and would need temporal validation and monitoring on real, consented data. **The Premium flag is a post-hoc whole-window label that includes the September label window, so the retention list is illustrative and not a leakage-free evaluation.** The next steps are to validate on fresh time periods, review false positives/negatives with business owners, and add monitored production data-quality checks.
