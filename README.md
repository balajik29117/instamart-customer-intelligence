# Instamart Customer Intelligence & Support Copilot

## Overview

This repository contains a reproducible, end-to-end capstone built from fictional synthetic grocery-delivery data. It connects four stages: (A) data cleaning and SQL analytics, (B) customer segmentation and lapse prediction, (C) privacy-safe support-ticket triage, and (D) policy retrieval and a refund decision agent. All currency values are in Indian Rupees (INR). No real customer information, paid API, hosted language model, or API key is used.

The two supplied files, `generate_data.py` and `mock_llm.py`, are kept unmodified. `generate_data.py` creates the deterministic raw data and five policy documents. All model-like calls use the offline `mock_llm.py` supplied in the assignment.

## Reproduce

Use Python 3.11 or newer. Run these commands from the repository root in order. The Windows activation command is shown; on macOS/Linux use `source .venv/bin/activate` instead.

1. `python -m venv .venv`
2. `.venv\Scripts\activate`
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python -m src.part_a`
6. `python -m src.part_b`
7. `python -m src.part_c`
8. `python -m src.part_d`

The generator's final line is `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. Each part writes its required files into `data/`, `outputs/`, or `state/`. `run_all.py` is an optional shortcut that runs the same commands in dependency order. Fixed random seeds are used wherever randomness is required. The four direct runtime dependencies and their tested versions are pinned in `requirements.txt`.

## Part A

### Raw-data audit

The raw files contain customers (1,200 × 5), products (40 × 4), orders (13,084 × 9), and tickets (164 × 5). `outputs/part_a_audit.json` records every column's inferred pandas dtype and blank/missing count and percentage. The raw-column audit is summarised below (missing includes blank strings preserved by `keep_default_na=False`).

| Raw table (shape) | Column | pandas dtype | Missing / blank count | Missing / blank % |
|---|---|---|---:|---:|
| customers (1,200 × 5) | customer_id | object | 0 | 0.00% |
| customers | city | object | 0 | 0.00% |
| customers | signup_date | object | 0 | 0.00% |
| customers | age | object | 61 | 5.08% |
| customers | gender | object | 30 | 2.50% |
| products (40 × 4) | product_id | object | 0 | 0.00% |
| products | product_name | object | 0 | 0.00% |
| products | category | object | 0 | 0.00% |
| products | price_inr | int64 | 0 | 0.00% |
| orders (13,084 × 9) | order_id | object | 0 | 0.00% |
| orders | customer_id | object | 0 | 0.00% |
| orders | product_id | object | 0 | 0.00% |
| orders | quantity | int64 | 0 | 0.00% |
| orders | order_ts | object | 0 | 0.00% |
| orders | status | object | 0 | 0.00% |
| orders | discount_pct | int64 | 0 | 0.00% |
| orders | order_amount | object | 0 | 0.00% |
| orders | delivery_minutes | object | 2,541 | 19.42% |
| tickets (164 × 5) | ticket_id | object | 0 | 0.00% |
| tickets | customer_id | object | 0 | 0.00% |
| tickets | order_id | object | 8 | 4.88% |
| tickets | text | object | 4 | 2.44% |
| tickets | true_category | object | 0 | 0.00% |

The main audit checkpoints are:

| Audit item | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate order rows | 119 |
| Pending order rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts | 120 |
| Negative delivery durations | 106 |
| Blank delivery durations | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and changes

- **Customers:** city values are trimmed and title-cased (24 raw spellings become six cities; 476 values change); gender variants are mapped to `male`, `female`, or `unknown` (950 raw values are normalised, including 30 blanks); ages outside 10–100 are set to missing and all 67 missing/invalid ages are filled using the median age within that city via `groupby(...).transform(...)`; signup dates are parsed as dates. Final city counts are Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191 and Mumbai 184. Final gender counts are female 610, male 560 and unknown 30. There are no missing ages; all are between 18 and 60.
- **Orders, in the required order:** timestamps are parsed; 119 exact duplicate rows are removed; records are sorted by timestamp and only the latest row for each order ID is kept (leaving 12,309 orders and removing the earlier pending versions); order amounts are converted to numeric and 112 invalid values remaining after deduplication are replaced with `round(price_inr × quantity × (1 − discount_pct / 100), 2)` (the raw audit finds 120 non-numeric entries before deduplication); 105 negative delivery durations remaining after deduplication are set to missing (106 occur in the raw audit); 774 missing durations for completed orders are filled with the median for that product category among completed orders; cancelled orders keep missing delivery durations because they were not delivered.
- **Storage:** the three cleaned CSV files are saved under `data/clean/`; SQLite database `data/instamart.db` contains `customers`, `products` and `orders`. The `order_ts` values are stored as `YYYY-MM-DD HH:MM:SS` text.

The cleaned orders contain 11,110 completed and 1,199 cancelled rows, no pending rows, no missing order amounts, no missing delivery duration among completed orders, and 1,199 missing delivery durations among cancelled orders. Completed-order revenue is INR 2,998,668.80. The validation assertions in `src/part_a.py` print exactly `ALL CHECKPOINTS PASSED` when they hold.

### SQL analysis

The seven executable queries are in `sql/queries.sql`; `src/part_a.py` executes them against SQLite using `pandas.read_sql_query`.

| Query | Result |
|---|---:|
| Q1 — completed orders above INR 500, customer total above INR 8,000 | 30 customers |
| Q2 — city × product-category revenue | 30 rows; INR 2,998,668.80 total |
| Q3 — products priced above average catalogue price | 11 products |
| Q4 — customers with no completed order | 25 customers |
| Q5 — customers above average completed spend | 550 customers; average customer spend INR 2,552.06 |
| Q6 — RANK, DENSE_RANK and ROW_NUMBER by city | 1,175 customers; 8 rows have RANK different from ROW_NUMBER |
| Q7 — daily revenue with running total | 183 days; final running total INR 2,998,668.80 |

The largest Q2 cell is Bengaluru / Electronics at INR 233,909.45. Q6's top customer in each city is Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), and Pune C0426 (INR 10,377.10). RANK can leave gaps after tied values, while ROW_NUMBER assigns a unique sequential number to every row; therefore ties can make the two rankings differ.

### pandas cross-check and exploratory analysis

A pandas `pivot_table(..., margins=True)` reproduces the SQL city/category revenue total exactly: INR 2,998,668.80. City summaries (completed orders) include Bengaluru INR 523,643.40 / mean INR 276.47 / 1,894 orders and Mumbai INR 479,058.50 / mean INR 297.00 / 1,613 orders.

Completed-order amount has mean INR 269.91, median INR 135.00 and skewness 4.61, indicating a strongly right-skewed distribution. Q1 is INR 64.80 and Q3 is INR 297.00; the 1.5×IQR fences are −INR 283.50 and INR 645.30. There are no lower-fence outliers and 897 upper-fence outliers. The largest absolute correlation is order amount with discount percentage (about −0.095); this is a very weak association and does not establish causation. The next-largest is order amount with quantity (about +0.087). The data describes completed synthetic grocery-delivery orders; unusually expensive baskets create the right tail, while discount and quantity have only weak linear relationships with amount.

## Part B

### RFM and customer segments

RFM is calculated from completed orders as of 2025-10-01 00:00:00. The table contains 1,175 customers. Recency median is 19 days (range 0–180), frequency mean is 9.46 (maximum 33), and monetary mean is INR 2,552.06 (maximum INR 14,766.55).

K-means uses StandardScaler and `KMeans(n_clusters=K, n_init=10, random_state=42)` for K=2 through 8:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.481 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected because it has the strongest silhouette score among the tested values and yields four useful customer groups. Inertia always decreases as K increases because adding clusters gives the model more freedom to fit the data; a lower inertia alone would always favour more clusters and can over-segment customers.

| Segment ID | Business name | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---:|---|---:|---:|---:|---:|---|
| 0 | Steady Shoppers | 436 | 28.86 | 3.77 | 1,057.62 | Promote weekly bundles and convenience replenishment. |
| 1 | Premium Basket Builders | 138 | 24.12 | 6.77 | 7,129.70 | Offer early access, premium support and curated cross-sells. |
| 2 | Frequent Loyalists | 419 | 8.77 | 19.40 | 3,410.02 | Reward repeat behaviour with loyalty benefits. |
| 3 | Occasional Essentials | 182 | 109.04 | 2.20 | 685.99 | Use low-cost reactivation offers and reminders. |

Premium is assigned only to the segment with the highest mean monetary value (segment 1). Cluster numbers are arbitrary labels and do not imply an order. PCA on scaled RFM explains 63.43% and 21.80% of variance in its first two components (85.23% combined). `outputs/customer_segments.csv` contains the required customer-level columns, including `segment_id`, `segment_name` and `is_premium`.

### Lapse prediction

The cutoff is 2025-09-01. The cohort contains 1,156 customers with at least one completed order before the cutoff; 443 (38.32%) have no completed order in September. The stratified split contains 924 training rows and 232 test rows. The all-not-lapsed baseline has 61.6% accuracy and positive-class F1 of 0.

Features use only pre-cutoff completed orders plus customer attributes: recency, frequency, monetary, average discount, cancellation rate among final-status orders, average delivery time, favourite product category, city, age, gender and tenure. No order dated 2025-09-01 or later entered the features, and task-3 segment labels were not used as features. Numeric scaling and one-hot encoding (drop-first) are inside a scikit-learn Pipeline and are fitted on training rows only; the test rows are only transformed. No feature column has missing values. The fitted training scaler's frequency mean is 8.115 versus the full-cohort mean of 8.029 because the scaler sees only the training subset.

Grid searches use the specified grids: LogisticRegression(max_iter=1000), C ∈ {0.01, 0.1, 1, 10, 100}, and RandomForestClassifier(n_estimators=100, random_state=42), max_depth ∈ {3, 5, 8}, min_samples_leaf ∈ {5, 10, 20}, both with 5-fold F1 scoring.

| Model | Best parameters | Best CV F1 | Test confusion matrix (TN, FP / FN, TP) | Precision | Recall | F1 |
|---|---|---:|---|---:|---:|---:|
| Logistic regression | C=1 | 0.657 | [[105, 38], [32, 57]] | 0.600 | 0.640 | 0.620 |
| Random forest | max_depth=5, min_samples_leaf=5 | 0.689 | [[110, 33], [30, 59]] | 0.641 | 0.663 | 0.652 |

The random forest is selected because it achieves the stronger test F1 (0.652). Missing a customer who is about to lapse can mean losing future orders, so recall matters; however, targeting loyal customers unnecessarily also spends retention budget. The selected model balances both costs better than the baseline and the tuned logistic regression in this run.

The oversampling experiment duplicates positive training examples only, using random_state=42; training counts move from 570 not-lapsed / 354 lapsed to 570 / 570. On the unchanged test set, logistic-regression recall rises from 0.640 to 0.865, precision changes from 0.600 to 0.583, and F1 rises from 0.620 to 0.697. Recall moves most, meaning more genuinely lapsing customers are caught, at the cost of slightly more false alarms.

The five largest random-forest importances are frequency (0.318), monetary (0.281), recency (0.113), cancellation rate (0.067) and average delivery time (0.051). Logistic-regression coefficient signs: recency is positive (higher recency increases predicted lapse risk), frequency is negative (more orders reduce risk), and monetary is positive in this fitted model (holding other features fixed, higher monetary is associated with higher predicted lapse odds in this sample). The selected model is saved and reloaded from `outputs/lapse_model.joblib`; test predictions are identical after reload. `outputs/retention_list.csv` contains the top 10 Premium-segment test customers by predicted probability; 0 of these 10 actually lapsed. The Premium flag is calculated from the whole-window RFM, including September, so this retention list is illustrative and is not a leakage-free evaluation.

## Part C

### Guardrails, redaction and prompts

Tickets are processed in file order. Empty or whitespace-only tickets are marked `skipped_empty`; tickets whose supplied `est_tokens(text)` exceeds 400 are marked `flagged_long`. Four tickets fall into each category, and the long tickets estimate to 734 tokens. Neither category reaches the model.

Redaction is applied in this exact order: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. Across all 164 tickets the substitutions are 55 emails, 91 phone numbers and 132 order IDs (278 total). The code re-applies all three regular expressions to each redacted text and asserts that no matches remain. The model's user message is exactly the redacted ticket text, with no added instruction or prefix. Both the zero-shot and few-shot system prompts list the four allowed categories and require JSON-only output; the few-shot prompt includes labelled examples including fraud.

### Retry and output validation

Every model call is retried on `TransientLLMError` with up to three retries and exponential backoff with jitter: `0.05 × 2**n × random.uniform(0.9, 1.15)`. Each retry is logged with ticket ID, retry number and wait. The full run produces 19 transient failures, all recovered. Replies are accepted as `classified_json` only when valid JSON has one of the four allowed categories. Non-JSON text is accepted as `classified_keyword` only when regex word extraction finds exactly one allowed category; all other outputs are `flagged_invalid_output` and sent to human review.

The full-run status counts are 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long` and 4 `skipped_empty`. Of the 156 eligible tickets, 147 are classified. The zero-shot run is correct on 126/147 (85.7%); its errors are 11 fraud tickets and 10 technical tickets predicted as billing. The few-shot run is correct on 147/147 (100%).

| Prompt | Classified | Accuracy | Total tokens | Tokens per classified ticket |
|---|---:|---:|---:|---:|
| Zero-shot | 147 | 85.7% | 11,116 | 75.62 |
| Few-shot | 147 | 100.0% | 16,576 | 112.76 |

The zero-shot confusion table (rows are true category; columns are predicted category) is:

| True \ Predicted | Billing | Delivery | Fraud | Technical |
|---|---:|---:|---:|---:|
| Billing | 38 | 0 | 0 | 0 |
| Delivery | 0 | 36 | 0 | 0 |
| Fraud | 11 | 0 | 23 | 0 |
| Technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 37.14 additional tokens per classified ticket (112.76 versus 75.62), but improves accuracy by 14.3 percentage points and removes all 21 zero-shot category errors. For a support workflow, that extra prompt cost is justified when misrouting a fraud or technical issue has a higher operational cost than the additional tokens.

### Budget and routing

The zero-shot budget demonstration uses TOKEN_BUDGET=6,000 and stops making model calls once the running total has reached the budget; 72 tickets are marked `budget_exhausted`. The final few-shot run uses TOKEN_BUDGET=100,000 and writes 164 rows to `outputs/triage_results.csv` with columns `ticket_id`, `customer_id`, `status`, `category`, `segment_name`, `priority`, `route` and `tokens_used`.

`skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` go to `human_review` at NORMAL priority. Valid fraud classifications go to `security_team` at HIGH priority; other valid tickets from Premium customers go to `premium_desk` at HIGH priority; remaining valid tickets go to `support_queue` at NORMAL priority. JSON and keyword categories are rule-validated; empty, long, invalid and budget-exhausted tickets are handed to humans rather than guessed.

## Part D

### Policy retrieval and grounded answers

The five policy files are split into 40-word chunks with 10-word overlap, producing 15 chunks: refund 4, delivery 3, escalation 3, payment 3 and privacy 2. `TfidfVectorizer()` fits on the chunk text; queries are transformed with the same vectorizer and ranked by cosine similarity. Term frequency (TF) measures how often a word appears in a chunk; inverse document frequency (IDF) downweights words common across many chunks. TF-IDF is lexical, so it can miss two sentences that mean the same thing but use different vocabulary. For the query `refund policy late delivery`, all three highest-ranked chunks come from `refund_policy.txt`.

The `answer(question)` function retrieves the top three chunks and sends a POLICY ASSISTANT system message plus a user message containing exactly the context and question format required. The model is instructed to answer only from the supplied passages or use the fallback. Evaluation results:

| Chunking | Expected document in top 3 (12 answerable) | Expected fact in answer (12 answerable) | Fallback for unanswerable (2) |
|---|---:|---:|---:|
| 10-word overlap | 12/12 | 12/12 | 2/2 |
| Zero overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, the questions about the refund delivery-time threshold and the failed-payment return time lose the expected fact because the relevant sentence is split across a chunk boundary. The overlapping chunks preserve enough surrounding context for the assistant to answer all 12 answerable questions.

### Tools and ReAct refund agent

`TOOLS_SPEC` describes five tools and their JSON-style parameters; `TOOL_MAP` dispatches each requested tool to its Python function. `get_order` reads the Part A SQLite database; `search_policy` reuses TF-IDF retrieval; `calculate_refund` returns the full order amount only for delivery over 45 minutes; `issue_refund` and `escalate_to_human` append JSON lines to `outputs/refund_ledger.jsonl`. Only completed orders qualify, and refunds above INR 1,000 are escalated rather than issued automatically. Simulated requests are treated as being inside the seven-day policy window.

| Scenario | Status | Refund / escalation amount (INR) | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 1,042 |
| S2 | REFUND | 135.00 | 5 | 1,042 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,029 |
| S4 | DENIED | 0.00 | 4 | 721 |
| S5 | DENIED | 0.00 | 3 | 424 |
| S6 | NOT_FOUND | 0.00 | 2 | 130 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,174 |

The S1 run uses five model calls. S7 reaches the six-iteration limit and ends in `FALLBACK_MAX_ITERATIONS`. The stop-safely demonstrations end with `FALLBACK_TOKEN_LIMIT` for S1 at token_limit=200 and `FALLBACK_TIMEOUT` for S1 at time_limit_seconds=0.

The crash-and-resume demonstration checkpoints after each tool observation. It simulates a crash immediately after the second tool execution, then resumes from the saved message list and completed steps. The exact tool sequence across the interrupted and resumed run is `get_order → search_policy → calculate_refund → issue_refund`; four tool executions occur, with none repeated. The resumed result is `REFUND` for INR 128.25. Checkpoint files are stored in `state/`.

**ReAct versus plan-and-execute (under 250 words).** ReAct calls the model repeatedly as each observation arrives: S1 used five calls and 1,042 tokens, S5 used three calls and 424 tokens, and S6 used two calls and 130 tokens. A plan-and-execute approach could ask for a complete tool sequence once, which may reduce model calls and prompt overhead on predictable cases. However, it must still branch safely when an observation is unexpected: a missing order should stop with NOT_FOUND, a cancelled order should be denied, and an above-limit refund should be escalated. For this workflow I prefer bounded ReAct because the tool observations determine the next safe action and the checkpoint mechanism supports recovery. The iteration, token and time limits prevent the agent from continuing indefinitely, while the tools—not the model—enforce the refund calculation and ledger action.

## Limitations and Next Steps

All records and policies are synthetic and are not real Instamart customer data or official policies. The offline mock LLM is deterministic and simplified; its accuracy is not evidence of production performance. The Premium flag comes from whole-window RFM, including September, so the retention list is illustrative and is not a leakage-free evaluation. Before production use, the workflow would need a genuinely time-separated evaluation, representative labelled tickets, stronger PII detection, access controls, model monitoring, calibrated risk thresholds, idempotent payment integrations, and mandatory human approval for high-risk actions.
