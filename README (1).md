# 📦 The COD Trust Score

**A CIBIL-style risk score for Cash-on-Delivery (COD) orders in Indian e-commerce.**

A storytelling Streamlit dashboard that reads a synthetic-but-realistic dataset of online
orders and predicts which COD shoppers a brand can safely deliver to — and which should
be asked to pay online instead.

> Built for the **Data Analytics – MGB** project-based learning module.

---

## The idea in one paragraph

In India, about **60–65% of online orders are Cash on Delivery**, and roughly **1 in 3 COD
parcels comes back** — the brand pays shipping both ways and earns nothing
(*Return-to-Origin*, or RTO). Killing COD is not an option; it's how Indians trust new
brands. So instead, this project gives every shopper a **300–900 COD Trust Score** —
exactly like a CIBIL credit score, but read from return history instead of repayment
history — and sorts them into three tiers:

| Score band | Tier | What the brand does |
|---|---|---|
| **720–900** | Free COD | offer COD freely |
| **580–719** | COD + fee | offer COD with a small ₹50 fee (risk-priced, like a higher interest rate) |
| **300–579** | Prepaid-only | switch COD off, ask the shopper to pay online |

---

## How the dashboard reads — the 10-chapter story

1. **📦 Overview** — the problem: what COD returns cost Indian D2C brands (with industry sources)
2. **🧾 Data** — the synthetic store: what one order looks like, why synthetic
3. **🧹 Data Preparation** — the messy export, fixed step by step, every change logged
4. **📊 Descriptive Analytics** — who sends parcels back? (interactive lens)
5. **🔍 Diagnostic Analytics** — Chi-square + Cramér's V on every clue
6. **🎯 Classification** — five classifiers, 5-fold stratified CV, the calibrated one becomes the score
7. **⚖️ Prescriptive · Verdict** — interactive policy sliders, ₹ trade-off math, limitations
8. **🧩 Clustering** — K-Means (elbow + silhouette + interactive 3D) and a Gaussian-Mixture latent-class model
9. **💰 Regression** — VIF check, then Linear / Ridge / Lasso / Decision Tree with a live λ dial
10. **🔗 Association Rules** — Apriori with support / confidence / lift sliders on risky trait combos

Chapters **1–5** are the *individual* phase deliverable. Chapters **6–10** are the *group*
phase extension, covering classification, clustering, regression, and association rule mining.
The whole dashboard runs as one app.

---

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501).

## Deploy free on Streamlit Community Cloud

1. Push this whole folder to a new GitHub repo (`app.py`, `requirements.txt`,
   `cod_orders.csv`, `generate_data.py`, `README.md`, and the `.streamlit/` folder).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app** → pick your repo
   and branch, set the main file to `app.py`.
3. Deploy. Your live URL will be `https://<your-name>-<repo>.streamlit.app`.

---

## Files in this repo

| File | What it is |
|---|---|
| `app.py` | The full 7-chapter Streamlit dashboard |
| `cod_orders.csv` | The synthetic dataset (7,235 messy rows, ~2,420 shoppers) |
| `generate_data.py` | Regenerates the dataset; prints a validation report |
| `requirements.txt` | Python dependencies (pandas pinned `<3.0` — pandas 3's arrow-backed strings are still unstable with Streamlit's pyarrow serializer) |
| `.streamlit/config.toml` | Light, minimalist teal theme |
| `README.md` | This file |

To regenerate the dataset with different parameters:

```bash
python generate_data.py
```

---

## Method notes (the honest version)

- **Data:** 7,200 orders across 2,600 customers, 18 states, 7 categories. Every shopper
  has a hidden "reliability" trait that drives both their past returns and their current
  RTO probability — this is what makes history a learnable signal (the CIBIL mechanic).
- **Messiness on purpose:** the raw CSV has 34 duplicate rows, `Rs`/`INR`/comma values,
  `%` discounts (some impossibly >100%), five case variants for each city tier, and 7%
  missing addresses — all caught and cleaned in Chapter 3.
- **Classification:** Logistic Regression, KNN, Decision Tree, Random Forest, Gradient
  Boosting — all trained only on **COD orders** with features knowable *before* dispatch
  (no leakage). Each model has a stated role (LogReg = interpretable baseline, KNN =
  expected curse-of-dimensionality victim, DT = readable but overfits, RF = variance fix,
  GB = accuracy contender) and all five are judged on **5-fold stratified
  cross-validation, reported as mean ± std** — so real differences can be told apart
  from split luck. ROC-AUC sits in the **0.72–0.75** band — believable, not magical.
  The naive majority-class baseline is shown next to the table so accuracy can't flatter.
- **Why Gradient Boosting builds the score:** Logistic Regression with balanced class
  weights is the AUC leader, but its predicted risks run hot (mean predicted ≈ 47% vs
  actual 35%) — that mis-calibration would wrongly push ~39% of shoppers to prepaid-only.
  Gradient Boosting's predicted risk almost exactly matches the real rate. **A score must
  mean what it says**, so calibration wins over a tenth of a point of AUC.
- **Score:** `score = 900 − probability × 600`, computed via 5-fold cross-validated
  predictions so no order is scored by a model that already saw it.
- **Clustering:** customers are aggregated (orders, avg value, avg discount, %COD, return
  rate) and grouped with **K-Means** — K chosen via elbow + silhouette (both agree the data
  has 2 natural shopper types). A **Gaussian Mixture Model** gives the same population soft,
  probabilistic membership — the numeric analogue of Latent Class Analysis. A 3D PCA
  projection (Plotly) lets you rotate and inspect the separation interactively.
- **Regression:** Linear, Ridge, Lasso, and a Decision Tree Regressor predict **OrderValue**
  (₹) instead of the RTO outcome — R² lands around 0.5, mostly explained by product category,
  a useful and honest (non-magical) result for a different question: what drives basket size,
  not who returns. **Regularization is motivated, not ritual:** a VIF check runs first
  (including the engineered PriorOrders/PriorRTORate/FirstTime cousins); since
  multicollinearity is mild, Ridge/Lasso are framed as insurance and verified live with an
  interactive **λ dial** plus full **shrinkage-path plots** (watch Lasso zero the weak
  coefficients one by one).
- **Association rule mining:** COD orders are turned into "baskets" of categorical traits
  (tier, category, address quality, device, discount band, first-time flag, outcome) and
  mined with **mlxtend's Apriori** algorithm. Three interactive dials — **support,
  confidence, and lift** — plus a Returned/Delivered business lens let you tighten the
  thresholds live until only the strongest rules survive; every bubble in the rule
  landscape is hoverable to read its IF → THEN out loud.
- **Limitations, stated up front (Verdict chapter):** synthetic-data circularity (history
  is *built* to be predictive here — real data will be noisier), predictors ≠ causes (the
  honest next step for any lever is an A/B test), the ₹ policy math rests on assumed
  abandonment/acceptance rates exposed as dials, no time dimension, and fairness needs
  live monitoring because geography can proxy income.

---

## The honesty caveat

A low COD Score is **not** a verdict that a shopper is dishonest or "bad". It only means
this particular order is risky to ship on cash — so the brand asks for prepayment instead.
The score adjusts the **offer**, never judges the **person**. Used carelessly it could
unfairly shut out whole towns or income groups; a real deployment must monitor for that
and keep a prepaid path open for everyone.

---

## Built with

[Streamlit](https://streamlit.io) · [scikit-learn](https://scikit-learn.org) ·
[pandas](https://pandas.pydata.org) · [matplotlib](https://matplotlib.org) ·
[scipy](https://scipy.org) · [Plotly](https://plotly.com/python) ·
[mlxtend](https://rasbt.github.io/mlxtend)
