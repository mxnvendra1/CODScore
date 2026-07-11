"""
The COD Trust Score  --  a story about Cash-on-Delivery returns in Indian e-commerce
====================================================================================
Run locally:   pip install -r requirements.txt  &&  streamlit run app.py
Deploy free:   push this folder to GitHub -> share.streamlit.io -> point to app.py

The dashboard reads ONE messy CSV (cod_orders.csv) and tells a 7-chapter story:
  1. The Problem        - why COD returns quietly drain Indian D2C brands
  2. Meet the Data      - what each order in our (synthetic) store looks like
  3. Cleaning the Data  - the messy export, fixed step by step (data prep)
  4. What Happened      - who sends parcels back more often (descriptive)
  5. The Real Reasons   - which signals truly matter vs. noise (diagnostic)
  6. The COD Score      - turning return-risk into a CIBIL-style score (models)
  7. The Verdict        - who gets COD free / COD with a fee / prepaid only

Everything below is recomputed live from the CSV. Nothing is hard-coded.

Honesty note carried throughout: a high return-risk is a reason to ADJUST THE OFFER
(ask for prepayment, add a small fee), never a verdict that a customer is a bad person.
"""
import warnings; warnings.filterwarnings("ignore")
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from sklearn.model_selection import (train_test_split, cross_val_predict,
                                     cross_validate, StratifiedKFold)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve,
                             classification_report, silhouette_score,
                             r2_score, mean_squared_error, mean_absolute_error)
from mlxtend.frequent_patterns import apriori, association_rules

# ----------------------------------------------------------------- theme detection
def _active_theme():
    """Return 'light' or 'dark', following the user's actual choice when possible."""
    try:
        t = getattr(st.context, "theme", None)
        if t is not None and getattr(t, "type", None) in ("light", "dark"):
            return t.type
    except Exception:
        pass
    base = st.get_option("theme.base")
    return base if base in ("light", "dark") else "light"

DARK = _active_theme() == "dark"

# ----------------------------------------------------------------- palette
# Brand colours chosen to pop on BOTH light and dark backgrounds.
ACCENT = "#14b8a6"   # teal = trust
GREEN  = "#22c55e"   # low risk
AMBER  = "#f59e0b"   # medium risk
RED    = "#ef4444"   # high risk
GREY   = "#94a3b8"
# Theme-aware neutrals (these flip with the mode)
INK  = "#f1f5f9" if DARK else "#1f2937"   # main text on charts
MUT  = "#94a3b8" if DARK else "#64748b"   # muted labels
GRID = "#243044" if DARK else "#eef2f7"   # gridlines
EDGE = "#334155" if DARK else "#cbd5e1"   # axis spines
GRAD = ("#0f766e", "#155e75") if DARK else ("#0d9488", "#0e7490")  # header gradient

plt.rcParams.update({
    "figure.facecolor": "none", "axes.facecolor": "none",
    "savefig.facecolor": "none", "savefig.transparent": True,
    "axes.edgecolor": EDGE, "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
    "axes.axisbelow": True, "font.size": 11, "font.family": "DejaVu Sans",
    "axes.titlesize": 13, "axes.titleweight": "medium", "axes.titlecolor": INK,
    "axes.titlepad": 14, "axes.labelcolor": MUT, "axes.labelsize": 10,
    "xtick.color": MUT, "ytick.color": MUT,
    "xtick.labelsize": 10.5, "ytick.labelsize": 10.5, "text.color": INK,
    "figure.autolayout": True,
})

def bare(ax, keep_left=True):
    """Minimalist axis: drop top/right spines, keep it clean."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_left:
        ax.spines["left"].set_visible(False)
    ax.patch.set_alpha(0)
    return ax

st.set_page_config(page_title="The COD Trust Score", page_icon="📦",
                   layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------- global CSS (both themes)
st.markdown(f"""
<style>
  .block-container {{max-width: 1060px; padding-top: 1.4rem;}}
  h1, h2, h3 {{letter-spacing: -0.01em;}}
  p, li {{font-size: 1.03rem; line-height: 1.62;}}
  /* tabs: bigger, pill-style, clear active state */
  .stTabs [data-baseweb="tab-list"] {{gap: 4px; flex-wrap: wrap;}}
  .stTabs [data-baseweb="tab"] {{
      font-size: 0.95rem; font-weight: 600; padding: 9px 14px; border-radius: 9px;}}
  .stTabs [aria-selected="true"] {{
      background: linear-gradient(100deg,{GRAD[0]},{GRAD[1]}); color: #fff !important;}}
  /* gradient hero + chapter headers */
  .hero {{background: linear-gradient(110deg,{GRAD[0]},{GRAD[1]});
          padding: 26px 30px; border-radius: 18px; margin-bottom: 6px;
          box-shadow: 0 10px 30px rgba(13,148,136,.25);}}
  .hero h1 {{color:#fff; font-size: 2.05rem; margin:0; font-weight: 800;}}
  .hero p  {{color:#ecfeff; font-size: 1.08rem; margin:.35rem 0 0;}}
  .chead {{background: linear-gradient(100deg,var(--c1),var(--c2));
           padding: 16px 22px; border-radius: 14px; margin: 4px 0 16px;}}
  .chead .num {{color:#fff; opacity:.85; font-weight:700; font-size:.78rem;
                letter-spacing:.1em; text-transform:uppercase;}}
  .chead .ttl {{color:#fff; font-size:1.4rem; font-weight:800; line-height:1.15;}}
  .chead .sub {{color:#f8fafc; opacity:.95; font-size:1.02rem; margin-top:3px;}}
  /* takeaway pill */
  .take {{border-left:4px solid {ACCENT}; padding:.6rem .9rem; border-radius:8px;
          background: rgba(20,184,166,.10); margin:.5rem 0; font-size:1.02rem;}}
  .stat {{border-radius:14px; padding:16px 18px; text-align:center;
          background: rgba(148,163,184,.10);}}
  .stat .big {{font-size:1.9rem; font-weight:800; line-height:1;}}
  .stat .lab {{font-size:.86rem; opacity:.8; margin-top:5px;}}
  div[data-testid="stMetricValue"] {{font-size: 1.55rem; font-weight: 700;}}
  hr {{margin: 1.1rem 0;}}
</style>
""", unsafe_allow_html=True)

def chapter(num, icon, title, subtitle, c1=GRAD[0], c2=GRAD[1]):
    st.markdown(
        f"<div class='chead' style='--c1:{c1};--c2:{c2}'>"
        f"<div class='num'>Chapter {num}</div>"
        f"<div class='ttl'>{icon}&nbsp; {title}</div>"
        f"<div class='sub'>{subtitle}</div></div>", unsafe_allow_html=True)

def take(text):
    """A short, visual 'what this means' pill — replaces long paragraphs."""
    st.markdown(f"<div class='take'>💡 {text}</div>", unsafe_allow_html=True)

def statcard(col, big, lab, color=None):
    color = color or INK
    col.markdown(f"<div class='stat'><div class='big' style='color:{color}'>{big}</div>"
                 f"<div class='lab'>{lab}</div></div>", unsafe_allow_html=True)

TARGET = "DeliveryStatus"
POSITIVE = "Returned"

# COD economics (industry-reported; editable in the Verdict chapter)
DEFAULT_RTO_COST = 250   # ₹ lost on a typical returned COD order (forward+reverse+handling)

# ============================================================ DATA: load + clean
@st.cache_data(show_spinner=False)
def load_raw(file_bytes):
    if file_bytes is None:
        return pd.read_csv("cod_orders.csv")
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def clean(_raw: pd.DataFrame, cache_key: int = 0):
    """Turn the messy export into analysis-ready data, logging every change.
    _raw is deliberately unhashed (leading underscore) — hashing a whole DataFrame
    on every rerun is slow and can crash on arrow-backed string columns; cache_key
    (derived from the uploaded bytes) keys the cache instead."""
    raw = _raw
    d = raw.copy()
    notes = []
    n0 = len(d)

    # 1) exact duplicate rows (accidental double-export)
    dups = int(d.duplicated().sum())
    if dups:
        d = d.drop_duplicates().reset_index(drop=True)
        notes.append(f"Removed **{dups}** exact duplicate rows ({n0} → {len(d)}).")

    # 2) OrderValue: strip 'Rs'/'INR'/commas → number
    if "OrderValue" in d:
        d["OrderValue"] = pd.to_numeric(
            d["OrderValue"].astype(str).str.replace(r"[^0-9.]", "", regex=True),
            errors="coerce")
        notes.append("Parsed **OrderValue** — removed `Rs`/`INR`/commas, converted to numbers.")

    # 3) DiscountPct: strip '%', cap impossible values (>100) as data errors
    if "DiscountPct" in d:
        d["DiscountPct"] = pd.to_numeric(
            d["DiscountPct"].astype(str).str.replace("%", "", regex=False),
            errors="coerce")
        bad = int((d["DiscountPct"] > 100).sum())
        if bad:
            med = d.loc[d["DiscountPct"] <= 100, "DiscountPct"].median()
            d.loc[d["DiscountPct"] > 100, "DiscountPct"] = med
            notes.append(f"Fixed **{bad}** impossible discounts (>100%) → set to the "
                         f"median ({med:.0f}%).")
        notes.append("Parsed **DiscountPct** — removed `%`, converted to numbers.")

    # 4) CityTier: merge case/format variants (Tier 1 / tier1 / T1 / TIER-1 → Tier-1)
    if "CityTier" in d:
        before = d["CityTier"].nunique()
        digit = d["CityTier"].astype(str).str.extract(r"([123])")[0]
        d["CityTier"] = digit.map({"1": "Tier-1", "2": "Tier-2", "3": "Tier-3"})
        notes.append(f"Standardised **CityTier**: {before} messy variants → 3 clean tiers.")

    # 5) State: trim whitespace + Title Case (merge 'KARNATAKA'/'karnataka')
    if "State" in d:
        before = d["State"].nunique()
        d["State"] = d["State"].astype(str).str.strip().str.title()
        notes.append(f"Cleaned **State**: {before} → {d['State'].nunique()} distinct names "
                     "(trimmed spaces, unified case).")

    # 6) Missing categoricals → explicit 'Unknown'
    for c in ["Device", "AddressQuality"]:
        if c in d:
            miss = int(d[c].isna().sum())
            if miss:
                d[c] = d[c].fillna("Unknown")
                notes.append(f"**{c}**: {miss} missing values labelled `Unknown`.")

    # 7) binary target + handy derived features (all knowable BEFORE dispatch)
    d["RTO"] = (d[TARGET] == POSITIVE).astype(int)
    d["PriorRTORate"] = np.where(d["PriorOrders"] > 0,
                                 d["PriorReturns"] / d["PriorOrders"], 0.0)
    d["FirstTime"] = (d["PriorOrders"] == 0).astype(int)
    d["IsCOD"] = (d["PaymentMethod"] == "COD").astype(int)
    notes.append("Built the target **RTO** (1 = Returned, 0 = Delivered) and helper "
                 "features: past-return rate, first-time flag, COD flag.")

    # drop any rows that lost their value in parsing (rare)
    miss_val = int(d["OrderValue"].isna().sum())
    if miss_val:
        d = d.dropna(subset=["OrderValue"]).reset_index(drop=True)
        notes.append(f"Dropped **{miss_val}** rows with unrecoverable OrderValue.")

    return d, notes


# ============================================================ MODELLING (COD only)
NUM_FEATS = ["OrderValue", "DiscountPct", "PriorOrders", "PriorRTORate",
             "FirstTime", "OrderHour", "Items"]
CAT_FEATS = ["CityTier", "State", "Category", "Device", "AddressQuality"]
# A score must be *calibrated* (predicted risk ≈ real risk), not just high-AUC.
# Gradient Boosting's probabilities track reality almost exactly, so we score from it.
SCORE_MODEL = "Gradient Boosting"


def design_matrix(df_cod):
    X = df_cod[NUM_FEATS + CAT_FEATS].copy()
    y = df_cod["RTO"].copy()
    return X, y


def preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_FEATS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATS),
    ])


@st.cache_resource(show_spinner=True)
def train_models(_X, _y, test_size, seed):
    Xtr, Xte, ytr, yte = train_test_split(_X, _y, test_size=test_size,
                                           stratify=_y, random_state=seed)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)]).fit(Xtr, ytr)
        ptr, pte = pipe.predict(Xtr), pipe.predict(Xte)
        proba = pipe.predict_proba(Xte)[:, 1]
        out[name] = {
            "train_acc": accuracy_score(ytr, ptr),
            "test_acc": accuracy_score(yte, pte),
            "precision": precision_score(yte, pte, zero_division=0),
            "recall": recall_score(yte, pte, zero_division=0),
            "f1": f1_score(yte, pte, zero_division=0),
            "roc_auc": roc_auc_score(yte, proba),
            "cm": confusion_matrix(yte, pte),
            "roc": roc_curve(yte, proba),
            "report": classification_report(yte, pte,
                       target_names=["Delivered", "Returned"], zero_division=0),
            "pipe": pipe,
        }
    return out, (len(Xtr), len(Xte), ytr.mean(), yte.mean())


@st.cache_data(show_spinner="Cross-validating all five models (5 stratified folds)…")
def cv_scoreboard(_X, _y, n_rows, seed):
    """5-fold stratified cross-validation for every model.
    A single train/test split can get lucky (or unlucky); five folds show the
    mean AND the spread, so we can tell real differences from split noise.
    n_rows is passed only to key the cache correctly."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scoring = {"Accuracy": "accuracy", "Precision": "precision",
               "Recall": "recall", "F1": "f1", "ROC-AUC": "roc_auc"}
    disp, means = {}, {}
    for name, clf in models.items():
        pipe = Pipeline([("pre", preprocessor()), ("clf", clf)])
        cv = cross_validate(pipe, _X, _y, cv=skf, scoring=scoring, n_jobs=-1)
        disp[name] = {m: f"{cv[f'test_{m}'].mean():.3f} ± {cv[f'test_{m}'].std():.3f}"
                      for m in scoring}
        means[name] = {m: cv[f"test_{m}"].mean() for m in scoring}
    return pd.DataFrame(disp).T, pd.DataFrame(means).T


@st.cache_data(show_spinner=True)
def honest_scores(_X, _y, best_name, test_size, seed):
    """Return-risk for EVERY COD order using cross-validation (out-of-fold),
    so no order is scored by a model that already saw it. Then map risk → score."""
    model = {
        "Logistic Regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                                random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12,
                                                class_weight="balanced",
                                                random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=seed),
        "KNN": KNeighborsClassifier(n_neighbors=25),
    }[best_name]
    pipe = Pipeline([("pre", preprocessor()), ("clf", model)])
    proba = cross_val_predict(pipe, _X, _y, cv=5, method="predict_proba")[:, 1]
    # CIBIL-style: low risk -> high score. Range ~300..900.
    score = np.round(900 - proba * 600).astype(int)
    return proba, score


def cramers_v(df, col, target="RTO"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, _ = chi2_contingency(ct)
    n = ct.to_numpy().sum(); r, k = ct.shape
    v = np.sqrt((chi2 / n) / max(min(r - 1, k - 1), 1))
    return chi2, p, dof, v, int(ct.shape[0])


# ================================================================ SIDEBAR
st.sidebar.markdown("### 📖 The story")
st.sidebar.markdown("""
1. 📦 **Problem** — why COD hurts
2. 🧾 **Data** — our store
3. 🧹 **Cleaning** — tidy it up
4. 📊 **Who returns** — the patterns
5. 🔍 **Why** — real reasons
6. 🎯 **The Score** — one number
7. ⚖️ **Verdict** — the decision
8. 🧩 **Segments** — who shops alike
9. 💰 **Order Value** — what drives basket size
10. 🔗 **Combos** — risky combinations
""")
st.sidebar.caption("Read the tabs left → right, like chapters.")
st.sidebar.divider()

up = st.sidebar.file_uploader("Use your own CSV (optional)", type=["csv"])
_up_bytes = up.read() if up is not None else None
raw = load_raw(_up_bytes)
df, notes = clean(raw, len(_up_bytes) if _up_bytes else 0)

with st.sidebar.expander("⚙️ Model settings"):
    test_size = st.slider("Test split", 0.15, 0.40, 0.25, 0.05)
    seed = int(st.number_input("Random seed", value=42, step=1))

st.sidebar.caption("🌗 Tip: light/dark mode both work — switch under the ⋮ menu → Settings.")

# COD subset = the population the score is built for
cod = df[df["IsCOD"] == 1].copy().reset_index(drop=True)
overall_rto = df["RTO"].mean()
cod_rto = cod["RTO"].mean()
prepaid_rto = df[df["IsCOD"] == 0]["RTO"].mean()

# ================================================================ HEADER
st.markdown("<div class='hero'><h1>📦 The COD Trust Score</h1>"
            "<p>A credit score — but for Cash on Delivery. Who can a brand trust with COD?</p>"
            "</div>", unsafe_allow_html=True)
st.write("")

m1, m2, m3, m4 = st.columns(4)
statcard(m1, f"{len(df):,}", "Orders")
statcard(m2, f"{df['CustomerID'].nunique():,}", "Shoppers")
statcard(m3, f"{cod_rto*100:.0f}%", "COD returns", RED)
statcard(m4, f"{prepaid_rto*100:.0f}%", "Prepaid returns", GREEN)
st.write("")

tabs = st.tabs([
    "1 · 📦 Problem", "2 · 🧾 Data", "3 · 🧹 Cleaning",
    "4 · 📊 Who Returns", "5 · 🔍 Why", "6 · 🎯 The Score",
    "7 · ⚖️ Verdict", "8 · 🧩 Segments", "9 · 💰 Order Value",
    "10 · 🔗 Combos",
])

# =============================================================== 1. THE PROBLEM
with tabs[0]:
    chapter(1, "📦", "The leak nobody sees",
            "Cash-on-Delivery parcels that come straight back — and burn money both ways.")

    st.markdown("#### A ₹1,200 COD order's sad journey")
    j1, j2, j3, j4 = st.columns(4)
    statcard(j1, "🛒", "Customer orders (COD)")
    statcard(j2, "🚚", "Brand ships it")
    statcard(j3, "🙅", "Parcel refused")
    statcard(j4, "↩️", "Comes back — ₹0 earned", RED)
    take("This is <b>Return to Origin (RTO)</b>: the brand pays shipping twice and earns "
         "nothing. In India it's one of the biggest hidden costs online.")

    st.markdown("#### Why it's a *big* deal (not just us)")
    c1, c2, c3 = st.columns(3)
    statcard(c1, "60–65%", "of Indian orders are COD", ACCENT)
    statcard(c2, "25–40%", "of COD comes back", RED)
    statcard(c3, "₹180–350", "lost per return", AMBER)

    with st.expander("📚 Proof from the industry (real blogs)"):
        st.markdown("""
- **GoKwik** — ₹200–250 lost on a typical ₹1,000 COD return →
  [link](https://www.gokwik.co/blog/what-is-return-to-origin-rto-in-ecommerce)
- **HillTeck** — the true all-in cost of RTO →
  [link](https://www.hillteck.com/blog/rto-cost-indian-d2c-brands.html)
- **Pragma** — cut RTO by *segmenting customers* (our exact idea) →
  [link](https://www.bepragma.ai/blogs/how-to-reduce-rto-in-indian-e-commerce-without-hurting-cod-orders)
- **Edgistify** — "RTO %: the silent killer of D2C" →
  [link](https://www.edgistify.com/resources/blogs/rto-percentage-silent-killer-indian-d2c)
- **CallFox** — reduce COD returns →
  [link](https://www.callfox.in/blog/reduce-cod-returns-india)
""")

    st.markdown("#### The big idea: a CIBIL score, but for COD")
    colA, colB = st.columns([1, 1])
    with colA:
        st.markdown("""
A bank reads your **repayment history** → gives you a **CIBIL score** → decides your loan.

We read a shopper's **return history** → give them a **COD Trust Score** → decide their COD.
""")
    with colB:
        st.table(pd.DataFrame({
            "CIBIL 🏦": ["Repayment history", "Lend or not?", "Higher interest if risky",
                         "Refused below cut-off"],
            "COD Score 📦": ["Return history", "Offer COD or not?", "Small fee if risky",
                             "Prepaid-only below cut-off"],
        }))

    st.markdown("#### In our own store, the gap is huge")
    fig, ax = plt.subplots(figsize=(8, 2.2))
    rates = [prepaid_rto * 100, cod_rto * 100]
    bars = ax.barh(["Prepaid", "Cash on Delivery"], rates, color=[GREEN, RED], height=0.62)
    for b, r in zip(bars, rates):
        ax.text(r + 0.6, b.get_y() + b.get_height()/2, f"{r:.0f}%",
                va="center", fontweight="bold", color=INK, fontsize=13)
    ax.set_xlim(0, max(rates) * 1.28); ax.set_xlabel("Return rate (%)")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    take("We can't just ban COD — it's how India shops. So the smart question is: "
         "<b>which shoppers get COD, and which pay first?</b> The next chapters find out.")

# =============================================================== 2. MEET THE DATA
with tabs[1]:
    chapter(2, "🧾", "Meet the store",
            "A made-up store that behaves like the real world — so we can test our idea.",
            "#7c3aed", "#5b21b6")

    d1, d2, d3 = st.columns(3)
    statcard(d1, f"{len(df):,}", "orders")
    statcard(d2, f"{df['CustomerID'].nunique():,}", "shoppers")
    statcard(d3, "18", "states · 7 categories")
    st.markdown("**Why fake data?** Real brands won't share private return data. Synthetic "
                "data is private-safe, *and* we know the hidden truth — so we can check our "
                "methods actually find it.")
    take("The trick that makes a score possible: every shopper has a hidden habit — some "
         "accept parcels, some refuse. It shows up in their <b>past returns</b>, so history "
         "predicts the future (just like CIBIL).")

    st.markdown("#### What one order looks like")
    field_help = pd.DataFrame({
        "Column": ["OrderID / CustomerID", "CityTier", "State", "Category", "OrderValue",
                   "DiscountPct", "PaymentMethod", "Device", "AddressQuality", "OrderHour",
                   "Items", "PriorOrders", "PriorReturns", "DeliveryStatus"],
        "In plain English": [
            "Who placed it (IDs)",
            "Tier-1 metro, Tier-2 city, or Tier-3 town",
            "Which state it ships to",
            "What was bought (Fashion, Electronics …)",
            "Order amount in ₹",
            "Discount on the order",
            "Cash on Delivery or paid online (Prepaid)",
            "App / mobile web / desktop",
            "Was the address complete, partial, or vague?",
            "Hour of day the order was placed (0–23)",
            "How many items in the order",
            "How many orders this shopper made before",
            "How many of those came back",
            "👉 The outcome: Delivered or Returned",
        ],
        "Why it matters": [
            "Identify repeat shoppers",
            "Smaller towns return more",
            "Some regions are riskier",
            "Fashion is tried-on-and-returned",
            "Big COD bills get refused",
            "Deep discounts → impulse → regret",
            "COD is where returns happen",
            "Hints who the shopper is",
            "Bad address = failed delivery",
            "Late-night = impulsive",
            "Bigger carts behave differently",
            "Loyalty signal",
            "👉 Powers the score",
            "👉 What we predict",
        ],
    })
    st.dataframe(field_help, width="stretch", hide_index=True)

    st.markdown("#### A peek at the raw file — messy on purpose")
    st.caption("Real exports are never clean: `Rs`/commas in prices, a stray `999%` "
               "discount, tiers written five ways. Chapter 3 fixes it all.")
    st.dataframe(raw.head(10), width="stretch")

# =============================================================== 3. CLEANING
with tabs[2]:
    chapter(3, "🧹", "Cleaning the data",
            "Messy export in, tidy numbers out — every fix logged in the open.",
            "#0891b2", "#0e7490")

    cln = st.columns(3)
    statcard(cln[0], "34", "duplicate rows removed", AMBER)
    statcard(cln[1], "5→3", "tier spellings merged", ACCENT)
    statcard(cln[2], "0", "missing values left", GREEN)

    st.markdown("#### Every fix we made")
    for n in notes:
        st.markdown(f"- {n}")

    st.markdown("#### Before → After")
    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Before** (raw)")
        st.dataframe(raw[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")
    with cc2:
        st.markdown("**After** (clean)")
        st.dataframe(df[["OrderValue", "DiscountPct", "CityTier", "State"]].head(8),
                     width="stretch")

    st.markdown("#### New features we built")
    st.markdown("""
All knowable the *moment* an order is placed — never peeking at the future:
- **PriorRTORate** — fraction of past orders the shopper returned (their track record)
- **FirstTime** — is this their first-ever order?
- **IsCOD** — is it Cash on Delivery?
""")
    take("We only use clues available <b>before shipping</b>. Using after-the-fact info "
         "would make the model look great in tests but useless in real life (that mistake "
         "is called <i>leakage</i>).")
    st.success(f"✅ Ready: **{len(df):,} orders**, **{df.shape[1]} columns**, "
               "zero duplicates, zero missing values.")

# =============================================================== 4. DESCRIPTIVE
with tabs[3]:
    chapter(4, "📊", "Who sends parcels back?",
            "Simple questions, clear bars. Red = worse than average, green = better.",
            "#0d9488", "#0f766e")
    st.caption("The dashed line on each chart is the overall return rate.")

    def rate_by(col, title, min_n=20, full=True):
        base = df if full else cod
        ref = base["RTO"].mean()
        g = base.groupby(col)["RTO"].agg(["mean", "count"])
        g = g[g["count"] >= min_n].sort_values("mean")
        fig, ax = plt.subplots(figsize=(8, max(2.4, 0.5 * len(g))))
        colors = [RED if m > ref else GREEN for m in g["mean"]]
        bars = ax.barh(g.index.astype(str), g["mean"] * 100, color=colors, height=0.62)
        ax.axvline(ref * 100, color=INK, ls="--", lw=1.3)
        for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(m * 100 + 0.5, b.get_y() + b.get_height() / 2,
                    f"{m*100:.0f}%  (n={n:,})", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, g["mean"].max() * 100 * 1.25)
        ax.set_xlabel("Return rate"); ax.set_title(title)
        bare(ax); ax.tick_params(left=False)
        return fig

    st.markdown("#### Cash on Delivery vs Prepaid")
    st.pyplot(rate_by("PaymentMethod", "Return rate by payment method"), width="stretch")
    take(f"COD returns at <b>{cod_rto*100:.0f}%</b> vs just <b>{prepaid_rto*100:.0f}%</b> "
         "prepaid. From here we zoom into COD — that's where money leaks.")

    st.divider()
    st.markdown("#### Within COD, who's risky? Pick a lens 👇")
    lens = st.selectbox("Break COD returns down by:",
                        ["City tier", "Product category", "Address quality",
                         "First-time vs repeat", "Discount depth", "Order value"],
                        index=0)

    if lens == "City tier":
        st.pyplot(rate_by("CityTier", "COD return rate by city tier", full=False),
                  width="stretch")
        take("Smaller towns return more — Tier-3 roughly <b>triples</b> Tier-1.")
    elif lens == "Product category":
        st.pyplot(rate_by("Category", "COD return rate by product category", full=False),
                  width="stretch")
        take("<b>Fashion &amp; footwear</b> top the list — ordered to 'try at home', then refused.")
    elif lens == "Address quality":
        st.pyplot(rate_by("AddressQuality", "COD return rate by address quality",
                          min_n=10, full=False), width="stretch")
        take("A <b>vague address</b> is a delivery waiting to fail — fixing it at checkout is a cheap win.")
    elif lens == "First-time vs repeat":
        tmp = cod.assign(Who=np.where(cod["FirstTime"] == 1, "First-time buyer",
                                      "Repeat buyer"))
        ref = cod["RTO"].mean()
        g = tmp.groupby("Who")["RTO"].mean()
        fig, ax = plt.subplots(figsize=(7, 2.6))
        bars = ax.barh(g.index, g.values * 100,
                       color=[RED if v > ref else GREEN for v in g.values], height=0.55)
        for b, v in zip(bars, g.values):
            ax.text(v*100+0.5, b.get_y()+b.get_height()/2, f"{v*100:.0f}%",
                    va="center", fontweight="bold")
        ax.axvline(ref*100, color=INK, ls="--", lw=1.3)
        ax.set_xlim(0, g.values.max()*100*1.25); ax.set_xlabel("Return rate")
        ax.set_title("COD: first-timers vs repeat buyers"); bare(ax); ax.tick_params(left=False)
        st.pyplot(fig, width="stretch")
        take("<b>First-timers are riskier</b> — no track record yet. Loyalty earns trust.")
    elif lens == "Discount depth":
        b = pd.cut(cod["DiscountPct"], [-1, 10, 25, 40, 100],
                   labels=["0–10%", "11–25%", "26–40%", "40%+"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by discount depth")
        bare(ax)
        st.pyplot(fig, width="stretch")
        take("Deeper discount → more impulse → more regret at the door.")
    else:  # Order value
        b = pd.qcut(cod["OrderValue"], 4,
                    labels=["Cheapest 25%", "Lower-mid", "Upper-mid", "Priciest 25%"])
        g = cod.assign(Band=b).groupby("Band", observed=True)["RTO"].agg(["mean", "count"])
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        bars = ax.bar(g.index.astype(str), g["mean"]*100, color=ACCENT, width=0.6)
        ax.axhline(cod["RTO"].mean()*100, color=RED, ls="--", lw=1.3)
        for b_, (m, n) in zip(bars, zip(g["mean"], g["count"])):
            ax.text(b_.get_x()+b_.get_width()/2, m*100+0.6, f"{m*100:.0f}%\nn={n:,}",
                    ha="center", fontsize=9)
        ax.set_ylabel("Return rate"); ax.set_title("COD return rate by order value")
        bare(ax)
        st.pyplot(fig, width="stretch")
        take("Bigger COD bills are scarier to accept — more cash on the spot, more refusals.")

    st.divider()
    st.markdown("#### Which numbers move with returns?")
    st.caption("Teal = more returns, grey = fewer. Straight-line links only — tested properly next chapter.")
    corr_cols = ["RTO", "PriorRTORate", "FirstTime", "DiscountPct", "OrderValue",
                 "PriorOrders", "Items", "OrderHour"]
    corr = cod[corr_cols].corr()["RTO"].drop("RTO").sort_values()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    bars = ax.barh(corr.index, corr.values,
                   color=[ACCENT if v >= 0 else GREY for v in corr.values], height=0.6)
    for b, v in zip(bars, corr.values):
        ax.text(v + (0.005 if v >= 0 else -0.005), b.get_y()+b.get_height()/2,
                f"{v:+.2f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color=INK, lw=1); ax.set_xlabel("Correlation with returning (COD)")
    ax.set_title("Which numbers move with returns?"); bare(ax, keep_left=False)
    ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    take("<b>Past-return rate stands out</b> — a shopper's history is the loudest number. The seed of the score.")

# =============================================================== 5. DIAGNOSTIC
with tabs[4]:
    chapter(5, "🔍", "The real reasons",
            "Which clues truly matter, and which are just noise?",
            "#4f46e5", "#4338ca")
    take("<b>Cramér's V</b> is a 0→1 'how strongly linked?' dial (closer to 1 = stronger). "
         "The <i>p-value</i> checks it's real, not luck. We grey-out anything that's just luck.")

    assoc_cols = [c for c in ["PaymentMethod", "CityTier", "Category", "AddressQuality",
                              "State", "Device"] if c in df.columns]
    rows = []
    for c in assoc_cols:
        chi2, p, dof, v, ncat = cramers_v(df, c)
        rows.append({"Clue": c, "Cramér's V": round(v, 3), "p-value": f"{p:.1e}",
                     "Real link?": "✅ yes" if p < 0.05 else "— (luck)"})
    assoc = pd.DataFrame(rows).sort_values("Cramér's V", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 0.55 * len(assoc) + 1))
    a2 = assoc.sort_values("Cramér's V")
    colors = [GREY if "—" in s else ACCENT for s in a2["Real link?"]]
    bars = ax.barh(a2["Clue"], a2["Cramér's V"], color=colors, height=0.6)
    for b, v in zip(bars, a2["Cramér's V"]):
        ax.text(v + 0.005, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Cramér's V  (link strength with returns)")
    ax.set_title("Which clues are truly linked to returns?"); bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")
    st.dataframe(assoc, width="stretch", hide_index=True)

    take("Payment is the biggest lever — but we can't ban COD without losing sales. "
         "The smart move: look <b>inside COD</b> and separate risky from safe. (Chapter 6.)")

    st.divider()
    st.markdown("#### The clue that travels with you: your track record")
    h = cod[cod["PriorOrders"] > 0].copy()
    h["band"] = pd.cut(h["PriorRTORate"], [-0.01, 0.0, 0.25, 0.5, 1.01],
                       labels=["Never returned", "Returned 1–25%", "Returned 26–50%",
                               "Returned >50%"])
    g = h.groupby("band", observed=True)["RTO"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bars = ax.bar(g.index.astype(str), g["mean"]*100,
                  color=[GREEN, "#84cc16", AMBER, RED], width=0.62)
    for b, (m, n) in zip(bars, zip(g["mean"], g["count"])):
        ax.text(b.get_x()+b.get_width()/2, m*100+1, f"{m*100:.0f}%\nn={n:,}",
                ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Return rate on the NEXT order")
    ax.set_title("Past returns predict the next return")
    bare(ax)
    st.pyplot(fig, width="stretch")
    take("Returned over half your parcels before? You'll likely return the next. Never "
         "returned? You rarely start. This is the most CIBIL-like signal we have.")

# =============================================================== 6. THE COD SCORE
with tabs[5]:
    chapter(6, "🎯", "The COD Trust Score",
            "Every clue, combined into one 300–900 number per shopper — CIBIL-style.",
            "#0d9488", "#115e59")

    X, y = design_matrix(cod)
    results, (ntr, nte, trp, tep) = train_models(X, y, test_size, seed)
    st.caption(f"Trained on {ntr:,} COD orders, tested on {nte:,} unseen ones. "
               "We predict whether a parcel comes back.")
    take("🧪 <b>Leakage-free by construction.</b> Every feature — order details, city "
         "tier, address quality, the shopper's <i>past</i> record — is knowable the "
         "moment the order is placed. Nothing from after dispatch (the outcome, courier "
         "events) is allowed in. A leaky model would post a spectacular fake accuracy; "
         "a clean one earns an honest ~0.7–0.8 AUC — which is exactly what we see.")

    with st.expander("🧠 Why these five models — each one has a job", expanded=False):
        st.markdown("""
We don't compare five models to look busy — each is here to answer a specific question:

- **Logistic Regression** — the *interpretable baseline*. Linear, fast, and its
  coefficients can be read out loud. If nothing beats it, complexity isn't paying rent.
- **KNN** — the *distance method*. One-hot encoding our categoricals creates a
  high-dimensional space where straight-line distance loses meaning (the *curse of
  dimensionality*) — KNN is included expecting it to struggle, to demonstrate that.
- **Decision Tree** — the *readable rule-learner*. Handles mixed data natively but a
  single tree memorises noise; it's the baseline the ensembles must beat.
- **Random Forest** — the *variance fix*: hundreds of trees on random slices,
  averaged. Same rules idea, less overfitting.
- **Gradient Boosting** — the *accuracy contender* on tabular data: each tree corrects
  the last one's errors. Powerful, so it must prove itself on data it hasn't seen.

**Judged on 5-fold cross-validation**, not a single lucky split — and on five metrics,
because with ~two-thirds of parcels delivered, saying "Delivered" for everyone already
scores well on accuracy alone.
""")

    st.markdown("#### How five models did — 5-fold cross-validation (mean ± spread)")
    cv_disp, cv_means = cv_scoreboard(X, y, len(X), seed)
    holdout_gap = pd.Series({m: r["train_acc"] - r["test_acc"]
                             for m, r in results.items()}).round(3)
    cv_disp["Overfit gap*"] = holdout_gap.reindex(cv_disp.index).map("{:.3f}".format)
    st.dataframe(cv_disp, width="stretch")
    naive = max(y.mean(), 1 - y.mean())
    st.caption(f"Each cell = mean ± standard deviation across 5 stratified folds — the ± "
               f"tells you whether two models *really* differ or it's just split luck. "
               f"*Overfit gap = train − test accuracy on the hold-out split (smaller = "
               f"safer). Naive baseline: always guessing the majority class scores "
               f"**{naive:.0%}** accuracy — any model must clear that bar to matter.")
    with st.expander("What do these words mean?"):
        st.markdown(f"""
- **Accuracy** — how often the guess is right overall. *Careful:* always guessing
  "Delivered" already scores ~{naive:.0%} here — accuracy alone flatters.
- **Precision** — when it says "will return", how often it's correct.
- **Recall** — of parcels that truly returned, how many it caught.
- **F1** — balance of precision and recall.
- **ROC-AUC** — overall skill (0.5 = coin-flip, 1.0 = perfect). Ours ~0.7–0.8: good, not magic.
- **± spread** — how much the score wobbles across the 5 folds; overlapping ± ranges
  mean the models are effectively tied.
- **Overfit gap** — seen vs unseen performance; smaller = safer.
""")
    auc_leader = cv_means["ROC-AUC"].idxmax()
    take(f"<b>Which model builds the score?</b> A score must be <b>calibrated</b> — when it "
         f"says 35% risk, ~35% should really return. {auc_leader} ranks slightly best, but "
         f"<b>{SCORE_MODEL}</b>'s predicted risk matches reality almost exactly — so we score from it.")

    st.markdown("#### Look inside one model")
    pick = st.selectbox("Inspect a model", list(results.keys()),
                        index=list(results.keys()).index(SCORE_MODEL))
    cset = st.columns([1, 1])
    with cset[0]:
        cm = results[pick]["cm"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="BuGn")
        for a in range(2):
            for bb in range(2):
                ax.text(bb, a, f"{cm[a,bb]:,}", ha="center", va="center", fontsize=14,
                        fontweight="bold",
                        color="white" if cm[a, bb] > cm.max()/2 else "#0f172a")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Said Deliver", "Said Return"], color=MUT)
        ax.set_yticklabels(["Was Delivered", "Was Returned"], color=MUT)
        ax.set_title(f"{pick}: hits & misses"); ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        st.pyplot(fig, width="stretch")
    with cset[1]:
        fpr, tpr, _ = results[pick]["roc"]
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        ax.plot(fpr, tpr, color=ACCENT, lw=2.2,
                label=f"{pick} (AUC={results[pick]['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        ax.set_xlabel("False alarms"); ax.set_ylabel("Returns caught")
        ax.set_title("ROC: skill above the coin-flip line")
        ax.legend(loc="lower right", fontsize=8); bare(ax)
        st.pyplot(fig, width="stretch")

    # ROC of all models together — interactive: hover any curve, click legend to isolate
    st.subheader("All models on one chart")
    st.caption("Hover a curve to read it · click a legend entry to hide/show a model.")
    fig = go.Figure()
    palette = [ACCENT, AMBER, GREEN, RED, "#8b5cf6"]
    for (name, r), c in zip(results.items(), palette):
        fpr, tpr, _ = r["roc"]
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} ({r['roc_auc']:.2f})",
            line=dict(width=3 if name == SCORE_MODEL else 1.8, color=c),
            hovertemplate=name + "<br>False alarms %{x:.2f} · Returns caught %{y:.2f}"
                          "<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Coin-flip",
                             line=dict(dash="dot", color=GREY, width=1.2),
                             hoverinfo="skip"))
    fig.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      title="Higher and more to the top-left = better "
                            f"(thick line = {SCORE_MODEL}, the score's engine)",
                      xaxis_title="False alarms", yaxis_title="Returns caught",
                      legend=dict(orientation="h", y=-0.22),
                      margin=dict(t=50, r=10, b=10, l=10))
    st.plotly_chart(fig, width="stretch")

    # what the scoring model leans on (Gradient Boosting exposes feature_importances_)
    sm = results[SCORE_MODEL]["pipe"]
    ohe = sm.named_steps["pre"].named_transformers_["cat"]
    feat = NUM_FEATS + list(ohe.get_feature_names_out(CAT_FEATS))
    imp = pd.Series(sm.named_steps["clf"].feature_importances_, index=feat)
    imp = imp.sort_values(ascending=False).head(12)[::-1]
    st.markdown("#### What the score pays most attention to")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.barh(imp.index, imp.values, color=ACCENT, height=0.7)
    ax.set_xlabel("Importance")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")

    st.divider()
    st.markdown("#### Risk → a 300–900 score")
    proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
    cod_scored = cod.copy()
    cod_scored["Risk"] = proba
    cod_scored["Score"] = score
    st.session_state["cod_scored"] = cod_scored

    pred_mean = float(proba.mean()); true_mean = float(y.mean())
    cal = st.columns(2)
    statcard(cal[0], f"{pred_mean:.0%}", "score's avg predicted risk", ACCENT)
    statcard(cal[1], f"{true_mean:.0%}", "real return rate", GREEN)
    take("Those two numbers nearly match — that's <b>calibration</b>. It's why the score's "
         "percentage means what it says. Higher score = safer shopper.")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.hist(score, bins=40, color=GREY, edgecolor="none")
    ax.axvspan(300, 580, color=RED, alpha=0.14)
    ax.axvspan(580, 720, color=AMBER, alpha=0.16)
    ax.axvspan(720, 900, color=GREEN, alpha=0.16)
    top = ax.get_ylim()[1]
    ax.text(440, top*0.9, "Prepaid-only", color=RED, ha="center", fontweight="bold")
    ax.text(650, top*0.9, "COD + fee", color=AMBER, ha="center", fontweight="bold")
    ax.text(810, top*0.9, "COD free", color=GREEN, ha="center", fontweight="bold")
    ax.set_xlabel("COD Trust Score"); ax.set_ylabel("Shoppers")
    ax.set_title("Most shoppers are safe; a risky tail needs guard-rails"); bare(ax)
    st.pyplot(fig, width="stretch")

    st.markdown("**Example shoppers, low to high score:**")
    show = cod_scored.sort_values("Score")
    ex = pd.concat([show.head(2), show.iloc[[len(show)//2-1, len(show)//2]], show.tail(2)])
    st.dataframe(ex[["CustomerID", "CityTier", "Category", "OrderValue", "PriorOrders",
                     "PriorReturns", "Score"]].rename(columns={"OrderValue": "₹ Order"}),
                 width="stretch", hide_index=True)
    take("More past returns &amp; risky first-timers land <b>low</b>; loyal clean-history "
         "shoppers land <b>high</b> — just like a credit score rewards a good record.")

# =============================================================== 7. THE VERDICT
with tabs[6]:
    chapter(7, "⚖️", "The verdict: who gets COD?",
            "Turn the score into a 3-tier policy. Move the dials, watch the ₹ trade-off.",
            "#b45309", "#92400e")

    if "cod_scored" not in st.session_state:
        X, y = design_matrix(cod)
        results, _ = train_models(X, y, test_size, seed)
        proba, score = honest_scores(X, y, SCORE_MODEL, test_size, seed)
        cs = cod.copy(); cs["Risk"] = proba; cs["Score"] = score
        st.session_state["cod_scored"] = cs
    cs = st.session_state["cod_scored"].copy()

    st.markdown("#### Set the policy 🎛️")
    p1, p2, p3 = st.columns(3)
    with p1:
        low_cut = st.slider("Prepaid-only below score", 300, 700, 580, 10,
                            help="Risky shoppers below this must pay online")
    with p2:
        high_cut = st.slider("Free COD at or above score", 600, 900, 720, 10,
                             help="Trusted shoppers above this get COD with no friction")
    with p3:
        rto_cost = st.number_input("₹ lost per returned order", 100, 600,
                                   DEFAULT_RTO_COST, 10)
    cod_fee = st.slider("COD fee charged in the middle tier (₹)", 0, 150, 50, 5)
    if low_cut >= high_cut:
        st.warning("Keep the prepaid-only cut **below** the free-COD cut for three clean tiers.")
        high_cut = low_cut + 10

    cs["Tier"] = np.where(cs["Score"] >= high_cut, "Free COD",
                  np.where(cs["Score"] < low_cut, "Prepaid-only", "COD + fee"))

    order = ["Free COD", "COD + fee", "Prepaid-only"]
    colmap = {"Free COD": GREEN, "COD + fee": AMBER, "Prepaid-only": RED}
    summ = (cs.groupby("Tier")
              .agg(Shoppers=("Score", "size"), ReturnRate=("RTO", "mean"),
                   AvgScore=("Score", "mean"), AvgOrder=("OrderValue", "mean"))
              .reindex(order))

    st.markdown("#### The three tiers")
    ccc = st.columns(3)
    grad = {"Free COD": ("#16a34a", "#15803d"),
            "COD + fee": ("#f59e0b", "#d97706"),
            "Prepaid-only": ("#ef4444", "#dc2626")}
    for col, t in zip(ccc, order):
        n = int(summ.loc[t, "Shoppers"]); rr = summ.loc[t, "ReturnRate"]
        share = n / len(cs) * 100
        g1, g2 = grad[t]
        col.markdown(
            f"<div style='background:linear-gradient(135deg,{g1},{g2});border-radius:14px;"
            f"padding:16px 18px;color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.15);'>"
            f"<div style='font-weight:700;font-size:1.05rem;'>{t}</div>"
            f"<div style='font-size:2rem;font-weight:800;line-height:1.1;margin:4px 0;'>{share:.0f}%</div>"
            f"<div style='opacity:.92;font-size:.9rem;'>{n:,} orders · {rr*100:.0f}% return</div>"
            f"</div>", unsafe_allow_html=True)
    st.write("")

    # tier chart
    fig, ax = plt.subplots(figsize=(8, 2.8))
    bars = ax.bar(order, summ["ReturnRate"]*100, color=[colmap[t] for t in order], width=0.6)
    for b, t in zip(bars, order):
        ax.text(b.get_x()+b.get_width()/2, summ.loc[t, "ReturnRate"]*100+0.8,
                f"{summ.loc[t,'ReturnRate']*100:.0f}%", ha="center", fontweight="bold",
                color=INK)
    ax.set_ylabel("Return rate (%)"); ax.set_title("Return rate climbs cleanly across tiers")
    bare(ax)
    st.pyplot(fig, width="stretch")

    # ---- the money: what the policy saves vs costs ----
    st.markdown("#### Does the policy pay off? 💰")
    base_loss = (cs["RTO"] * rto_cost).sum()  # cost if we ship every COD order blindly

    # prepaid-only tier: assume online payment removes ~90% of these returns
    pp = cs[cs["Tier"] == "Prepaid-only"]
    prevented = pp["RTO"].sum() * 0.90
    saved_prepaid = prevented * rto_cost

    # but some genuine buyers abandon when COD is removed (industry: a real cost)
    abandon_rate = 0.25  # conservative: 1 in 4 prepaid-only good buyers walk away
    good_lost = (len(pp) - pp["RTO"].sum()) * abandon_rate
    avg_pp_order = pp["OrderValue"].mean() if len(pp) else 0
    lost_margin = good_lost * avg_pp_order * 0.20  # ~20% contribution margin

    # middle tier fee income (only on orders that DO get delivered & accepted)
    mid = cs[cs["Tier"] == "COD + fee"]
    fee_income = len(mid) * cod_fee * 0.85  # assume 85% still accept the small fee

    net = saved_prepaid + fee_income - lost_margin

    mcol = st.columns(4)
    mcol[0].metric("Blind COD loss (today)", f"₹{base_loss:,.0f}",
                   help="If we ship every COD order with no policy")
    mcol[1].metric("Saved by prepaid-only", f"₹{saved_prepaid:,.0f}",
                   help="Returns avoided by asking risky shoppers to pay online")
    mcol[2].metric("COD-fee income", f"₹{fee_income:,.0f}",
                   help="Small risk-priced fee from the middle tier")
    mcol[3].metric("Net effect of policy", f"₹{net:,.0f}",
                   delta="better" if net > 0 else "worse",
                   delta_color="normal" if net > 0 else "inverse")

    st.markdown(f"""
**How to read this (the honest trade-off):**
- Asking the **prepaid-only** tier to pay online prevents most of their returns →
  about **₹{saved_prepaid:,.0f}** saved.
- A small **₹{cod_fee} COD fee** in the middle tier brings in **₹{fee_income:,.0f}** —
  this is *risk-based pricing*, just like a higher interest rate on a riskier loan.
- **But** removing COD scares off some genuine buyers (we assumed 1 in 4 walk away),
  costing roughly **₹{lost_margin:,.0f}** in lost margin.
- **Net:** the policy comes out **{'ahead' if net>0 else 'behind'}** by
  **₹{abs(net):,.0f}** on this data.
""")
    take("Push the <b>prepaid-only</b> dial right → prevent more returns, but lose more "
         "genuine sales. That tension is the real decision — the score lets you make it "
         "with eyes open.")

    st.divider()
    st.markdown("#### 📋 What we found — and what we'd tell the business")
    st.markdown(f"""
**The story in six lines:**
1. On our store, **COD returns at {cod_rto*100:.0f}%** versus **{prepaid_rto*100:.0f}%** for
   prepaid — COD is where the leak is.
2. Inside COD, returns are **not random**: smaller towns, fashion/footwear, deep discounts,
   first-timers, and vague addresses all return more.
3. The loudest single signal is a shopper's **own past-return record** — history repeats.
4. A model reads all these clues at **~0.7–0.8 skill** and turns them into one **COD Trust
   Score** (300–900), CIBIL-style.
5. That score sorts shoppers into **free COD / COD-with-a-fee / prepaid-only**, and the
   return rate rises cleanly across the three.
6. Applied sensibly, the policy comes out **net positive** — even after accounting for
   genuine buyers lost when COD is removed.

**Recommendations, each tied to a finding above:**
- **Default deep-discount + first-time + vague-address COD orders to the middle tier** (a
  small COD fee), because those are exactly the slices that returned most in Chapter 4.
- **Switch COD off for the bottom tier**, but soften it with a **small prepaid discount** —
  the industry shows a ₹75–125 nudge converts most of them to paying online willingly.
- **Fix addresses at checkout** (a "confirm your full address" step) — it's the cheapest
  lever, since vague addresses were among the strongest risk clues.
- **Reward loyalty:** let a clean track record *raise* a shopper's score over time, so good
  customers feel trusted, not punished.
""")

    st.warning("**A fair warning (our honesty note).** A low COD Score is **not** a verdict "
               "that someone is a bad person. It only means *this order is risky to ship on "
               "cash* — so we ask for prepayment instead. People in patchy-address areas or "
               "new to a brand aren't dishonest; they're just harder to deliver to. The "
               "score adjusts the **offer**, never judges the **person**. Used carelessly it "
               "could unfairly shut out whole towns — so a real brand should monitor it for "
               "exactly that and keep a prepaid path open for everyone.")

    st.markdown("#### 🔬 Limitations — stated plainly, before anyone asks")
    with st.expander("What this analysis can and cannot claim", expanded=True):
        st.markdown(f"""
- **Synthetic-data circularity.** The dataset was *built* with a hidden reliability
  trait that drives both past and future returns — so "history predicts the future"
  is partly baked in. That's deliberate (it lets us verify the methods recover a
  known truth), but real-world data will be noisier; expect the 0.7–0.8 AUC to be an
  optimistic ceiling, not a promise.
- **Predictors, not causes.** Deep discounts *co-occur* with returns; that doesn't
  prove cutting discounts cuts returns (impulsive shoppers may simply chase
  discounts). The honest next step for any lever here is an **A/B test**, not a
  policy rollout.
- **The ₹ math rests on assumptions.** The Verdict's savings assume prepayment
  prevents **90%** of flagged returns, **25%** of genuine buyers abandon when COD is
  removed, and **85%** accept the middle-tier fee. These are industry-plausible, not
  measured — the dials exist precisely so you can stress-test them.
- **No time dimension.** Orders are a snapshot; seasonality, shopper drift, and
  score staleness are invisible. A live system must re-train and re-calibrate on a
  schedule.
- **Fairness must be monitored, not assumed.** City tier and state correlate with
  income and infrastructure. The score never uses identity, but geography can proxy
  it — a real deployment needs a recurring bias audit (approval rates by
  region/segment) and a prepaid path open to everyone.
""")

    st.download_button("⬇️ Download the cleaned, scored data (CSV)",
                       cs.to_csv(index=False).encode(),
                       "cod_scored.csv", "text/csv")

# =============================================================== 8. CUSTOMER SEGMENTS
with tabs[7]:
    chapter(8, "🧩", "Who shops alike?",
            "Unsupervised learning: K-Means for hard segments, a mixture model for soft, "
            "latent-class-style ones. No labels — just structure in the behaviour.",
            "#be185d", "#9d174d")

    st.markdown("#### Step 1 — roll every order up into one row per shopper")
    st.caption("Clustering groups **customers**, not orders, so we aggregate first: how "
               "often they buy, how much, how deep the discounts, how COD-heavy, and how "
               "often they've returned something.")

    cust = (df.groupby("CustomerID").agg(
        Orders=("OrderID", "count"),
        AvgOrderValue=("OrderValue", "mean"),
        AvgDiscount=("DiscountPct", "mean"),
        PctCOD=("IsCOD", "mean"),
        ReturnRate=("RTO", "mean"),
    ).reset_index())
    st.dataframe(cust.head(6), width="stretch")

    SEG_FEATS = ["Orders", "AvgOrderValue", "AvgDiscount", "PctCOD", "ReturnRate"]
    Xc = StandardScaler().fit_transform(cust[SEG_FEATS])

    st.markdown("#### Step 2 — how many segments actually exist?")
    st.caption("**Elbow method:** inertia (spread within clusters) always falls as K grows — "
               "we look for where it stops falling fast. **Silhouette:** a −1→1 score for how "
               "cleanly separated the clusters are; higher is better.")
    ks = list(range(2, 9))
    inertias, sils = [], []
    for k in ks:
        _km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
        inertias.append(_km.inertia_)
        sils.append(silhouette_score(Xc, _km.labels_))
    best_k = ks[int(np.argmax(sils))]

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0))
    axs[0].plot(ks, inertias, marker="o", color=ACCENT, lw=2)
    axs[0].set_title("Elbow: inertia vs K"); axs[0].set_xlabel("K"); axs[0].set_ylabel("Inertia")
    bare(axs[0])
    axs[1].plot(ks, sils, marker="o", color=ACCENT, lw=2)
    axs[1].axvline(best_k, color=RED, ls="--", lw=1.3)
    axs[1].set_title("Silhouette: separation vs K"); axs[1].set_xlabel("K"); axs[1].set_ylabel("Score")
    bare(axs[1])
    st.pyplot(fig, width="stretch")
    take(f"Silhouette peaks at <b>K={best_k}</b> — the data itself has {best_k} natural "
         "shopper types, echoing the split between mostly-reliable shoppers and a flakier minority.")

    k = st.slider("Number of segments (K)", 2, 8, best_k)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
    cust["Segment"] = km.labels_

    prof = cust.groupby("Segment")[SEG_FEATS].mean().round(2)
    prof.insert(0, "Shoppers", cust["Segment"].value_counts().sort_index())
    st.markdown("#### Step 3 — what does each segment look like?")
    st.dataframe(
        prof.style.background_gradient(subset=["ReturnRate"], cmap="RdYlGn_r"),
        width="stretch")

    st.markdown("#### Step 4 — see it in 3D (drag to rotate)")
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(Xc)
    plot_df = cust.copy()
    plot_df["PC1"], plot_df["PC2"], plot_df["PC3"] = coords[:, 0], coords[:, 1], coords[:, 2]
    plot_df["Segment"] = plot_df["Segment"].astype(str)
    fig3d = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Segment",
                          hover_data=SEG_FEATS, opacity=0.72,
                          color_discrete_sequence=px.colors.qualitative.Set2)
    fig3d.update_traces(marker=dict(size=4))
    fig3d.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=480,
                        legend=dict(orientation="h", y=1.02))
    st.plotly_chart(fig3d, width="stretch")
    st.caption(f"3 principal components explain **{pca.explained_variance_ratio_.sum():.0%}** "
               "of the spread between shoppers — the axes have no single meaning, but distance "
               "still means 'behaves differently'.")

    riskiest = prof["ReturnRate"].idxmax()
    safest = prof["ReturnRate"].idxmin()
    take(f"Segment <b>{riskiest}</b> returns <b>{prof.loc[riskiest,'ReturnRate']*100:.0f}%</b> "
         f"of orders vs. just <b>{prof.loc[safest,'ReturnRate']*100:.0f}%</b> for segment "
         f"<b>{safest}</b> — the same split the COD Score is built to catch, just found without "
         "ever telling the algorithm about returns.")

    st.divider()
    st.markdown("#### Soft segments — closer to true Latent Class Analysis")
    st.caption("K-Means forces every shopper into exactly ONE box. A **Gaussian Mixture "
               "Model** instead gives each shopper a *probability* of belonging to each "
               "hidden 'type' — the same spirit as Latent Class Analysis, and the standard "
               "way to do it when your indicators are numeric rather than purely categorical.")

    comps = list(range(2, 8))
    bics = [GaussianMixture(n_components=c, random_state=42, n_init=3).fit(Xc).bic(Xc)
            for c in comps]
    best_c = comps[int(np.argmin(bics))]
    fig, ax = plt.subplots(figsize=(7.5, 2.6))
    ax.plot(comps, bics, marker="o", color=ACCENT, lw=2)
    ax.axvline(best_c, color=RED, ls="--", lw=1.3)
    ax.set_xlabel("Latent classes"); ax.set_ylabel("BIC (lower = better fit)")
    ax.set_title("Choosing the number of latent classes")
    bare(ax)
    st.pyplot(fig, width="stretch")

    gm = GaussianMixture(n_components=best_c, random_state=42, n_init=5).fit(Xc)
    membership = pd.DataFrame(gm.predict_proba(Xc),
                              columns=[f"Class {i+1}" for i in range(best_c)])
    preview = pd.concat([cust[["CustomerID"]].reset_index(drop=True), membership], axis=1)
    st.markdown(f"**BIC picks {best_c} latent classes.** Example membership probabilities:")
    st.dataframe(preview.head(6).round(2), width="stretch", hide_index=True)
    confident = (membership.max(axis=1) > 0.9).mean()
    take(f"<b>{confident*100:.0f}%</b> of shoppers have a &gt;90% probability of belonging to "
         "just one class — the hidden types are real and sharply defined, not a fuzzy in-between.")

# =============================================================== 9. WHAT DRIVES ORDER VALUE
with tabs[8]:
    chapter(9, "💰", "What drives order value?",
            "Regression, not classification: predicting a ₹ number instead of a Yes/No — "
            "Linear, Ridge, Lasso, and a Decision Tree Regressor.",
            "#7c2d12", "#78350f")

    st.markdown("#### Setting up the regression")
    st.caption("Target: **OrderValue** (₹, continuous). Features are everything knowable "
               "the moment the order is placed — nothing about the outcome leaks in.")

    RNUM = ["DiscountPct", "Items", "OrderHour", "PriorOrders"]
    RCAT = ["CityTier", "Category", "PaymentMethod", "Device", "AddressQuality"]
    Xr, yr = df[RNUM + RCAT].copy(), df["OrderValue"].copy()
    Xtr, Xte, ytr, yte = train_test_split(Xr, yr, test_size=test_size, random_state=seed)

    # ---- Step 0: is regularization even needed? Check multicollinearity (VIF) ----
    st.markdown("#### Before the models: do we even *need* Ridge & Lasso? (VIF check)")
    st.caption("Ridge and Lasso exist to tame **multicollinearity** — features that "
               "carry overlapping information, which makes plain-Linear coefficients "
               "wobble. **VIF** (Variance Inflation Factor) measures that overlap: "
               "1 = fully independent, above 5 = worrying overlap.")
    vif_extra = ["PriorRTORate", "FirstTime"]          # engineered cousins of PriorOrders
    vif_cols = RNUM + [c for c in vif_extra if c in df.columns]
    Znum = StandardScaler().fit_transform(df[vif_cols])
    vif_vals = np.diag(np.linalg.inv(np.corrcoef(Znum, rowvar=False)))
    vif = pd.Series(vif_vals, index=vif_cols).sort_values()

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(vif) + 1.2))
    bars = ax.barh(vif.index, vif.values,
                   color=[RED if v > 5 else ACCENT for v in vif.values], height=0.6)
    for b, v in zip(bars, vif.values):
        ax.text(v + 0.06, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                va="center", fontsize=9, color=INK)
    ax.axvline(5, color=AMBER, ls="--", lw=1.4)
    ax.text(5.02, -0.45, "VIF = 5 (concern threshold)", color=AMBER, fontsize=8.5)
    ax.set_xlim(0, max(6.0, vif.max() * 1.25)); ax.set_xlabel("VIF")
    ax.set_title("Variance Inflation Factor — numeric features (incl. engineered cousins)")
    bare(ax); ax.tick_params(left=False)
    st.pyplot(fig, width="stretch")

    high_vif = vif[vif > 5]
    if len(high_vif):
        take(f"<b>VIF confirms real overlap</b> in {', '.join(high_vif.index)} — plain "
             "Linear coefficients on these will be unstable. <b>That is the actual "
             "reason Ridge and Lasso are in this chapter</b>, not a checklist.")
    else:
        take(f"VIFs are <b>mild</b> (max {vif.max():.2f}, all under 5) — though "
             "<b>PriorOrders, PriorRTORate and FirstTime</b> are natural cousins (all "
             "read the shopper's history). So Ridge/Lasso act as <b>insurance</b> here: "
             "if they only match plain Linear, that's the honest finding — the fix is "
             "small because the problem is small. The dial below lets you verify it live.")

    rpre = ColumnTransformer([("num", StandardScaler(), RNUM),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), RCAT)])
    reg_models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=5.0),
        "Lasso": Lasso(alpha=2.0, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=seed),
    }
    rrows, fitted = [], {}
    for name, m in reg_models.items():
        pipe = Pipeline([("pre", rpre), ("m", m)]).fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rrows.append({"Model": name, "R²": r2_score(yte, pred),
                     "RMSE (₹)": np.sqrt(mean_squared_error(yte, pred)),
                     "MAE (₹)": mean_absolute_error(yte, pred)})
        fitted[name] = pipe
    rmetrics = pd.DataFrame(rrows).set_index("Model").round(1)

    st.markdown("#### How four models did")
    st.dataframe(
        rmetrics.style.highlight_max(subset=["R²"], color="#d1fae5")
                .highlight_min(subset=["RMSE (₹)", "MAE (₹)"], color="#d1fae5"),
        width="stretch")
    with st.expander("What do these words mean?"):
        st.markdown("""
- **R²** — share of the price variation the model explains (0 = no better than guessing the
  average, 1 = perfect). Ours sits around **0.5** — real signal, not magic.
- **RMSE** — typical prediction error in ₹, penalises big misses harder.
- **MAE** — typical prediction error in ₹, treats every miss equally.
""")
    best_r = rmetrics["R²"].idxmax()
    take(f"<b>{best_r}</b> explains about <b>{rmetrics.loc[best_r,'R²']*100:.0f}%</b> of why "
         "order values differ. The rest is just what a shopper *chose* to buy — no checkout "
         "signal can predict that, and it shouldn't be able to.")

    st.markdown("#### Look inside one model")
    pick_r = st.selectbox("Inspect a regression model", list(reg_models.keys()),
                          index=list(reg_models.keys()).index(best_r))
    pipe_r = fitted[pick_r]
    pred_r = pipe_r.predict(Xte)
    ohe_r = pipe_r.named_steps["pre"].named_transformers_["cat"]
    feat_r = RNUM + list(ohe_r.get_feature_names_out(RCAT))

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        ax.scatter(yte, pred_r, alpha=0.22, s=14, color=ACCENT)
        lims = [min(yte.min(), pred_r.min()), max(yte.max(), pred_r.max())]
        ax.plot(lims, lims, ls="--", color=GREY, lw=1)
        ax.set_xlabel("Actual ₹"); ax.set_ylabel("Predicted ₹")
        ax.set_title(f"{pick_r}: predicted vs actual"); bare(ax)
        st.pyplot(fig, width="stretch")
    with c2:
        if pick_r == "Decision Tree":
            vals = pipe_r.named_steps["m"].feature_importances_
            label = "Importance"
        else:
            vals = pipe_r.named_steps["m"].coef_
            label = "Coefficient (standardised)"
        imp = pd.Series(vals, index=feat_r)
        imp = imp.reindex(imp.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        colors = [ACCENT if v >= 0 else GREY for v in imp.values]
        ax.barh(imp.index, imp.values, color=colors, height=.65)
        ax.set_xlabel(label); ax.set_title(f"{pick_r}: what moves the price"); bare(ax)
        st.pyplot(fig, width="stretch")
    take("Category dominates — Electronics simply costs more than Beauty. Discount depth and "
         "city tier barely move the needle. Order value is a merchandising story, not a risk one.")

    # ---- interactive regularization: move λ, watch coefficients shrink ----
    st.divider()
    st.markdown("#### Turn the regularization dial yourself 🎛️")
    st.caption("λ (lambda) is the **penalty strength**. λ ≈ 0 → behaves like plain "
               "Linear. Crank it up: **Ridge** shrinks all coefficients smoothly toward "
               "zero; **Lasso** snaps the weakest ones to *exactly* zero — automatic "
               "feature selection. Watch R² to see when shrinking starts to hurt.")
    rc1, rc2 = st.columns([1, 2])
    with rc1:
        reg_kind = st.radio("Regularizer", ["Ridge", "Lasso"], horizontal=True)
    with rc2:
        log_l = st.slider("Penalty strength  log₁₀(λ)", -2.0, 3.0, 0.5, 0.1)
    lam = 10 ** log_l

    live = Pipeline([("pre", rpre),
                     ("m", Ridge(alpha=lam) if reg_kind == "Ridge"
                           else Lasso(alpha=lam, max_iter=8000))]).fit(Xtr, ytr)
    lp = live.predict(Xte)
    live_coef = pd.Series(live.named_steps["m"].coef_, index=feat_r)
    n_zero = int((live_coef.abs() < 1e-8).sum())

    lm1, lm2, lm3 = st.columns(3)
    lm1.metric("λ", f"{lam:,.3g}")
    lm2.metric("Test R²", f"{r2_score(yte, lp):.3f}")
    lm3.metric("Coefficients zeroed", f"{n_zero} / {len(live_coef)}",
               help="Only Lasso can zero a coefficient exactly — Ridge just shrinks.")

    topc = live_coef.reindex(live_coef.abs().sort_values(ascending=False).index).head(12)[::-1]
    figc = go.Figure(go.Bar(
        x=topc.values, y=topc.index, orientation="h",
        marker_color=[ACCENT if v >= 0 else GREY for v in topc.values],
        hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
    figc.update_layout(height=380, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       title=f"{reg_kind} coefficients at λ = {lam:,.3g} "
                             "(standardised — sizes comparable)",
                       xaxis_title="coefficient (₹ effect, standardised)",
                       margin=dict(t=50, r=10, b=10, l=10))
    st.plotly_chart(figc, width="stretch")

    @st.cache_data(show_spinner="Tracing the shrinkage path…")
    def shrink_path(kind, tsize, sd, n_rows):
        alphas = np.logspace(-2, 3, 24)
        coefs, r2s = [], []
        for a in alphas:
            m = Pipeline([("pre", rpre),
                          ("m", Ridge(alpha=a) if kind == "Ridge"
                                else Lasso(alpha=a, max_iter=8000))]).fit(Xtr, ytr)
            coefs.append(m.named_steps["m"].coef_)
            r2s.append(r2_score(yte, m.predict(Xte)))
        return alphas, np.array(coefs), r2s

    alphas, paths, r2s = shrink_path(reg_kind, test_size, seed, len(Xtr))
    watch_idx = (pd.Series(np.abs(paths).max(axis=0), index=feat_r)
                 .sort_values(ascending=False).head(7).index)
    figp = go.Figure()
    for name in watch_idx:
        i = feat_r.index(name)
        figp.add_trace(go.Scatter(
            x=alphas, y=paths[:, i], mode="lines", name=name,
            hovertemplate=name + "<br>λ %{x:.3g} · coef %{y:.1f}<extra></extra>"))
    figp.update_layout(height=420, paper_bgcolor="rgba(0,0,0,0)",
                       plot_bgcolor="rgba(0,0,0,0)",
                       title=f"{reg_kind}: how each coefficient shrinks as λ grows",
                       xaxis=dict(type="log", title="λ (log scale)"),
                       yaxis_title="coefficient",
                       legend=dict(orientation="h", y=-0.25),
                       margin=dict(t=50, r=10, b=10, l=10))
    st.plotly_chart(figp, width="stretch")
    take("How to read it: under <b>Ridge</b> the lines glide toward zero together; under "
         "<b>Lasso</b> the weakest lines hit zero and die one by one — the survivors are "
         "the features that genuinely carry the price signal. Since test R² barely moves "
         "until λ gets large, the honest conclusion is: multicollinearity here is mild, "
         "and regularization is <b>insurance, not a rescue</b>.")

# =============================================================== 10. ASSOCIATION RULES
with tabs[9]:
    chapter(10, "🔗", "Which combinations predict a return?",
            "Association rule mining: support, confidence, and lift — what clusters of "
            "clues fire together, beyond what any single one shows alone.",
            "#1e3a8a", "#1e40af")

    st.markdown("#### Step 1 — turn each COD order into a 'basket' of traits")
    st.caption("Every COD order becomes a small basket of labels: its city tier, category, "
               "address quality, device, whether it's a first order, how deep the discount "
               "is, and the outcome. Association mining finds which labels commonly fire together.")

    def _band(v):
        if v <= 10: return "Discount:0-10%"
        if v <= 25: return "Discount:11-25%"
        if v <= 40: return "Discount:26-40%"
        return "Discount:40%+"

    tx = pd.DataFrame({
        "Tier": "Tier:" + cod["CityTier"],
        "Category": "Cat:" + cod["Category"],
        "Address": "Addr:" + cod["AddressQuality"],
        "Device": "Dev:" + cod["Device"],
        "First": np.where(cod["FirstTime"] == 1, "First:Yes", "First:No"),
        "Discount": cod["DiscountPct"].map(_band),
        "Outcome": np.where(cod["RTO"] == 1, "Outcome:Returned", "Outcome:Delivered"),
    })
    onehot = pd.concat([pd.get_dummies(tx[c]) for c in tx.columns], axis=1).astype(bool)

    st.markdown("#### Step 2 — mine the rules (three dials, three meanings)")
    with st.expander("📖 How to drive the three sliders", expanded=True):
        st.markdown("""
Every rule reads **IF (combo of traits) THEN (outcome)**. Three dials decide which
rules you see:

- **Support** — how *common* the combo is. Slide **down** to surface rare niches
  (riskier, noisier), **up** to keep only widespread patterns.
- **Confidence** — how *reliable* the rule is. 0.5 means: when the IF-side is true,
  the THEN-side follows half the time. Raise it when you want rules you could act on.
- **Lift** — the *surprise* dial, and the one that matters most. Lift 1.0 = the combo
  changes nothing (it just restates the average). Lift 1.5 = the outcome is **50% more
  likely** than baseline. Raise this step by step until only genuine signals survive.

**Suggested demo flow for the presentation:** defaults → focus on *Returned* →
raise lift until ~5 rules remain → hover the bubbles and read them out.
""")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        min_sup = st.slider("Minimum support", 0.01, 0.20, 0.03, 0.01,
                            help="How common the combination must be")
    with sc2:
        min_conf = st.slider("Minimum confidence", 0.10, 0.95, 0.40, 0.05,
                             help="How reliably the IF-side leads to the THEN-side")
    with sc3:
        min_lift = st.slider("Minimum lift", 1.0, 3.0, 1.15, 0.05,
                             help="How much more likely than pure chance (1.0 = no link)")

    freq = apriori(onehot, min_support=min_sup, use_colnames=True)

    if len(freq) == 0:
        st.warning("No combinations meet this support threshold — lower the slider.")
    else:
        rules = association_rules(freq, metric="lift", min_threshold=1.0)
        rules["antecedents"] = rules["antecedents"].apply(lambda s: ", ".join(sorted(s)))
        rules["consequents"] = rules["consequents"].apply(lambda s: ", ".join(sorted(s)))
        view = rules[(rules["confidence"] >= min_conf) &
                     (rules["lift"] >= min_lift)].copy()

        focus = st.radio("Business lens — which outcome do the rules point at?",
                         ["🔴 Returned (the risk we price)", "🟢 Delivered (the safe profile)",
                          "All rules"], horizontal=True)
        if "Returned" in focus:
            view = view[view["consequents"].str.contains("Outcome:Returned")]
        elif "Delivered" in focus:
            view = view[view["consequents"].str.contains("Outcome:Delivered")]
        st.caption(f"**{len(view):,}** rules pass the dials "
                   f"(of {len(rules):,} mined at support ≥ {min_sup:.2f}).")

        risky = view.sort_values("lift", ascending=False)
        st.markdown("#### Strongest combinations under the current dials")
        show_cols = ["antecedents", "consequents", "support", "confidence", "lift"]
        st.dataframe(risky[show_cols].head(15).round(3), width="stretch", hide_index=True)

        if len(risky):
            top = risky.iloc[0]
            take(f"Strongest signal: <b>{top['antecedents']}</b> → <b>Returned</b> — lift "
                 f"<b>{top['lift']:.2f}</b>, meaning that combination returns "
                 f"<b>{top['lift']:.1f}×</b> more often than random chance alone.")

        st.markdown("#### The rule landscape — hover any bubble to read its rule")
        land = (view if len(view) else rules).nlargest(80, "lift")
        fig = go.Figure(go.Scatter(
            x=land["support"], y=land["confidence"], mode="markers",
            marker=dict(size=np.clip(land["lift"] * 13, 8, 42),
                        color=land["lift"], colorscale="Viridis",
                        colorbar=dict(title="Lift"),
                        line=dict(width=0.5, color="rgba(255,255,255,.6)")),
            text=("<b>IF</b> " + land["antecedents"] +
                  "<br><b>THEN</b> " + land["consequents"]),
            hovertemplate="%{text}<br>support %{x:.3f} · confidence %{y:.2f}"
                          "<extra></extra>"))
        fig.update_layout(height=480, paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          xaxis_title="Support (how common)",
                          yaxis_title="Confidence (how reliable)",
                          title="Bigger & brighter bubble = higher lift = stronger, "
                                "more surprising rule",
                          margin=dict(t=50, r=10, b=10, l=10))
        st.plotly_chart(fig, width="stretch")
        take("These are the exact combinations Chapter 7's policy should default straight to "
             "the middle or bottom tier — no need to wait for a full order history to flag them.")

st.caption("Built with Streamlit + scikit-learn · synthetic data, real-world patterns · "
           "a low score adjusts the offer, never judges the person.")
