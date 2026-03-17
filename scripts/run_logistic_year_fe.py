"""
Logistic Regression with Year Fixed Effects for Credit Risk Contagion.

Models:
  A  – Financial controls + Year FE (baseline)
  B  – A + LAG contagion (all 7 edge types)
  C  – A + CURRENT contagion (all 7 edge types)
  D  – Same as B, restricted to currently-profitable firms
  E  – 7 individual edge-type models (financial + year FE + one lag contagion var)
"""

import os, json, warnings, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

# ── paths ───────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "processed", "stage1", "distress_task")
OUT_DIR  = os.path.join(DATA_DIR, "experiments")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Load & merge ────────────────────────────────────────────────────
parts = []
for split in ("train", "val", "test"):
    fp = os.path.join(DATA_DIR, f"{split}.csv")
    df = pd.read_csv(fp)
    df["split"] = split
    parts.append(df)
data = pd.concat(parts, ignore_index=True)

ctg = pd.read_csv(os.path.join(DATA_DIR, "contagion_features.csv"))
# Ensure merge keys are same dtype
data["Stkcd"] = data["Stkcd"].astype(str)
ctg["Stkcd"]  = ctg["Stkcd"].astype(str)
data["year"]   = data["year"].astype(int)
ctg["year"]    = ctg["year"].astype(int)

data = data.merge(ctg, on=["Stkcd", "year"], how="left")

LABEL = "label_loss_next_year"
data = data.dropna(subset=[LABEL])
data[LABEL] = data[LABEL].astype(int)

print(f"Total observations after merge: {len(data)}")
print(f"Positive rate: {data[LABEL].mean():.4f}")
print(f"Years: {sorted(data['year'].unique())}")

# ── 2. Feature definitions ─────────────────────────────────────────────
FINANCIAL = [
    "net_profit", "roa", "equity_ratio", "cashflow_to_assets",
    "asset_liability_ratio", "is_loss_current_year",
]

EDGE_TYPES = [
    "guarantee", "shared_nonlisted_guarantee", "equity_assoc",
    "equity_change", "co_controller", "market_corr", "industry",
]

LAG_CTG   = [f"ctg_{e}_lag_loss_ratio" for e in EDGE_TYPES]
CUR_CTG   = [f"ctg_{e}_loss_ratio"     for e in EDGE_TYPES]

# log(total_assets)
data["log_total_assets"] = np.log1p(data["total_assets"].clip(lower=0))
FINANCIAL_PLUS = FINANCIAL + ["log_total_assets"]

# Fill NaN contagion with 0
for c in LAG_CTG + CUR_CTG:
    if c in data.columns:
        data[c] = data[c].fillna(0)
    else:
        data[c] = 0.0

# Year dummies
data["year_cat"] = data["year"].astype(str)
year_dummies = pd.get_dummies(data["year_cat"], prefix="yr", drop_first=True, dtype=float)
YEAR_COLS = list(year_dummies.columns)
data = pd.concat([data, year_dummies], axis=1)

# ── helpers ─────────────────────────────────────────────────────────────

def _standardize(df, cols):
    """Z-score standardize; drop zero-variance columns."""
    out = df.copy()
    kept = []
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
        s = out[c].std()
        if s < 1e-12:
            continue
        out[c] = (out[c] - out[c].mean()) / s
        kept.append(c)
    return out, kept


def _sig(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def run_logit(df, feature_cols, label=LABEL, tag="model"):
    """Fit logistic regression, return result dict."""
    work = df.copy()
    # Standardize numeric features (not year dummies)
    numeric_feats = [c for c in feature_cols if not c.startswith("yr_")]
    work, kept_numeric = _standardize(work, numeric_feats)
    yr_feats = [c for c in feature_cols if c.startswith("yr_")]
    all_feats = kept_numeric + yr_feats

    # Drop any remaining cols with zero variance
    all_feats = [c for c in all_feats if work[c].std() > 1e-12]

    y = work[label].values
    X = sm.add_constant(work[all_feats].values.astype(float))
    col_names = ["const"] + all_feats

    # Fit
    try:
        model = sm.Logit(y, X)
        res = model.fit(disp=0, maxiter=200)
    except Exception:
        model = sm.Logit(y, X)
        res = model.fit(method="bfgs", disp=0, maxiter=500)

    # If not converged, try bfgs
    if not res.mle_retvals.get("converged", True):
        try:
            res = model.fit(method="bfgs", disp=0, maxiter=500)
        except Exception:
            pass

    # Extract
    summary = {
        "tag": tag,
        "n_obs": int(res.nobs),
        "pseudo_r2": float(res.prsquared),
        "log_likelihood": float(res.llf),
        "ll_null": float(res.llnull),
        "lr_chi2": float(res.llr),
        "lr_pvalue": float(res.llr_pvalue),
        "aic": float(res.aic),
        "bic": float(res.bic),
    }

    coef_table = []
    for i, name in enumerate(col_names):
        coef  = res.params[i]
        se    = res.bse[i]
        z     = res.tvalues[i]
        p     = res.pvalues[i]
        ci_lo = res.conf_int()[i, 0]
        ci_hi = res.conf_int()[i, 1]
        OR    = np.exp(coef)
        OR_lo = np.exp(ci_lo)
        OR_hi = np.exp(ci_hi)
        coef_table.append({
            "variable": name,
            "coef": float(coef),
            "std_err": float(se),
            "z": float(z),
            "p": float(p),
            "sig": _sig(p),
            "odds_ratio": float(OR),
            "OR_ci_lo": float(OR_lo),
            "OR_ci_hi": float(OR_hi),
        })
    summary["coefficients"] = coef_table
    return summary


def print_model(res, contagion_vars=None, show_financial=True, show_year=False):
    """Pretty-print one model result."""
    print(f"\n{'='*80}")
    print(f"  {res['tag']}")
    print(f"{'='*80}")
    print(f"  N = {res['n_obs']:,}   Pseudo-R² = {res['pseudo_r2']:.4f}   "
          f"Log-L = {res['log_likelihood']:.1f}   AIC = {res['aic']:.1f}   BIC = {res['bic']:.1f}")
    print(f"  LR chi² = {res['lr_chi2']:.2f}   (p = {res['lr_pvalue']:.2e})")
    print(f"{'─'*80}")
    header = f"  {'Variable':<42} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>8}  {'OR':>7} {'95% CI OR':>18}"
    print(header)
    print(f"  {'─'*len(header.strip())}")

    for row in res["coefficients"]:
        name = row["variable"]
        # Filter what to show
        is_yr = name.startswith("yr_")
        is_ctg = name.startswith("ctg_")
        is_const = name == "const"

        if is_yr and not show_year:
            continue
        if not show_financial and not is_ctg and not is_const and not is_yr:
            continue

        sig = row["sig"]
        ci_str = f"[{row['OR_ci_lo']:.3f}, {row['OR_ci_hi']:.3f}]"
        print(f"  {name:<42} {row['coef']:>8.4f} {row['std_err']:>8.4f} "
              f"{row['z']:>8.3f} {row['p']:>8.4f}{sig:<3} {row['odds_ratio']:>7.3f} {ci_str:>18}")

    # Highlight contagion vars
    if contagion_vars:
        print(f"\n  Contagion variables detail:")
        print(f"  {'─'*74}")
        for row in res["coefficients"]:
            if row["variable"] in contagion_vars:
                sig = row["sig"]
                ci_str = f"[{row['OR_ci_lo']:.3f}, {row['OR_ci_hi']:.3f}]"
                edge = row["variable"].replace("ctg_","").replace("_lag_loss_ratio","").replace("_loss_ratio","")
                print(f"    {edge:<38} OR={row['odds_ratio']:.3f} {ci_str}  p={row['p']:.4f}{sig}")


# ── 3. Run models ──────────────────────────────────────────────────────
all_results = {}

# Model A: Financial + Year FE
print("\n" + "#"*80)
print("# Model A: Financial controls + Year FE (baseline)")
print("#"*80)
feats_A = FINANCIAL_PLUS + YEAR_COLS
res_A = run_logit(data, feats_A, tag="Model A: Financial + Year FE")
print_model(res_A, show_financial=True, show_year=True)
all_results["A"] = res_A

# Model B: Financial + Year FE + LAG contagion
print("\n" + "#"*80)
print("# Model B: Financial + Year FE + LAG contagion (all 7 edge types)")
print("#"*80)
feats_B = FINANCIAL_PLUS + YEAR_COLS + LAG_CTG
res_B = run_logit(data, feats_B, tag="Model B: Financial + Year FE + LAG contagion")
print_model(res_B, contagion_vars=LAG_CTG, show_financial=True)
all_results["B"] = res_B

# Model C: Financial + Year FE + CURRENT contagion
print("\n" + "#"*80)
print("# Model C: Financial + Year FE + CURRENT contagion (all 7 edge types)")
print("#"*80)
feats_C = FINANCIAL_PLUS + YEAR_COLS + CUR_CTG
res_C = run_logit(data, feats_C, tag="Model C: Financial + Year FE + CURRENT contagion")
print_model(res_C, contagion_vars=CUR_CTG, show_financial=True)
all_results["C"] = res_C

# Model D: Same as B but only for profitable firms
print("\n" + "#"*80)
print("# Model D: LAG contagion – currently profitable firms only")
print("#"*80)
data_profit = data[data["is_loss_current_year"] == 0].copy()
feats_D = [c for c in FINANCIAL_PLUS if c != "is_loss_current_year"] + YEAR_COLS + LAG_CTG
res_D = run_logit(data_profit, feats_D,
                  tag=f"Model D: LAG contagion (profitable only, N={len(data_profit)})")
print_model(res_D, contagion_vars=LAG_CTG, show_financial=True)
all_results["D"] = res_D

# Model E: Individual edge-type models
print("\n" + "#"*80)
print("# Model E: Individual edge-type models (each with one lag contagion variable)")
print("#"*80)
all_results["E"] = {}
for edge, lag_var in zip(EDGE_TYPES, LAG_CTG):
    feats_e = FINANCIAL_PLUS + YEAR_COLS + [lag_var]
    tag_e = f"Model E ({edge}): Financial + Year FE + {lag_var}"
    res_e = run_logit(data, feats_e, tag=tag_e)
    # Print compact
    print(f"\n  ── {edge} ──")
    ctg_row = [r for r in res_e["coefficients"] if r["variable"] == lag_var]
    if ctg_row:
        r = ctg_row[0]
        print(f"     coef={r['coef']:.4f}  SE={r['std_err']:.4f}  z={r['z']:.3f}  "
              f"p={r['p']:.4f}{r['sig']}  OR={r['odds_ratio']:.3f}  "
              f"95%CI=[{r['OR_ci_lo']:.3f},{r['OR_ci_hi']:.3f}]")
    print(f"     Pseudo-R²={res_e['pseudo_r2']:.4f}  N={res_e['n_obs']}")
    all_results["E"][edge] = res_e

# ── 4. Comparative summary table ───────────────────────────────────────
print("\n\n" + "="*90)
print("  COMPARATIVE SUMMARY: Contagion Variable Odds Ratios Across Models")
print("="*90)
print(f"  {'Edge Type':<35} {'Model B (lag)':>15} {'Model C (cur)':>15} {'Model D (profit)':>17} {'Model E (indiv)':>16}")
print(f"  {'─'*98}")

for edge in EDGE_TYPES:
    lag_var = f"ctg_{edge}_lag_loss_ratio"
    cur_var = f"ctg_{edge}_loss_ratio"

    def _extract(res, var):
        for r in res["coefficients"]:
            if r["variable"] == var:
                return f"{r['odds_ratio']:.3f}{r['sig']}"
        return "—"

    b_val = _extract(res_B, lag_var)
    c_val = _extract(res_C, cur_var)
    d_val = _extract(res_D, lag_var)
    e_val = _extract(all_results["E"][edge], lag_var)

    print(f"  {edge:<35} {b_val:>15} {c_val:>15} {d_val:>17} {e_val:>16}")

# ── 5. LR test: Model B vs Model A ────────────────────────────────────
lr_stat = 2 * (res_B["log_likelihood"] - res_A["log_likelihood"])
df_diff = len([r for r in res_B["coefficients"] if r["variable"].startswith("ctg_")])
lr_p = stats.chi2.sf(lr_stat, df_diff)
print(f"\n  LR test (Model B vs A): chi²={lr_stat:.2f}, df={df_diff}, p={lr_p:.2e}")

lr_stat_c = 2 * (res_C["log_likelihood"] - res_A["log_likelihood"])
df_diff_c = len([r for r in res_C["coefficients"] if r["variable"].startswith("ctg_")])
lr_p_c = stats.chi2.sf(lr_stat_c, df_diff_c)
print(f"  LR test (Model C vs A): chi²={lr_stat_c:.2f}, df={df_diff_c}, p={lr_p_c:.2e}")


# ── 6. Save ────────────────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, "logistic_year_fe_results.json")
# Convert numpy types for JSON
def _convert(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    raise TypeError(f"Not serializable: {type(obj)}")

with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=_convert)

print(f"\n  Results saved to: {out_path}")
print("  Done.")
