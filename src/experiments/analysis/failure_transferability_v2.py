"""
Failure Transferability v2 — statistically rigorous variant
============================================================

Improvements over v1:
  1. Drops PCA (4 models, 6 pairs — consistent with rest of thesis).
  2. Excess Jaccard = J_observed − E[J_random], where E[J_random] uses the
     same failure-set sizes and a file universe of N files, with
     E[|A∩B|] = |A|·|B|/N. Controls for trivial sharing driven by set sizes.
  3. Spearman ρ(severity, Jaccard) with p-values per experiment × failure_type.
  4. Entropy link: computes #models-failing-per-file at the most severe setting,
     then Spearman(entropy, consensus_count).
  5. Per-pair summary (6 pairs) and heatmap at highest severity.
  6. Threshold robustness: FN∈{0.03, 0.05, 0.10} side-by-side.

Outputs → results/analysis/failure_transferability_v2/
  excess_jaccard.csv
  severity_trends.csv
  entropy_failure_link.csv
  per_pair_summary.csv
  threshold_robustness.csv
  plots/*.png
"""

import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from itertools import combinations
from pathlib import Path
from scipy.stats import spearmanr, entropy as sp_entropy
import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS_DIR = PROJECT_ROOT / "results" / "experiments"
BASELINE_CSV = PROJECT_ROOT / "results" / "tables" / "baseline_final_subset.csv"
SUBSET_CSV = PROJECT_ROOT / "results" / "tables" / "final_subset.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "analysis" / "failure_transferability_v2"
PLOT_DIR = OUTPUT_DIR / "plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(PROJECT_ROOT / "src"))

MODELS = ["IForest", "LOF", "MP", "AE"]              # PCA dropped
MODEL_PAIRS = list(combinations(MODELS, 2))          # 6 pairs
FN_THRESHOLD_DEFAULT = 0.05
FP_THRESHOLD_DEFAULT = 0.10
FN_THRESHOLDS_ROBUST = [0.03, 0.05, 0.10]

EXPERIMENTS = [
    ("white_noise_snr",     "White Noise",
     lambda c: float(re.search(r"snr_([-\d]+)dB", c).group(1)),
     "SNR (dB)", True),                               # invert: lower SNR = more severe
    ("freeze",              "Sensor Freeze",
     lambda c: float(re.search(r"frac_([\d.]+)_ns_(\d+)", c).group(1)),
     "Corruption Fraction", False),
    ("spikes_normal_only",  "Spikes",
     lambda c: float(re.search(r"frac_([\d.]+)_mult", c).group(1)),
     "Corruption Fraction", False),
    ("swap_segment",        "Segment Swap",
     lambda c: float(re.search(r"frac_([\d.]+)_ns", c).group(1)),
     "Corruption Fraction", False),
    ("missing_true_impact", "Missing Data",
     lambda c: float(re.search(r"frac_([\d.]+)", c).group(1)),
     "Corruption Fraction", False),
]

# ─── helpers ───────────────────────────────────────────────────────
def load_baseline():
    df = pd.read_csv(BASELINE_CSV)
    df["model"] = df["model"].replace({"Autoencoder": "AE"})
    return df[df["model"].isin(MODELS)].copy()

def load_experiment(exp_name):
    path = EXPERIMENTS_DIR / exp_name / "checkpoint.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["model"] = df["model"].replace({"Autoencoder": "AE"})
    df = df[df["model"].isin(MODELS)].copy()
    if "error" in df.columns:
        df = df[df["error"].isna() | (df["error"] == "")].copy()
    for c in ["Precision", "Recall"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["Precision", "Recall"])

def jaccard(a, b):
    if not a and not b:
        return np.nan
    u = a | b
    return len(a & b) / len(u) if u else np.nan

def expected_jaccard(size_a, size_b, N):
    """E[Jaccard] under uniform random failure assignment with fixed sizes."""
    if N == 0 or (size_a == 0 and size_b == 0):
        return np.nan
    exp_inter = size_a * size_b / N
    exp_union = size_a + size_b - exp_inter
    return exp_inter / exp_union if exp_union > 0 else np.nan

def failure_set(df_cond, model, ftype, thr):
    m = df_cond[df_cond["model"] == model]
    col, op = ("Recall", thr) if ftype == "fn" else ("Precision", thr)
    return set(m[m[col] < op]["file"])

# ─── main transferability loop ─────────────────────────────────────
def analyze(thr_fn=FN_THRESHOLD_DEFAULT, thr_fp=FP_THRESHOLD_DEFAULT):
    baseline_df = load_baseline()
    rows = []             # excess_jaccard.csv
    consensus_rows = []   # for entropy link (only at most-severe per experiment)
    pairwise_severe = []  # per_pair_summary.csv

    for exp_name, display, extract, label, invert in EXPERIMENTS:
        df = load_experiment(exp_name)
        if df is None or df.empty:
            continue
        universe = set(df["file"].unique())
        N = len(universe)
        severities = []
        for cond in df["condition"].unique():
            try:
                sev = extract(cond)
            except Exception:
                continue
            severities.append((cond, sev))
        # most severe
        most_severe = max(severities, key=lambda t: -t[1] if invert else t[1])

        for cond, sev in severities:
            sub = df[df["condition"] == cond]
            for ftype, thr in [("fn", thr_fn), ("fp", thr_fp)]:
                fsets = {m: failure_set(sub, m, ftype, thr) for m in MODELS}
                # pairwise
                for a, b in MODEL_PAIRS:
                    sa, sb = len(fsets[a]), len(fsets[b])
                    j = jaccard(fsets[a], fsets[b])
                    e = expected_jaccard(sa, sb, N)
                    rows.append(dict(
                        experiment=display, condition=cond, severity=sev,
                        failure_type=ftype, model_a=a, model_b=b,
                        jaccard=j, expected_jaccard=e,
                        excess=(j - e) if (j==j and e==e) else np.nan,
                        size_a=sa, size_b=sb, N=N,
                    ))
                # most-severe: record per-pair + consensus
                if cond == most_severe[0]:
                    for a, b in MODEL_PAIRS:
                        pairwise_severe.append(dict(
                            experiment=display, failure_type=ftype,
                            model_a=a, model_b=b,
                            jaccard=jaccard(fsets[a], fsets[b]),
                        ))
                    # per-file consensus count
                    for f in universe:
                        n_fail = sum(1 for m in MODELS if f in fsets[m])
                        consensus_rows.append(dict(
                            experiment=display, failure_type=ftype,
                            file=f, n_models_failed=n_fail,
                        ))

    df_rows = pd.DataFrame(rows)
    df_rows.to_csv(OUTPUT_DIR / "excess_jaccard.csv", index=False)
    pd.DataFrame(pairwise_severe).to_csv(OUTPUT_DIR / "per_pair_summary.csv", index=False)

    return df_rows, pd.DataFrame(consensus_rows)

# ─── Spearman trends per experiment ────────────────────────────────
def severity_trends(df_rows):
    out = []
    for (exp, ft), g in df_rows.groupby(["experiment", "failure_type"]):
        # mean Jaccard across 6 pairs per severity
        agg = g.groupby("severity")[["jaccard", "excess"]].mean().reset_index()
        if len(agg) < 3:
            continue
        # For White Noise we invert so higher severity = smaller SNR (multiply by -1)
        x = agg["severity"].values
        if exp == "White Noise":
            x = -x
        for col in ["jaccard", "excess"]:
            rho, p = spearmanr(x, agg[col].values, nan_policy="omit")
            out.append(dict(experiment=exp, failure_type=ft, metric=col,
                            spearman_rho=rho, p_value=p, n_points=len(agg)))
    df = pd.DataFrame(out)
    df.to_csv(OUTPUT_DIR / "severity_trends.csv", index=False)
    return df

# ─── Entropy link ──────────────────────────────────────────────────
def compute_entropies():
    from data_loader import load_tsb_dataframe
    sub = pd.read_csv(SUBSET_CSV)
    out = []
    for _, r in sub.iterrows():
        fp = r["filepath"]
        try:
            df, _ = load_tsb_dataframe(fp)
            v = df["value"].values
            hist, _ = np.histogram(v, bins=100, density=True)
            ent = sp_entropy(hist + 1e-10)
            out.append({"file": r["filename_actual"], "entropy": ent})
        except Exception as e:
            out.append({"file": r["filename_actual"], "entropy": np.nan})
    return pd.DataFrame(out)

def entropy_link(consensus_df, entropies):
    # merge
    m = consensus_df.merge(entropies, on="file", how="left")
    rows = []
    for (exp, ft), g in m.groupby(["experiment", "failure_type"]):
        g2 = g.dropna(subset=["entropy"])
        if len(g2) < 5:
            continue
        rho, p = spearmanr(g2["entropy"], g2["n_models_failed"])
        rows.append(dict(experiment=exp, failure_type=ft,
                         spearman_rho=rho, p_value=p, n=len(g2),
                         mean_entropy_hi_consensus=g2[g2.n_models_failed>=3]["entropy"].mean(),
                         mean_entropy_lo_consensus=g2[g2.n_models_failed<=1]["entropy"].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "entropy_failure_link.csv", index=False)
    return df, m

# ─── Threshold robustness ──────────────────────────────────────────
def threshold_robustness():
    rows = []
    for thr in FN_THRESHOLDS_ROBUST:
        dr, _ = analyze(thr_fn=thr, thr_fp=FP_THRESHOLD_DEFAULT)
        for exp, g in dr[dr.failure_type=="fn"].groupby("experiment"):
            rows.append(dict(threshold=thr, experiment=exp,
                             mean_jaccard=g["jaccard"].mean(),
                             mean_excess=g["excess"].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "threshold_robustness.csv", index=False)
    return df

# ─── Plots ─────────────────────────────────────────────────────────
COLORS = {
    "White Noise": "#d62728", "Spikes": "#ff7f0e", "Sensor Freeze": "#9467bd",
    "Segment Swap": "#2ca02c", "Missing Data": "#1f77b4",
}
def plot_excess_by_severity(df_rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, ft, title in zip(axes, ["fn","fp"],
                              ["FN-failures (Recall<0.05)", "FP-failures (Precision<0.10)"]):
        for exp, g in df_rows[df_rows.failure_type==ft].groupby("experiment"):
            agg = g.groupby("severity")["excess"].mean().reset_index()
            x = agg["severity"].values
            if exp == "White Noise":
                x = -x  # higher = more severe
                xlab = "Noise severity (−SNR, dB)"
            else:
                xlab = "Corruption fraction"
            ax.plot(x, agg["excess"], "o-", label=exp, color=COLORS[exp], lw=2)
        ax.axhline(0, color="gray", linestyle=":", alpha=0.7, label="Random expectation")
        ax.set_xlabel("Severity"); ax.set_ylabel("Excess Jaccard (J − E[J_random])")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "excess_jaccard_by_severity.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_per_pair_heatmap(pairwise_severe_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, ft, title in zip(axes, ["fn","fp"],
                              ["FN Jaccard at most-severe setting", "FP Jaccard at most-severe setting"]):
        g = pairwise_severe_df[pairwise_severe_df.failure_type==ft]
        # build matrix: rows=experiment, cols=pair
        g = g.copy()
        g["pair"] = g["model_a"] + "-" + g["model_b"]
        pivot = g.pivot_table(index="experiment", columns="pair", values="jaccard")
        # order
        exp_order = ["White Noise","Spikes","Sensor Freeze","Segment Swap","Missing Data"]
        pair_order = [f"{a}-{b}" for a,b in MODEL_PAIRS]
        pivot = pivot.reindex(index=exp_order, columns=pair_order)
        im = ax.imshow(pivot.values, cmap="Reds", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pair_order))); ax.set_xticklabels(pair_order, rotation=45, ha="right")
        ax.set_yticks(range(len(exp_order))); ax.set_yticklabels(exp_order)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i,j]
                if v==v:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v>0.5 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="Jaccard")
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "per_pair_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_entropy_vs_consensus(merged):
    # focus on FN at most-severe
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8), sharey=True)
    exp_order = ["White Noise","Spikes","Sensor Freeze","Segment Swap","Missing Data"]
    for ax, exp in zip(axes, exp_order):
        g = merged[(merged.experiment==exp) & (merged.failure_type=="fn")].dropna(subset=["entropy"])
        if len(g)==0: continue
        # jitter y
        y = g["n_models_failed"].values + (np.random.RandomState(0).rand(len(g))-0.5)*0.25
        ax.scatter(g["entropy"], y, alpha=0.5, color=COLORS[exp], s=22)
        rho, p = spearmanr(g["entropy"], g["n_models_failed"])
        ax.set_title(f"{exp}\nρ={rho:+.2f}, p={p:.2g}", fontsize=10)
        ax.set_xlabel("Entropy")
        ax.set_yticks([0,1,2,3,4]); ax.grid(alpha=0.3)
    axes[0].set_ylabel("# models failing on file")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "entropy_vs_consensus.png", dpi=140, bbox_inches="tight")
    plt.close()

def plot_threshold_robustness(df):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for exp, g in df.groupby("experiment"):
        ax.plot(g["threshold"], g["mean_excess"], "o-", label=exp, color=COLORS[exp], lw=2)
    ax.axhline(0, color="gray", linestyle=":")
    ax.set_xlabel("FN threshold (Recall <)"); ax.set_ylabel("Mean excess Jaccard")
    ax.set_title("Robustness: mean excess Jaccard vs FN threshold")
    ax.grid(alpha=0.3); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "threshold_robustness.png", dpi=140, bbox_inches="tight")
    plt.close()

# ─── main ──────────────────────────────────────────────────────────
def main():
    print(f"Output: {OUTPUT_DIR}")
    print("Running core analysis (thr_fn=0.05, thr_fp=0.10)...")
    df_rows, consensus = analyze()
    print(f"  excess_jaccard.csv: {len(df_rows)} rows")

    print("Computing Spearman severity trends...")
    trends = severity_trends(df_rows)
    print(trends.round(3).to_string(index=False))

    print("\nComputing per-file entropies (reads 141 files)...")
    ents = compute_entropies()
    print(f"  {ents.entropy.notna().sum()}/{len(ents)} files with entropy")

    print("\nComputing entropy × consensus link...")
    elink, merged = entropy_link(consensus, ents)
    print(elink.round(3).to_string(index=False))

    print("\nThreshold robustness sweep (FN∈{0.03,0.05,0.10})...")
    rob = threshold_robustness()
    print(rob.round(3).to_string(index=False))

    print("\nGenerating plots...")
    plot_excess_by_severity(df_rows)
    pairwise_severe = pd.read_csv(OUTPUT_DIR / "per_pair_summary.csv")
    plot_per_pair_heatmap(pairwise_severe)
    plot_entropy_vs_consensus(merged)
    plot_threshold_robustness(rob)
    print(f"  plots saved to {PLOT_DIR}")

    print("\nDone.")

if __name__ == "__main__":
    main()
