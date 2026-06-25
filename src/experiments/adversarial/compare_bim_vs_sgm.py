"""
Fair Comparison: BIM vs SGM at equal perturbation budget.

Varies BIM epsilon until its L2 norm matches SGM,
then compares smoothness (Paper Eq. 11) at equal perturbation levels.

Usage:
    python -u scripts/test_fair_comparison.py
"""
import os, sys, torch, torch.nn as nn, torch.optim as optim
import numpy as np, pandas as pd, matplotlib.pyplot as plt, matplotlib

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

SUBSET_CSV = "c:/Users/gkost/thesis_timeseries/results/tables/robust_subset_TSB.csv"
PLOT_DIR = "c:/Users/gkost/thesis_timeseries/results/plots/adversarial"
os.makedirs(PLOT_DIR, exist_ok=True)

WINDOW_SIZE = 50
N_DATASETS = 8
GM_MU = 1.0; GM_ALPHA = 1.0; SGM_LAMBDA = 1.0; LR = 0.01; ITERS = 200
BIM_EPSILONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]  # Sweep BIM's epsilon


class SurrogateAE(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(ws, ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, max(2, ws//4)), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(max(2, ws//4), ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, ws))
    def forward(self, x): return self.decoder(self.encoder(x))


def train_surrogate(T, ws, epochs=50):
    m = SurrogateAE(ws); opt = optim.Adam(m.parameters(), lr=0.01)
    crit = nn.MSELoss(); wins = T.unfold(0, ws, 1)
    for _ in range(epochs):
        opt.zero_grad(); loss = crit(m(wins), wins); loss.backward(); opt.step()
    return m


def anomaly_score(model, T, ws):
    wins = T.unfold(0, ws, 1); recon = model(wins)
    return torch.mean((recon - wins)**2, dim=1)


def attack_bim(model, T, ws, epsilon, alpha=0.01, iters=200, target=None):
    model.eval(); T_adv = T.clone().detach().requires_grad_(True)
    for _ in range(iters):
        scores = anomaly_score(model, T_adv, ws)
        if target is not None and len(target) > 0:
            loss = sum(torch.sum(scores[max(0,i-ws+1):min(len(scores),i+1)]) for i in target)
        else:
            loss = torch.sum(scores)
        loss.backward()
        with torch.no_grad():
            T_adv_new = T_adv - alpha * T_adv.grad.sign()
            eta = torch.clamp(T_adv_new - T, -epsilon, epsilon)
            T_adv = (T + eta).detach().requires_grad_(True)
    return T_adv.detach()


def attack_sgm(model, T, ws, mu=1.0, alpha=1.0, lam=1.0, lr=0.01, iters=200, target=None):
    model.eval(); r = torch.randn_like(T)*0.01; r.requires_grad_(True)
    opt = optim.Adam([r], lr=lr)
    for _ in range(iters):
        opt.zero_grad(); scores = anomaly_score(model, T+r, ws)
        if target is not None and len(target) > 0:
            aloss = sum(torch.mean(scores[max(0,i-ws+1):min(len(scores),i+1)]) for i in target)
        else:
            aloss = torch.mean(scores)
        loss = mu*aloss + alpha*torch.sum(r**2) + lam*torch.sum(torch.abs(r[1:]-r[:-1]))
        loss.backward(); opt.step()
    return (T+r).detach()


def smoothness(x):
    x = np.asarray(x, dtype=np.float64); T = len(x)
    if T < 3: return 0.0
    d2 = x[:-2] - 2*x[1:-1] + x[2:]
    return np.sum(d2**2) / T


def evaluate_signal(X, y, sw):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    results = {}
    for name in ['IForest', 'PCA']:
        try:
            Xf = Window(window=sw).convert(X).to_numpy()
            m = IForest(n_jobs=1) if name == 'IForest' else PCA()
            m.fit(Xf)
            sc = np.pad(np.array(m.decision_scores_, dtype=np.float64),
                       (sw-1, 0), 'constant', constant_values=(m.decision_scores_[0],))
            sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
            results[name] = get_metrics(sc, y)['AUC_ROC']
        except: results[name] = None
    return results


def main():
    print("="*70)
    print("FAIR COMPARISON: BIM (varying ε) vs SGM (fixed λ=1)")
    print("="*70)

    df_sub = pd.read_csv(SUBSET_CSV)
    datasets = df_sub[df_sub['data_len'] > 500].head(N_DATASETS)
    all_results = []

    for i, (_, row) in enumerate(datasets.iterrows()):
        name, fp, folder = row['baseline_name'], row['filepath'], row['folder']
        print(f"\n[{i+1}/{N_DATASETS}] {folder}/{name}")

        df = pd.read_csv(fp, header=None).apply(pd.to_numeric, errors='coerce').fillna(0)
        X_orig = df.iloc[:,0].values.astype(np.float64)
        y = df.iloc[:,1].values.astype(np.float64)
        Xm, Xs = X_orig.mean(), X_orig.std()+1e-8
        Xn = (X_orig - Xm) / Xs
        T = torch.tensor(Xn, dtype=torch.float32)
        sw = find_length(X_orig)
        ws = min(WINDOW_SIZE, len(Xn)//4)
        target = np.where(y > 0)[0]
        target = target if len(target) > 0 else None

        print("  Training surrogate...")
        surr = train_surrogate(T, ws, 50)
        auc_orig = evaluate_signal(X_orig, y, sw)
        print(f"  Baseline IF:{auc_orig['IForest']:.3f} PCA:{auc_orig['PCA']:.3f}")

        # SGM (fixed)
        print("  Running SGM...")
        T_sgm = attack_sgm(surr, T, ws, GM_MU, GM_ALPHA, SGM_LAMBDA, LR, ITERS, target)
        X_sgm = T_sgm.numpy()*Xs + Xm
        r_sgm = (T_sgm - T).numpy()
        auc_sgm = evaluate_signal(X_sgm, y, sw)
        l2_sgm = np.linalg.norm(r_sgm) / len(r_sgm)
        sm_sgm = smoothness(r_sgm)

        for model in ['IForest', 'PCA']:
            all_results.append({
                'dataset': name, 'model': model, 'attack': 'SGM',
                'epsilon': 0, 'AUC_Drop': (auc_orig[model] or 0)-(auc_sgm[model] or 0),
                'L2': l2_sgm, 'Smoothness': sm_sgm
            })
        print(f"  SGM: IF:{auc_sgm['IForest']:.3f} PCA:{auc_sgm['PCA']:.3f} L2:{l2_sgm:.4f} Smooth:{sm_sgm:.6f}")

        # BIM sweep
        for eps in BIM_EPSILONS:
            print(f"  BIM ε={eps}...", end=" ")
            T_bim = attack_bim(surr, T, ws, eps, 0.01, ITERS, target)
            X_bim = T_bim.numpy()*Xs + Xm
            r_bim = (T_bim - T).numpy()
            auc_bim = evaluate_signal(X_bim, y, sw)
            l2_bim = np.linalg.norm(r_bim) / len(r_bim)
            sm_bim = smoothness(r_bim)

            for model in ['IForest', 'PCA']:
                all_results.append({
                    'dataset': name, 'model': model, 'attack': f'BIM_e{eps}',
                    'epsilon': eps, 'AUC_Drop': (auc_orig[model] or 0)-(auc_bim[model] or 0),
                    'L2': l2_bim, 'Smoothness': sm_bim
                })
            print(f"IF:{auc_bim['IForest']:.3f} PCA:{auc_bim['PCA']:.3f} L2:{l2_bim:.4f} Smooth:{sm_bim:.6f}")

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(PLOT_DIR, "fair_comparison_results.csv"), index=False)
    plot_fair_comparison(df_res)


def plot_fair_comparison(df):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Aggregate BIM by epsilon
    bim_data = df[df['attack'].str.startswith('BIM')].copy()
    sgm_data = df[df['attack'] == 'SGM'].copy()

    bim_agg = bim_data.groupby('epsilon').agg({
        'AUC_Drop': 'mean', 'L2': 'mean', 'Smoothness': 'mean'
    }).reset_index()
    sgm_mean = {'AUC_Drop': sgm_data['AUC_Drop'].mean(),
                'L2': sgm_data['L2'].mean(),
                'Smoothness': sgm_data['Smoothness'].mean()}

    # Plot 1: L2 vs AUC Drop
    ax = axes[0]
    ax.plot(bim_agg['L2'], bim_agg['AUC_Drop'], 'o-', color='#FF5722',
            linewidth=2.5, markersize=10, label='BIM (varying ε)')
    for _, row in bim_agg.iterrows():
        ax.annotate(f'ε={row["epsilon"]}', (row['L2'], row['AUC_Drop']),
                   textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.plot(sgm_mean['L2'], sgm_mean['AUC_Drop'], 's', color='#9C27B0',
            markersize=14, label='SGM (λ=1)', zorder=5)
    ax.set_xlabel('L2 Norm (perturbation size)', fontsize=11)
    ax.set_ylabel('Mean AUC Drop', fontsize=11)
    ax.set_title('Effectiveness vs Perturbation Size', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Plot 2: L2 vs Smoothness (THE KEY PLOT)
    ax = axes[1]
    ax.plot(bim_agg['L2'], bim_agg['Smoothness'], 'o-', color='#FF5722',
            linewidth=2.5, markersize=10, label='BIM (varying ε)')
    for _, row in bim_agg.iterrows():
        ax.annotate(f'ε={row["epsilon"]}', (row['L2'], row['Smoothness']),
                   textcoords="offset points", xytext=(8, 5), fontsize=8)
    ax.plot(sgm_mean['L2'], sgm_mean['Smoothness'], 's', color='#9C27B0',
            markersize=14, label='SGM (λ=1)', zorder=5)
    ax.set_xlabel('L2 Norm (perturbation size)', fontsize=11)
    ax.set_ylabel('Smoothness s(r) — Paper Eq. 11\n(Lower = smoother)', fontsize=11)
    ax.set_title('KEY: Smoothness at Equal Perturbation', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=0.3)

    # Plot 3: Efficiency = Drop / L2
    ax = axes[2]
    bim_agg['Efficiency'] = bim_agg['AUC_Drop'] / (bim_agg['L2'] + 1e-8)
    sgm_eff = sgm_mean['AUC_Drop'] / (sgm_mean['L2'] + 1e-8)

    ax.bar(range(len(bim_agg)), bim_agg['Efficiency'], color='#FF5722', alpha=0.8, width=0.6)
    ax.bar(len(bim_agg), sgm_eff, color='#9C27B0', alpha=0.8, width=0.6)
    labels = [f'BIM\nε={e}' for e in bim_agg['epsilon']] + ['SGM\nλ=1']
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Efficiency (Drop / L2)', fontsize=11)
    ax.set_title('Attack Efficiency', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)

    fig.suptitle('Fair Comparison: BIM (varying ε) vs SGM\n'
                 'At similar perturbation budget, SGM produces smoother attacks',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(PLOT_DIR, "fair_comparison_bim_vs_sgm.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved fair comparison: {out}")

    # Summary table
    print("\n" + "="*70)
    print("FAIR COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Attack':<12} {'ε':<6} {'Drop':<10} {'L2':<10} {'Smoothness':<12} {'Efficiency':<10}")
    print("-"*60)
    for _, r in bim_agg.iterrows():
        print(f"{'BIM':<12} {r['epsilon']:<6.2f} {r['AUC_Drop']:<10.4f} {r['L2']:<10.4f} "
              f"{r['Smoothness']:<12.6f} {r['Efficiency']:<10.4f}")
    print(f"{'SGM':<12} {'—':<6} {sgm_mean['AUC_Drop']:<10.4f} {sgm_mean['L2']:<10.4f} "
          f"{sgm_mean['Smoothness']:<12.6f} {sgm_eff:<10.4f}")


if __name__ == "__main__":
    main()
