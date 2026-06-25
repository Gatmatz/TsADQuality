"""
Distillation Attack: Varying α (L2 regularization weight).

Lower α = more perturbation allowed = bigger AUC drops.
Compares Generic AE vs Distilled at different α values.

Usage:
    python -u scripts/test_distillation_alpha.py
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
ITERS = 200
LR = 0.01
ALPHAS = [0.01, 0.1, 0.5, 1.0]  # L2 regularization weights to try


class SurrogateScoreNet(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ws, ws), nn.ReLU(),
            nn.Linear(ws, ws//2), nn.ReLU(),
            nn.Linear(ws//2, max(2, ws//4)), nn.ReLU(),
            nn.Linear(max(2, ws//4), 1)
        )
    def forward(self, x): return self.net(x).squeeze(-1)


class SurrogateAE(nn.Module):
    def __init__(self, ws):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(ws, ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, max(2, ws//4)), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(max(2, ws//4), ws//2), nn.ReLU(),
                                     nn.Linear(ws//2, ws))
    def forward(self, x): return self.decoder(self.encoder(x))


def train_distillation(X, target_name, ws, sw, epochs=100):
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    Xf = Window(window=sw).convert(Xc).to_numpy()
    det = IForest(n_jobs=1) if target_name == 'IForest' else PCA()
    det.fit(Xf)
    scores = np.pad(np.array(det.decision_scores_, dtype=np.float64),
                    (sw-1,0), 'constant', constant_values=(det.decision_scores_[0],))
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    s_min, s_max = scores.min(), scores.max()+1e-8
    scores_n = (scores - s_min)/(s_max - s_min)

    Xn = (X - X.mean())/(X.std()+1e-8)
    T = torch.tensor(Xn, dtype=torch.float32)
    wins = T.unfold(0, ws, 1)
    st = torch.tensor(scores_n, dtype=torch.float32)
    ws_scores = st.unfold(0, ws, 1).mean(dim=1)

    model = SurrogateScoreNet(ws)
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); loss = crit(model(wins), ws_scores)
        loss.backward(); opt.step()

    with torch.no_grad():
        corr = np.corrcoef(model(wins).numpy(), ws_scores.numpy())[0,1]
    return model, corr


def train_ae(X, ws, epochs=50):
    Xn = (X - X.mean())/(X.std()+1e-8)
    T = torch.tensor(Xn, dtype=torch.float32)
    m = SurrogateAE(ws); opt = optim.Adam(m.parameters(), lr=0.01)
    crit = nn.MSELoss(); wins = T.unfold(0, ws, 1)
    for _ in range(epochs):
        opt.zero_grad(); loss = crit(m(wins), wins); loss.backward(); opt.step()
    return m


def attack_distilled(surrogate, T, ws, alpha_l2, lr=0.01, iters=200, target=None):
    surrogate.eval()
    r = torch.randn_like(T)*0.01; r.requires_grad_(True)
    opt = optim.Adam([r], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        wins = (T+r).unfold(0, ws, 1)
        scores = surrogate(wins)
        if target is not None and len(target) > 0:
            loss_s = sum(torch.mean(scores[max(0,i-ws+1):min(len(scores),i+1)]) for i in target)
        else:
            loss_s = torch.mean(scores)
        loss = loss_s + alpha_l2 * torch.sum(r**2)
        loss.backward(); opt.step()
    return (T+r).detach()


def attack_ae(model, T, ws, alpha_l2, lr=0.01, iters=200, target=None):
    model.eval()
    r = torch.randn_like(T)*0.01; r.requires_grad_(True)
    opt = optim.Adam([r], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        wins = (T+r).unfold(0, ws, 1)
        recon = model(wins)
        scores = torch.mean((recon-wins)**2, dim=1)
        if target is not None and len(target) > 0:
            loss_s = sum(torch.mean(scores[max(0,i-ws+1):min(len(scores),i+1)]) for i in target)
        else:
            loss_s = torch.mean(scores)
        loss = loss_s + alpha_l2 * torch.sum(r**2)
        loss.backward(); opt.step()
    return (T+r).detach()


def smoothness(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3: return 0.0
    d2 = x[:-2] - 2*x[1:-1] + x[2:]
    return np.sum(d2**2)/len(x)


def evaluate(X, y, sw):
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    res = {}
    for name in ['IForest', 'PCA']:
        try:
            Xf = Window(window=sw).convert(X).to_numpy()
            m = IForest(n_jobs=1) if name == 'IForest' else PCA()
            m.fit(Xf)
            sc = np.pad(np.array(m.decision_scores_, dtype=np.float64),
                       (sw-1,0), 'constant', constant_values=(m.decision_scores_[0],))
            sc = np.nan_to_num(sc, nan=0.0, posinf=0.0, neginf=0.0)
            res[name] = get_metrics(sc, y)['AUC_ROC']
        except: res[name] = None
    return res


def main():
    print("="*70)
    print("DISTILLATION ATTACK: Varying α (L2 regularization)")
    print(f"Alphas: {ALPHAS}")
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
        Xn = (X_orig-Xm)/Xs
        T = torch.tensor(Xn, dtype=torch.float32)
        sw = find_length(X_orig)
        ws = min(WINDOW_SIZE, len(Xn)//4)
        target = np.where(y > 0)[0]
        target = target if len(target) > 0 else None

        auc_orig = evaluate(X_orig, y, sw)
        print(f"  Baseline IF:{auc_orig['IForest']:.3f} PCA:{auc_orig['PCA']:.3f}")

        # Train surrogates once
        print("  Training AE...")
        ae = train_ae(X_orig, ws)

        for tgt in ['IForest', 'PCA']:
            print(f"\n  Target: {tgt}")
            dist, corr = train_distillation(X_orig, tgt, ws, sw)
            print(f"  Distillation corr: {corr:.3f}")

            for alpha in ALPHAS:
                # Distilled attack
                T_d = attack_distilled(dist, T, ws, alpha, LR, ITERS, target)
                X_d = T_d.numpy()*Xs+Xm
                auc_d = evaluate(X_d, y, sw)
                r_d = (T_d-T).numpy()
                l2 = np.linalg.norm(r_d)/len(r_d)
                drop = (auc_orig[tgt] or 0)-(auc_d[tgt] or 0)

                # Generic AE attack (same α for fair comparison)
                T_a = attack_ae(ae, T, ws, alpha, LR, ITERS, target)
                X_a = T_a.numpy()*Xs+Xm
                auc_a = evaluate(X_a, y, sw)
                r_a = (T_a-T).numpy()
                l2_a = np.linalg.norm(r_a)/len(r_a)
                drop_a = (auc_orig[tgt] or 0)-(auc_a[tgt] or 0)

                print(f"  α={alpha:<5} | Dist: Δ={drop:+.3f} L2={l2:.5f} | AE: Δ={drop_a:+.3f} L2={l2_a:.5f}")

                all_results.append({'dataset': name, 'target': tgt, 'alpha': alpha,
                    'attack': 'Distilled', 'Drop': drop, 'L2': l2, 'Smooth': smoothness(r_d)})
                all_results.append({'dataset': name, 'target': tgt, 'alpha': alpha,
                    'attack': 'Generic_AE', 'Drop': drop_a, 'L2': l2_a, 'Smooth': smoothness(r_a)})

    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(PLOT_DIR, "distillation_alpha_sweep.csv"), index=False)
    plot_alpha_sweep(df_res)


def plot_alpha_sweep(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, tgt in enumerate(['IForest', 'PCA']):
        # Left: Drop vs α
        ax = axes[row, 0]
        for attack, color, marker in [('Distilled','#2196F3','o'), ('Generic_AE','#FF9800','s')]:
            mask = (df['target']==tgt) & (df['attack']==attack)
            agg = df.loc[mask].groupby('alpha').agg({'Drop':'mean','L2':'mean'}).reset_index()
            ax.plot(agg['alpha'], agg['Drop'], f'{marker}-', color=color, linewidth=2.5,
                    markersize=10, label=attack)
            for _, r in agg.iterrows():
                ax.annotate(f'{r["Drop"]:+.3f}', (r['alpha'], r['Drop']),
                           textcoords="offset points", xytext=(8,5), fontsize=8)
        ax.set_xlabel('α (L2 weight)', fontsize=11)
        ax.set_ylabel('Mean AUC Drop', fontsize=11)
        ax.set_title(f'{tgt}: Attack Effectiveness vs α', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_xscale('log')

        # Right: L2 vs α
        ax = axes[row, 1]
        for attack, color, marker in [('Distilled','#2196F3','o'), ('Generic_AE','#FF9800','s')]:
            mask = (df['target']==tgt) & (df['attack']==attack)
            agg = df.loc[mask].groupby('alpha').agg({'Drop':'mean','L2':'mean'}).reset_index()
            ax.plot(agg['alpha'], agg['L2'], f'{marker}-', color=color, linewidth=2.5,
                    markersize=10, label=attack)
        ax.set_xlabel('α (L2 weight)', fontsize=11)
        ax.set_ylabel('Mean L2 Norm', fontsize=11)
        ax.set_title(f'{tgt}: Perturbation Size vs α', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10); ax.grid(alpha=0.3)
        ax.set_xscale('log')

    fig.suptitle('Distilled vs Generic AE at Different α\n'
                 'Lower α → more perturbation allowed → bigger drops',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0,0,1,0.93])
    out = os.path.join(PLOT_DIR, "distillation_alpha_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {out}")

    # Summary
    print("\n"+"="*70)
    print("ALPHA SWEEP SUMMARY")
    print("="*70)
    for tgt in ['IForest', 'PCA']:
        print(f"\n{tgt}:")
        print(f"  {'α':<6} {'Distilled Drop':<16} {'AE Drop':<16} {'Dist L2':<12} {'AE L2':<12}")
        for alpha in ALPHAS:
            d_mask = (df['target']==tgt)&(df['attack']=='Distilled')&(df['alpha']==alpha)
            a_mask = (df['target']==tgt)&(df['attack']=='Generic_AE')&(df['alpha']==alpha)
            dd = df.loc[d_mask,'Drop'].mean()
            ad = df.loc[a_mask,'Drop'].mean()
            dl = df.loc[d_mask,'L2'].mean()
            al = df.loc[a_mask,'L2'].mean()
            print(f"  {alpha:<6.2f} {dd:+.4f}          {ad:+.4f}          {dl:.5f}      {al:.5f}")


if __name__ == "__main__":
    main()
