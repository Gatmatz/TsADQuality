"""Explanatory figure: How noise affects the ROC curve (AUC → 0.5)."""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "results", "plots")


def make_labels(n=2000, anomaly_ratio=0.1):
    labels = np.zeros(n, dtype=int)
    n_anom = int(n * anomaly_ratio)
    labels[:n_anom] = 1
    return labels


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    np.random.seed(42)

    n = 2000
    labels = make_labels(n, anomaly_ratio=0.1)

    # --- Panel 1: Clean Signal, inverse ranking (AUC ~ 0.20) ---
    # Anomalies get LOWER scores than normals → inverse ranking
    scores_clean = np.random.normal(0.55, 0.20, n)
    scores_clean[labels == 1] = np.random.normal(0.38, 0.18, labels.sum())
    scores_clean = np.clip(scores_clean, 0, 1)

    # --- Panel 2: Moderate Noise (AUC ~ 0.45) ---
    scores_moderate = 0.35 * scores_clean + 0.65 * np.random.uniform(0, 1, n)

    # --- Panel 3: High Noise / Pure Random (AUC ~ 0.50) ---
    scores_random = np.random.uniform(0, 1, n)

    panels = [
        (scores_clean, 'Clean Signal', '#2d6a2e', '-', 2.0,
         'Inverse Ranking', 'Good, but inverse ranking.'),
        (scores_moderate, 'Moderate Noise', '#d4a017', '-', 1.8,
         'Random Scores', 'Mixed signals'),
        (scores_random, 'High Noise', '#333333', '--', 1.5,
         'Pure Random', 'Complete noise'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    for ax, (scores, title, color, ls, lw, annotation, subtitle) in zip(axes, panels):
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        # Diagonal reference
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        # ROC curve
        ax.plot(fpr, tpr, color=color, linewidth=lw, linestyle=ls)

        # Annotation text (positioned to avoid overlap with curve)
        if annotation == 'Inverse Ranking':
            ax.text(0.05, 0.55, annotation, fontsize=11, color=color,
                    fontstyle='italic', fontweight='bold')
        elif annotation == 'Random Scores':
            ax.text(0.20, 0.88, annotation, fontsize=11, color=color,
                    fontstyle='italic', fontweight='bold')
        else:
            ax.text(0.35, 0.72, annotation, fontsize=11, color=color,
                    fontstyle='italic', fontweight='bold')

        # AUC text (bottom-right area)
        ax.text(0.55, 0.12, f'AUC = {roc_auc:.2f}', fontsize=14,
                fontweight='bold', transform=ax.transAxes)

        ax.set_xlabel('FPR', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlim(-0.02, 1.1)
        ax.set_ylim(-0.02, 1.05)

        # Subtitle below
        ax.text(0.5, -0.18, subtitle, fontsize=10, ha='center',
                transform=ax.transAxes, fontstyle='italic')

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(PLOTS_DIR, 'roc_explanation.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == '__main__':
    main()
