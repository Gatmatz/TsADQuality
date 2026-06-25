"""Create a single PDF with key plots from all experiments."""
import os
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_PDF = os.path.join(PROJECT_ROOT, "results", "analysis", "experiment_plots.pdf")

# One key plot per experiment (the 12 we keep)
PLOTS = [
    ("1. White Noise (SNR)", "white_noise_snr/plots/snr_AUC_ROC.png"),
    ("2. Missing — True Impact", "missing_true_impact/plots/true_impact_AUC_ROC.png"),
    ("2b. Missing — True Impact Heatmap", "missing_true_impact/plots/true_impact_heatmap.png"),
    ("3. Missing — Masking", "missing_masking/plots/masking_AUC_ROC.png"),
    ("4. Missing — MCAR vs MNAR (Point)", "missing_mnar/plots/mnar_AUC_ROC.png"),
    ("4b. Missing — MCAR vs MNAR Anomalies Lost", "missing_mnar/plots/mnar_anomalies_lost.png"),
    ("4c. Missing — MCAR vs MNAR Heatmap", "missing_mnar/plots/mnar_heatmap.png"),
    ("5. Gilbert-Elliott", "gilbert_elliott_true_impact/plots/ge_true_impact_heatmap.png"),
    ("6. Freeze", "freeze/plots/freeze_AUC_ROC.png"),
    ("6b. Freeze Heatmap", "freeze/plots/freeze_heatmap.png"),
    ("7. Point Swap", "swap_point/plots/swap_point_AUC_ROC.png"),
    ("8. Segment Swap", "swap_segment/plots/swap_segment_AUC_ROC.png"),
    ("8b. Segment Swap Heatmap", "swap_segment/plots/swap_segment_heatmap.png"),
    ("9. Permutation", "swap_permutation/plots/permutation_AUC_ROC_bar.png"),
    ("10. Spikes (Normal Only)", "spikes_normal_only/plots/spikes_normal_AUC_ROC.png"),
    ("10b. Spikes Heatmap", "spikes_normal_only/plots/spikes_normal_heatmap.png"),
    ("10c. Spikes Inverted Rate", "spikes_normal_only/plots/spikes_normal_inverted_rate.png"),
    ("11. Noise Position", "noise_position/plots/noise_position_AUC_ROC.png"),
]

RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "experiments")


def main():
    os.makedirs(os.path.dirname(OUTPUT_PDF), exist_ok=True)

    with PdfPages(OUTPUT_PDF) as pdf:
        for title, rel_path in PLOTS:
            full_path = os.path.join(RESULTS_BASE, rel_path)
            if not os.path.exists(full_path):
                print(f"  [SKIP] {title}: {rel_path}")
                continue

            img = mpimg.imread(full_path)
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
            fig.tight_layout()
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  [OK] {title}")

    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == '__main__':
    main()
