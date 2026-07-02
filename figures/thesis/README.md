# Ενημερωμένα Σχήματα Πτυχιακής

Όλα τα 35 Σχήματα του Κεφαλαίου 4, με:
- **Έντονες baseline γραμμές** (linewidth 1→2.0, alpha 0.5→0.85)
- **Ενιαία παλέτα αλγορίθμων** σε όλα τα plots

| Σχ. | Όνομα PNG | Περιγραφή |
|---|---|---|
| 4.1 | Figure_4_1.png | AUC-ROC vs SNR (per algorithm) |
| 4.2 | Figure_4_2.png | AUC-PR vs SNR (multi-metric panel) |
| 4.3 | Figure_4_3.png | Recall/Drop vs SNR |
| 4.4 | Figure_4_4.png | Spikes AUC-PR multi-metric |
| 4.5 | Figure_4_5.png | Spikes σχετική πτώση AUC-PR |
| 4.6 | Figure_4_6.png | Spikes AUC-ROC |
| 4.7 | Figure_4_7.png | Point swap multi-metric |
| 4.8 | Figure_4_8.png | Segment Swap multi-metric vs fraction |
| 4.9 | Figure_4_9.png | Segment Swap heatmap |
| 4.10 | Figure_4_10.png | Segment Swap σχετική πτώση |
| 4.11 | Figure_4_11.png | Permutation multi-metric |
| 4.12 | Figure_4_12.png | Permutation R-AUC-ROC/VUS-ROC/Affiliation |
| 4.13 | Figure_4_13.png | Swap drops comparison |
| 4.14 | Figure_4_14.png | Freeze multi-metric |
| 4.15-17 | Figure_4_15/16/17.png | Freeze heatmaps (AUC-ROC/AUC-PR/Recall) |
| 4.18-21 | Figure_4_18/19/20/21.png | Freeze per-algorithm (IForest/LOF/MP/AE) |
| 4.22 | Figure_4_22.png | MCAR point missing |
| 4.23 | Figure_4_23.png | MNAR_extreme point |
| 4.24 | Figure_4_24.png | MNAR_high point |
| 4.25 | Figure_4_25.png | Lost anomalies scatter |
| 4.26 | Figure_4_26.png | MCAR burst |
| 4.27 | Figure_4_27.png | MNAR_extreme burst |
| 4.28 | Figure_4_28.png | MNAR_high burst |
| 4.29 | Figure_4_29.png | Gilbert-Elliott heatmap |
| 4.30 | Figure_4_30.png | Drops 20% missing |
| 4.31-35 | Figure_4_31..35.png | Compound corruptions (Shapley κ.λπ.) |

## Πώς να τα βάλω στο docx

1. Άνοιξε το docx
2. Κλικ στο παλιό σχήμα → δεξί κλικ → Change Picture → from File
3. Επίλεξε το αντίστοιχο `Figure_4_X.png`

## ⚠️ Επιβεβαίωση χρειάζεται

Σε μερικά σχήματα η αντιστοίχιση είναι **εκτίμηση** (βασισμένη σε keywords από τις λεζάντες). Συγκεκριμένα έλεγξε αν ταιριάζουν με αυτά που έχεις:

- **Σχ. 4.2, 4.3** — επιβεβαίωσε αν είναι η σωστή έκδοση (το 4.2 είναι multi-metric panel)
- **Σχ. 4.12** — Permutation με R-AUC-ROC/VUS-ROC/Affiliation: μπορεί να μην ταιριάζει ακριβώς
- **Σχ. 4.15-17** — και τα 3 δείχνουν στο ίδιο heatmap PNG (πιθανώς multi-panel)
- **Σχ. 4.28** — best guess: mnar_AUC_ROC.png
- **Σχ. 4.33-34** — best guess για compound corruptions Shapley plots

Αν κάποιο δεν ταιριάζει, πες μου ποιο και θα το ξαναψάξω από τους άλλους φακέλους.
