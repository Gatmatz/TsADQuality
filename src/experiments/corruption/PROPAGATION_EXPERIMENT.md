# Corruption Propagation Experiment

## Research Question
When corruption is injected into a **localized segment** of a time series, how far do the effects propagate on the anomaly scores produced by different AD models?

## Motivation
Existing robustness studies (e.g., TSB-UAD) measure **aggregate** performance drop (AUC-ROC) when data is corrupted. However, they do not examine **where** the anomaly scores change. This experiment fills that gap by measuring the **spatial/temporal propagation** of corruption effects.

## Hypothesis
The propagation radius depends on whether the model uses **local** (window-based) or **global** (dataset-level) features:
- **Global models** (IForest, PCA): corruption in one segment changes global statistics (mean, covariance, tree splits), causing score changes **everywhere**
- **Local models** (MatrixProfile, AE): corruption only affects nearby subsequences within a window, so propagation is **limited**
- **Neighborhood models** (LOF): intermediate — depends on how corruption reshuffles k-nearest neighbors

## Method

### Step 1: Baseline scores
- Load clean time series
- Run each model on clean data
- Store `scores_clean` (anomaly scores at every point)

### Step 2: Localized corruption
- Inject corruption in a **single segment** (10% of the time series, centered at the midpoint)
- Corruption types: white noise (SNR=10dB), point spikes
- Run each model on corrupted data
- Store `scores_corrupted`

### Step 3: Score difference
- Compute `score_diff[t] = |scores_corrupted[t] - scores_clean[t]|` at every time step t
- Normalize by max(score_diff) to get values in [0, 1]

### Step 4: Propagation analysis
- Divide the time series into zones relative to the corrupted segment:
  - **Corruption zone**: the 10% where corruption was injected
  - **Near zone**: 0-10% distance from corruption boundary
  - **Mid zone**: 10-30% distance
  - **Far zone**: >30% distance
- Compute mean score_diff in each zone
- **Propagation ratio** = (impact_zone_size) / (corruption_zone_size)
  - impact_zone = all points where score_diff > threshold (e.g., 5% of max)

## Models
| Model | Type | Expected propagation |
|-------|------|---------------------|
| IForest | Global (random partitioning) | High — splits depend on global feature distribution |
| PCA | Global (covariance-based) | High — eigenvectors change with any perturbation |
| MatrixProfile | Local (subsequence distances) | Low — z-normalized distance is local |
| AE (MLP2) | Local (sliding window reconstruction) | Low-Medium — limited to sliding_window size |
| LOF | Neighborhood (k-NN based) | Medium — chain reaction through neighbor reshuffling |

## Theoretical justification

### IForest
Isolation Forest builds binary trees using random splits on feature values. When corruption changes the distribution of a segment, the split thresholds learned during training become suboptimal. Since each tree uses global data, even dist changing their anomaly scores.

### PCA
PCA computes the covariance matrix of all sliding windows. A corrupted segment contributes corrupted windows that shift the principal components (eigenvectors). Since ALL points are projected onto these shifted components, the reconstruction error changes globally. This is explained by rank-1 perturbation theory: a perturbation of magnitude epsilon in the data matrix causes O(epsilon / eigengap) change in eigenvectors.

### MatrixProfile
Matrix Profile computes the z-normalized Euclidean distance between each subsequence and its nearest neighbor. Corruption at position X only affects subsequences that overlap with [X - window, X + window]. Subsequences far from the corruption zone have identical distances, so their scores are unchanged.

### AE (Autoencoder)
The autoencoder scores each sliding window independently. Windows that don't overlap with the corrupted segment contain identical data, so their reconstruction error is unchanged. Propagation is bounded by sliding_window size.

### LOF
Local Outlier Factor depends on k-nearest neighbor distances. When corruption changes the feature vector of some points, their neighbors may change. This can create a chain reaction: point A gets a new neighbor B, which changes B's local density, which affects C's LOF score, etc. The propagation depends on the data density and k.

## Output filesant points may end up in different tree partitions,
- `propagation_results.csv`: per-file, per-model, per-corruption raw results
- `propagation_summary.csv`: aggregated propagation ratios (corruption_type x model)
- `propagation_curves.csv`: mean score_diff as function of distance from corruption zone

## Expected result (example table)

| Corruption | IForest | PCA | LOF | MP | AE |
|-----------|---------|-----|-----|----|----|
| Noise | 4.2x | 5.1x | 2.8x | 1.2x | 1.4x |
| Spikes | 3.5x | 4.8x | 2.3x | 1.1x | 1.3x |

Where Nx = propagation ratio (impact zone is N times larger than corruption zone).

## Practical implication
"If you detect corrupted sensor data from 14:00-15:00, you should distrust anomaly scores until 18:00 when using IForest, but only until 15:30 when using MatrixProfile."
