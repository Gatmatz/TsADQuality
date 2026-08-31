# TSB-AD local modifications

The TSB-AD benchmark is used as a vendored clone of
<https://github.com/TheDatumOrg/TSB-AD> and is **not** committed to this
repository (it carries ~358 MB of datasets and has its own git history).
This directory preserves the local changes made to it, so the environment
can be reconstructed.

Pinned upstream commit: `e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48`

## Contents

| File | Purpose |
| --- | --- |
| `local_fixes.patch` | Diff against tracked upstream files |
| `AE_2.py` | New model file (untracked upstream), goes to `TSB_AD/models/AE_2.py` |

## What the changes do

- **`TSB_AD/model_wrapper.py`** — adds `run_AutoEncoder_2()`, a wrapper around
  the `AE_2` model. The upstream `run_AutoEncoder` crashes during
  semi-supervised inference because of a scaler shape mismatch; `AE_2` is the
  corrected implementation.
- **`TSB_AD/models/KMeansAD.py`** — comments out per-window debug `print()`
  calls that flood stdout during the 350-series sweeps. No behavioural change.

## Reapplying

```bash
git clone https://github.com/TheDatumOrg/TSB-AD.git
cd TSB-AD
git checkout e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48
cp ../vendor_patches/tsb_ad/AE_2.py TSB_AD/models/AE_2.py
git apply ../vendor_patches/tsb_ad/local_fixes.patch
pip install -e .
```

Datasets are fetched separately with `download_tsb_ad_data.py`.
