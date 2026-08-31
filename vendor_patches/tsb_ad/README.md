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
# from the repository root
git clone https://github.com/TheDatumOrg/TSB-AD.git
git -C TSB-AD checkout e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48
cp vendor_patches/tsb_ad/AE_2.py TSB-AD/TSB_AD/models/AE_2.py
git -C TSB-AD apply ../vendor_patches/tsb_ad/local_fixes.patch
python download_tsb_ad_data.py
```

## Do NOT `pip install` TSB-AD

TSB-AD is **not** installed as a package, and installing it would break the
environment. Every runner adds the clone to `sys.path` itself:

```python
sys.path.insert(0, os.path.join(project_root, 'TSB-AD'))
from TSB_AD.model_wrapper import run_Unsupervise_AD
```

So the clone only has to sit at `TSB-AD/` in the repository root.

`pip install -e TSB-AD/` (or `-r TSB-AD/requirements.txt`) would pin
`numpy<2.0`, downgrading the `numpy 2.2.x` this project actually runs on and
dragging torch, scikit-learn, pandas and TSB-UAD down with it. The results
were produced on numpy 2.x; keep it that way.

Its dependencies are already satisfied by the project environment, with the
single exception of `torchinfo`, which is declared upstream but not imported
by any model used here.
