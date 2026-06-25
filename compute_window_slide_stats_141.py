"""Window length (find_length, floor 10) + slide stats για τις 141 σειρές του swap_segment.
Η find_length αναπαράγεται πιστά (FFT ACF, base=3, range 3..300, default 100) χωρίς statsmodels."""
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from data_loader import load_tsb_file


def acf_fft(x, nlags):
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    fsize = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    X = np.fft.rfft(x, fsize)
    acov = np.fft.irfft(X * np.conjugate(X), fsize)[: nlags + 1]
    acov = acov / n            # statsmodels adjusted=False
    return acov / acov[0]


def local_maxima(a):           # ισοδύναμο scipy argrelextrema(a, np.greater)
    return np.where((a[1:-1] > a[:-2]) & (a[1:-1] > a[2:]))[0] + 1


def find_length(data):
    if data.ndim > 1:
        return 0
    data = data[: min(20000, len(data))]
    base = 3
    auto_corr = acf_fft(data, 400)[base:]
    lm = local_maxima(auto_corr)
    if len(lm) == 0:
        return 100
    mx = lm[np.argmax(auto_corr[lm])]
    if mx < 3 or mx > 300:
        return 100
    return mx + base


def zscore(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / s if s > 0 else x - x.mean()


ck = pd.read_csv(os.path.join(ROOT, "results", "experiments", "swap_segment", "checkpoint.csv"), low_memory=False)
files = sorted(ck[ck["error"].isna()]["file"].unique())
print("files:", len(files))

index = {}
for dp, _, fns in os.walk(os.path.join(ROOT, "TSB-UAD", "data")):
    for fn in fns:
        if fn.endswith(".out"):
            index.setdefault(fn, os.path.join(dp, fn))

wins, missing = [], []
for f in files:
    p = index.get(f)
    if p is None:
        missing.append(f)
        continue
    try:
        data, label, _ = load_tsb_file(p)
        wins.append(max(int(find_length(zscore(data))), 10))
    except Exception:
        missing.append(f)

w = pd.Series(wins)
print("Window length (n=%d, floor=10):" % len(w))
print("  min=%d  Q1=%.1f  median=%.1f  mean=%.1f  Q3=%.1f  max=%d" % (
    w.min(), w.quantile(.25), w.median(), w.mean(), w.quantile(.75), w.max()))
print("Slide size: fixed = 1 (min=Q1=median=mean=Q3=max=1)")
print("missing/failed:", len(missing), missing[:5])
