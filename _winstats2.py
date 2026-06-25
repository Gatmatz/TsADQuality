import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
sys.path.insert(0,"."); sys.path.insert(0,"src")
from data_loader import load_tsb_file
def acf_fft(x,nl):
    x=np.asarray(x,float); x=x-x.mean(); n=len(x)
    fs=2**int(np.ceil(np.log2(2*n-1))); X=np.fft.rfft(x,fs)
    a=np.fft.irfft(X*np.conjugate(X),fs)[:nl+1]/n; return a/a[0]
def lmax(a): return np.where((a[1:-1]>a[:-2])&(a[1:-1]>a[2:]))[0]+1
def find_length(d):
    if d.ndim>1: return 0
    d=d[:min(20000,len(d))]; ac=acf_fft(d,400)[3:]; lm=lmax(ac)
    if len(lm)==0: return 100
    mx=lm[int(np.argmax(ac[lm]))]
    return 100 if (mx<3 or mx>300) else mx+3
def zs(x):
    x=np.asarray(x,float); s=x.std(); return (x-x.mean())/s if s>0 else x-x.mean()
idx={}
for dp,_,fns in os.walk("TSB-UAD/data"):
    for fn in fns:
        if fn.endswith(".out"): idx.setdefault(fn,os.path.join(dp,fn))
bl=pd.read_csv("results/tables/baseline_final_subset.csv")
files148=sorted(bl["file"].unique())
def stats(files,floor):
    wins=[]
    for f in files:
        p=idx.get(f)
        if p is None: continue
        try:
            d,l,_=load_tsb_file(p); v=int(find_length(zs(d)))
            wins.append(max(v,10) if floor else v)
        except: pass
    w=pd.Series(wins)
    return "n=%d min=%d Q1=%.1f med=%.1f mean=%.1f Q3=%.1f max=%d"%(len(w),w.min(),w.quantile(.25),w.median(),w.mean(),w.quantile(.75),w.max())
print("148 floor=10 :", stats(files148,True))
print("148 NO floor :", stats(files148,False))
