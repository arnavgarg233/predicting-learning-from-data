import numpy as np, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
r=np.load("raw.npz"); D=int(r["done"]); NS=[int(v) for v in r["NS"]]
L=r["loss"][:D]/np.log(2); doc=r["doc"][:D]; tb=r["tgtbytes"][:D].astype(float); cb=r["ctxb"][:D]
ud,inv=np.unique(doc,return_inverse=True); nd=len(ud)
def agg(f,w):  # per-doc sums
    return np.bincount(inv,weights=f,minlength=nd), np.bincount(inv,weights=w,minlength=nd)
def stat(Lcol_fn):
    pass
rng=np.random.default_rng(1); B=1000
bi=rng.integers(0,nd,(B,nd))
cnt=np.bincount(inv,minlength=nd).astype(float)
S_tb=np.bincount(inv,weights=tb,minlength=nd)
def boot_ratio(num,den): return (num[bi].sum(1)/den[bi].sum(1))
bpt=L.mean(0); mbt=tb.mean()
res={"n_targets":int(D),"n_docs":int(nd),"mean_bytes_per_target_token":float(mbt),"rows":[]}
S=[np.bincount(inv,weights=L[:,k],minlength=nd) for k in range(len(NS))]
bpb=[]; bpbse=[]; bptse=[]
for k,n in enumerate(NS):
    bt=boot_ratio(S[k],cnt); bb=boot_ratio(S[k],S_tb)
    bpb.append(S[k].sum()/S_tb.sum()); bpbse.append(bb.std()); bptse.append(bt.std())
bpb=np.array(bpb)
ib=NS.index(256)
dS=[S[k]-S[ib] for k in range(len(NS))]
dbpb=np.array([bpb[k]-bpb[ib] for k in range(len(NS))])
dse=[boot_ratio(dS[k],S_tb).std() for k in range(len(NS))]
for k,n in enumerate(NS):
    res["rows"].append(dict(n_tokens=int(n),bits_per_token=float(bpt[k]),bpt_se=float(bptse[k]),bits_per_byte=float(bpb[k]),bpb_se=float(bpbse[k]),
      delta_bits_per_byte=float(dbpb[k]),delta_se_paired=float(dse[k]),context_bytes_nominal=float(n*mbt),context_bytes_exact_mean=float(cb[:,k].mean())))
def fit(lo,hi,d=dbpb,Sx=None):
    ks=[k for k,n in enumerate(NS) if lo<=n<=hi]
    x=np.log([NS[k] for k in ks]); y=np.log(d[ks]); p=np.polyfit(x,y,1); return -p[0],np.exp(p[1])
res["fits"]={}
for lo,hi in [(4,64),(2,32),(8,128)]:
    g,a=fit(lo,hi)
    gb=[]
    for b in range(B):
        idx=bi[b]; den=S_tb[idx].sum(); base=S[ib][idx].sum()/den
        d=np.array([S[k][idx].sum()/den-base for k in range(len(NS))])
        if (d[[k for k,n in enumerate(NS) if lo<=n<=hi]]<=0).any(): continue
        gb.append(fit(lo,hi,d)[0])
    res["fits"][f"{lo}-{hi}"]=dict(gamma=float(g),a_bits_per_byte=float(a),gamma_boot_se=float(np.std(gb)),n_boot_valid=len(gb))
# fit in context-bytes units too
res["note"]="Reference-model (TinyStories-33M) upper bound on true entropy curve; targets restricted to doc positions >=256."
json.dump(res,open("refcurve.json","w"),indent=1)
fig,ax=plt.subplots(1,2,figsize=(10,4))
x=[n*mbt for n in NS]
ax[0].errorbar(x,bpb,bpbse,marker="o"); ax[0].set_xscale("log"); ax[0].set_xlabel("context (bytes)"); ax[0].set_ylabel("bits/byte"); ax[0].set_title("h(n), TinyStories-33M")
ax[1].errorbar(x[:-1],dbpb[:-1],dse[:-1],marker="o")
g=res["fits"]["4-64"]; xx=np.array(x[:-1]); ax[1].plot(xx,g["a_bits_per_byte"]*(xx/mbt)**-g["gamma"],"--",label=f"4-64 gamma={g['gamma']:.2f}")
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].set_xlabel("context (bytes)"); ax[1].set_ylabel("h(n)-h(256) bits/byte"); ax[1].legend()
plt.tight_layout(); plt.savefig("refcurve.png",dpi=150)
for r_ in res["rows"]: print(r_["n_tokens"],round(r_["bits_per_token"],3),round(r_["bits_per_byte"],4),round(r_["bpb_se"],4),round(r_["delta_bits_per_byte"],4),round(r_["delta_se_paired"],4),round(r_["context_bytes_exact_mean"],1))
print(mbt,nd,D,res["fits"])
