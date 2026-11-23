import os,sys
os.environ["OMP_NUM_THREADS"]="3"
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
def bytes_to_unicode():
    bs=list(range(33,127))+list(range(161,173))+list(range(174,256)); cs=bs[:]; n=0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256+n); n+=1
    return dict(zip(bs,map(chr,cs)))
torch.set_num_threads(3)
NS=[1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256]
NT=int(sys.argv[1]); CH=100
tok=AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
m=AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").eval()
txt=open("../../data/text/TinyStoriesV2-GPT4-valid.txt",encoding="utf8").read()
docs=[d.strip() for d in txt.split("<|endoftext|>") if d.strip()]
ids=[tok(d)["input_ids"] for d in docs]
b2u=bytes_to_unicode(); u2b={v:k for k,v in b2u.items()}
V=len(tok); tb=np.zeros(max(V,50257),dtype=np.int32)
for t,i in tok.get_vocab().items(): tb[i]=len(bytes(u2b[c] for c in t)) if all(c in u2b for c in t) else 0
elig=[(d,p) for d,x in enumerate(ids) for p in range(256,len(x))]
rng=np.random.default_rng(0)
sel=rng.permutation(len(elig))[:NT]
T=[elig[i] for i in sel]
doc=np.array([d for d,_ in T]); pos=np.array([p for _,p in T])
tgt=np.array([ids[d][p] for d,p in T])
ctxb=np.array([[tb[ids[d][p-n:p]].sum() for n in NS] for d,p in T])
loss=np.full((NT,len(NS)),np.nan,dtype=np.float32)
start=0
if os.path.exists("raw.npz"):
  o=np.load("raw.npz"); start=int(o["done"]); loss[:start]=o["loss"][:start]
for c in range(start,NT,CH):
  sl=slice(c,min(c+CH,NT))
  for k,n in enumerate(NS):
    x=torch.tensor([ids[d][p-n:p] for d,p in T[sl]])
    with torch.no_grad():
      h=m.transformer(x).last_hidden_state[:,-1]
      lp=torch.log_softmax(m.lm_head(h).float(),-1)
    loss[sl,k]=-lp[torch.arange(len(x)),torch.tensor(tgt[sl])].numpy()
  np.savez("raw.npz",loss=loss,doc=doc,pos=pos,tgt=tgt,tgtbytes=tb[tgt],ctxb=ctxb,NS=NS,done=min(c+CH,NT))
  print(min(c+CH,NT),flush=True)
