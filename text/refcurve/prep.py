import os
os.environ["OMP_NUM_THREADS"]="3"
import numpy as np, torch, time
from transformers import AutoTokenizer, AutoModelForCausalLM
torch.set_num_threads(3)
tok=AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
m=AutoModelForCausalLM.from_pretrained("roneneldan/TinyStories-33M").eval()
print(m.config.max_position_embeddings, sum(p.numel() for p in m.parameters())/1e6)
txt=open("../../data/text/TinyStoriesV2-GPT4-valid.txt",encoding="utf8").read()
docs=[d.strip() for d in txt.split("<|endoftext|>") if d.strip()]
print(len(docs))
ids=[tok(d)["input_ids"] for d in docs]
L=np.array([len(i) for i in ids]); print(L.mean(),(L>256).sum(),np.maximum(L-256,0).sum())
for n in (256,64):
  x=torch.randint(0,50000,(16,n))
  t=time.time()
  with torch.no_grad(): m.transformer(x)
  print(n,(time.time()-t)/16)
