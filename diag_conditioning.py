import torch, numpy as np
from otfs_data import make_federation, SCENARIOS
from models import PFDUNHyper
from federated import federated_train

dev="cpu"; N=M=16; L=N*M; Q=int(0.6*L)
cl = make_federation(4,N,M,Q,10.0,160,frames=1,seed=0)
m = PFDUNHyper(L,T=8,use_hyper=True).to(dev)
m,_ = federated_train(m,cl,25,1,1e-3,64,1.0,dev,seed=0,patience=4,eval_every=2)
m.eval()
# Do the hypernetwork outputs actually DIFFER across scenarios?
outs=[]
with torch.no_grad():
    for c in cl:
        s = c.S.mean(0,keepdim=True)
        g,l,sc,sh = m.hyper(s)
        outs.append(torch.cat([g.flatten(),l.flatten(),sc.flatten(),sh.flatten()]))
O = torch.stack(outs)
print("\n--- hypernetwork output across the 4 scenarios ---")
print("per-scenario mean |output| :", [f"{o.abs().mean():.4f}" for o in O])
spread = O.std(0).mean().item(); mag = O.abs().mean().item()
print(f"across-scenario std   : {spread:.6f}")
print(f"overall magnitude     : {mag:.6f}")
print(f"RELATIVE VARIATION    : {spread/max(mag,1e-9)*100:.2f}%")
# also: how different are the input embeddings themselves?
S = torch.stack([c.S.mean(0) for c in cl])
print("\nembedding s per scenario:")
for name,row in zip(SCENARIOS.keys(),S): print(f"  {name:12s}", np.round(row.numpy(),3))
print("embedding across-scenario std:", np.round(S.std(0).numpy(),4))
