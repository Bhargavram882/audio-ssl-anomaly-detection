import torch, numpy as np
from pathlib import Path
from ssl_models.byol_module import BYOLModule


CKPT = "checkpoints/byol.ckpt"
IN = Path("data/mels")
OUT = Path("data/embeddings"); OUT.mkdir(exist_ok=True)

model = BYOLModule.load_from_checkpoint(CKPT, lr=1e-3)
model.eval()

def embed(f):
    x = np.load(f)["feat"][None,None]
    with torch.no_grad():
        e = model.online_enc(torch.tensor(x).float()).numpy()
    np.savez_compressed(OUT/f.name, emb=e)

for f in IN.glob("*.npz"): embed(f)
print("Embeddings saved to", OUT)
