import sys, torch, numpy as np, librosa
from src.ssl_models.byol_module import BYOLModule
from sklearn.ensemble import IsolationForest

wav = sys.argv[1]
sr = 16000
y,_ = librosa.load(wav, sr=sr)
mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
logmel = librosa.power_to_db(mel)[None,None]

model = BYOLModule.load_from_checkpoint("checkpoints/byol.ckpt")
model.eval()
with torch.no_grad():
    emb = model.online_enc(torch.tensor(logmel).float()).numpy()

# simple one-class model trained only on normals
from pathlib import Path
import numpy as np
X_norm=[np.load(f)["emb"][0] for f in Path("data/embeddings").glob("normal_*.npz")]
clf = IsolationForest().fit(X_norm)
score = -clf.score_samples(emb)[0]
print("Anomaly score:", score)
