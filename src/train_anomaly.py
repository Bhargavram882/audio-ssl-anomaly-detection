import numpy as np
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

IN = Path("data/embeddings")
X_norm, X_anom = [], []
for f in IN.glob("*.npz"):
    x = np.load(f)["emb"][0]
    if f.name.startswith("normal"): X_norm.append(x)
    else: X_anom.append(x)

clf = IsolationForest(contamination=0.5).fit(X_norm)
scores = -clf.score_samples(X_anom+X_norm)
y = [1]*len(X_anom)+[0]*len(X_norm)
auc = roc_auc_score(y, scores)
print("IsolationForest AUC:", auc)
