"""Train the small MLP once and save its weights (used by the dashboard replay).
Needs scikit-learn only for this step:  python3 train_weights.py"""
import numpy as np, fhss_sim as fs
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
rng = np.random.default_rng(2026)
X, y, g = fs.gen_training(rng)
keep = []
for c in range(3):
    ii = np.where(y == c)[0]; keep += list(rng.choice(ii, min(6000, len(ii)), replace=False))
keep = np.array(keep); sc = StandardScaler().fit(X[keep])
sk = MLPClassifier((16, 8), max_iter=400, random_state=1, early_stopping=True).fit(sc.transform(X[keep]), y[keep])
np.savez("classifier_weights.npz", mu=sc.mean_, sd=sc.scale_, W0=sk.coefs_[0], b0=sk.intercepts_[0],
         W1=sk.coefs_[1], b1=sk.intercepts_[1], W2=sk.coefs_[2], b2=sk.intercepts_[2])
print("saved classifier_weights.npz, train acc", sk.score(sc.transform(X[keep]), y[keep]))
