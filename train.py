import json, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import IsolationForest
from data import *

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

train_p, val_p = make_profiles(10, 1), make_profiles(4, 2)   # val-серверы модель не видела
tr, va = build(train_p, seed=10), build(val_p, seed=20)

def featurize(df):
    parts, idx = [], []
    for sid, g in df.groupby("server_id", sort=False):
        calib = g[g.label == 0].iloc[:500]       # первые «спокойные» окна = калибровка сервера
        parts.append(zfeatures(g, server_stats(calib))); idx.append(g.index)
    return np.vstack(parts).astype(np.float32), np.concatenate(idx)

Xtr, itr = featurize(tr); Xva, iva = featurize(va)
tr, va = tr.loc[itr], va.loc[iva]
Xn = Xtr[tr.label.values == 0]
cut = int(len(Xn) * 0.9); perm = np.random.default_rng(0).permutation(len(Xn))
Xfit, Xes = Xn[perm[:cut]], Xn[perm[cut:]]       # early stopping по норме train-серверов

d = Xtr.shape[1]
model = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 8), nn.ReLU(),
                      nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, d)).to(dev)
opt = torch.optim.Adam(model.parameters(), 1e-3)
T = lambda a: torch.tensor(a, device=dev)
fit, es = T(Xfit), T(Xes)
best, bad, state = 1e9, 0, None
for ep in range(300):
    model.train()
    for b in torch.randperm(len(fit), device=dev).split(256):
        opt.zero_grad(); nn.functional.mse_loss(model(fit[b]), fit[b]).backward(); opt.step()
    model.eval()
    with torch.no_grad(): l = nn.functional.mse_loss(model(es), es).item()
    if l < best - 1e-5: best, bad, state = l, 0, {k: v.clone() for k, v in model.state_dict().items()}
    else:
        bad += 1
        if bad >= 20: break
model.load_state_dict(state); model.eval()
print(f"эпох: {ep+1}, лучший es-loss: {best:.4f}, device: {dev}")

def score(X):
    with torch.no_grad(): x = T(X); return ((model(x) - x) ** 2).mean(1).cpu().numpy()

# порог = 99-й перцентиль на норме val-серверов
vl = va.label.values
thr = np.percentile(score(Xva[vl == 0]), 99)
iso = IsolationForest(n_estimators=300, random_state=0).fit(Xn)
iso_s = lambda X: -iso.score_samples(X)
iso_thr = np.percentile(iso_s(Xva[vl == 0]), 99)

res = {}
for name, s, t in [("autoencoder", score(Xva), thr), ("isolation_forest", iso_s(Xva), iso_thr)]:
    pred = s > t
    r = {"fpr": float(pred[vl == 0].mean())}
    for a in ANOMALIES:
        r[a] = float(pred[va.anomaly_type.values == a].mean())
    r["recall_all"] = float(pred[vl == 1].mean())
    res[name] = r
df = pd.DataFrame(res).round(3)
print(df.to_string()); df.to_markdown = None

# ONNX + параметры для инференса
torch.onnx.export(model.cpu(), torch.zeros(1, d), "model_v1.onnx", input_names=["x"], output_names=["recon"],
                  dynamic_axes={"x": {0: "n"}, "recon": {0: "n"}}, dynamo=False)
json.dump({"threshold": float(thr), "features": FEATURES, "eps": EPS, "metrics": res},
          open("model_v1.json", "w"), indent=1)
