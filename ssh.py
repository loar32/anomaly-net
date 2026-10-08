"""SSH-модель: признаки из auth.log, автоэнкодер на первых днях, проверка на остальных."""
import re, sys, json
import numpy as np, pandas as pd
from logs import entropy

FEATS = ["fail_count", "uniq_ips", "invalid_user_ratio", "user_entropy", "new_ip_ratio", "top_ip_share"]
FAIL = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S* \S+ sshd(?:-session)?\[\d+\]: Failed \S+ for (invalid user )?(\S+) from (\S+)")
OK = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S* \S+ sshd(?:-session)?\[\d+\]: Accepted \S+ for (\S+) from (\S+)")


def parse(path):
    ev = []
    for ln in open(path, errors="replace"):
        m = FAIL.match(ln)
        if m: ev.append((m[1], m[4], m[3], bool(m[2]), 0)); continue
        m = OK.match(ln)
        if m: ev.append((m[1], m[3], m[2], False, 1))
    d = pd.DataFrame(ev, columns=["ts", "ip", "user", "invalid", "ok"]); d["ts"] = pd.to_datetime(d.ts)
    return d


def windows(d, freq="5min"):
    seen, rows = set(), []
    for t, g in d.groupby(pd.Grouper(key="ts", freq=freq)):   # пустые окна тоже нужны: тишина — это норма
        f = g[g.ok == 0]; ips = set(f.ip)
        rows.append(dict(t=t, fail_count=len(f), uniq_ips=len(ips),
                         invalid_user_ratio=f.invalid.mean() if len(f) else 0.0,
                         user_entropy=entropy(f.user) if len(f) else 0.0,
                         new_ip_ratio=len(ips - seen) / len(ips) if ips else 0.0,
                         top_ip_share=f.ip.value_counts().iloc[0] / len(f) if len(f) else 0.0,
                         accepted_count=int(g.ok.sum())))  # не в модели, нужен правилу входов
        seen |= ips
    return pd.DataFrame(rows).set_index("t")


def prep(w, stats):
    mean, std = stats
    z = (w[FEATS].values - mean) / (std + 0.1)
    return (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32)


if __name__ == "__main__":
    import torch, torch.nn as nn
    torch.manual_seed(0)
    w = windows(parse("logs/auth.log"))
    cut = w.index[0] + pd.Timedelta(days=3)
    tr, te = w[w.index < cut], w[w.index >= cut]
    stats = (tr[FEATS].mean().values, tr[FEATS].std().values)
    Xtr, Xte = prep(tr, stats), prep(te, stats)
    print(f"окон: train {len(tr)}, test {len(te)}")
    print(tr[FEATS].describe().loc[["mean", "50%", "max"]].round(2).to_string())

    d = len(FEATS)
    ae = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 3), nn.ReLU(), nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, d))
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    X = torch.tensor(Xtr)
    for ep in range(400):
        for b in torch.randperm(len(X)).split(64):
            opt.zero_grad(); nn.functional.mse_loss(ae(X[b]), X[b]).backward(); opt.step()
    score = lambda A: ((ae(torch.tensor(A)) - torch.tensor(A)) ** 2).mean(1).detach().numpy()
    thr = np.percentile(score(Xtr), 99)
    torch.onnx.export(ae, torch.zeros(1, d), "model_ssh.onnx", input_names=["x"], output_names=["recon"],
                      dynamic_axes={"x": {0: "n"}, "recon": {0: "n"}}, dynamo=False)
    json.dump({"threshold": float(thr), "features": FEATS, "mean": stats[0].tolist(), "std": stats[1].tolist()},
              open("model_ssh.json", "w"), indent=1)
    s = score(Xte); flag = s > thr
    print(f"порог {thr:.3f}; на test помечено {flag.mean():.1%} окон")
    print("топ-8 аномальных окон test:")
    print(te.assign(score=s).sort_values("score", ascending=False).head(8).round(2).to_string())

    # искусственные всплески поверх реальных окон test
    base = te.sample(300, random_state=0).copy()
    def burst(b, kind):
        b = b.copy()
        if kind == "brute_1ip":    b.fail_count *= 0 ; b.fail_count += 600; b.uniq_ips = 1; b.top_ip_share = 1.0; b.new_ip_ratio = 1.0
        if kind == "botnet":       b.fail_count += 800; b.uniq_ips += 200; b.new_ip_ratio = 1.0; b.top_ip_share = 0.01
        return b
    for k in ["brute_1ip", "botnet"]:
        print(f"{k}: ловит {(score(prep(burst(base, k), stats)) > thr).mean():.0%}")
