"""Инференс: python detect.py [auth.log] [--recent МИНУТ]
С --recent печатает только свежие находки (для cron); если заданы TG_TOKEN и TG_CHAT, шлёт их в Telegram."""
import json, os, sys, urllib.parse, urllib.request
import numpy as np, onnxruntime as ort, pandas as pd
from ssh import parse, windows, FEATS

RU = os.environ.get("ANOMALY_LANG", "en").lower().startswith("ru")
T = dict(
    summary=("окон {n}, аномальных {m}", "windows {n}, anomalous {m}"),
    bad=("аномальных окон {n}, пик score {peak:.2f}, неудач {f}, IP {ips}",
         "anomalous windows {n}, peak score {peak:.2f}, failed {f}, IPs {ips}"),
    login=("вход {user} с нового IP {ip} в {ts}", "login {user} from new IP {ip} at {ts}"),
    newhdr=("успешные входы с новых IP:\n", "successful logins from new IPs:\n"),
)
tr = lambda k, **kw: T[k][0 if RU else 1].format(**kw)

args = sys.argv[1:]
recent = int(args.pop(args.index("--recent") + 1)) if "--recent" in args else None
if recent is not None: args.remove("--recent")
path = args[0] if args else "/var/log/auth.log"

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "model_ssh.json")))
d = parse(path); w = windows(d)
z = (w[FEATS].values - np.array(cfg["mean"])) / (np.array(cfg["std"]) + 0.1)
x = (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32)
rec = ort.InferenceSession(os.path.join(here, "model_ssh.onnx")).run(None, {"x": x})[0]
w["score"] = ((rec - x) ** 2).mean(1)
bad = w[w.score > cfg["threshold"]]

# правило: успешный вход с IP, которого не было в первой половине лога
ok = d[d.ok == 1]; known = set(ok.ip[ok.ts < d.ts.min() + (d.ts.max() - d.ts.min()) / 2])
new = ok[~ok.ip.isin(known)].drop_duplicates("ip")

if recent is not None:
    since = d.ts.max() - pd.Timedelta(minutes=recent)
    bad, new = bad[bad.index >= since], new[new.ts >= since]
    out = []
    if len(bad): out.append(tr("bad", n=len(bad), peak=bad.score.max(), f=int(bad.fail_count.sum()), ips=int(bad.uniq_ips.max())))
    for r in new.itertuples(): out.append(tr("login", user=r.user, ip=r.ip, ts=r.ts))
    if out:
        msg = "[anomaly-net] " + "; ".join(out); print(msg)
        if os.environ.get("TG_TOKEN") and os.environ.get("TG_CHAT"):
            urllib.request.urlopen(f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
                                   urllib.parse.urlencode({"chat_id": os.environ["TG_CHAT"], "text": msg}).encode(), timeout=15)
    sys.exit(0)

print(tr("summary", n=len(w), m=len(bad)))
print(bad.sort_values("score", ascending=False).head(10)[FEATS + ["score"]].round(2).to_string())
print(tr("newhdr"), new[["ts", "ip", "user"]].to_string(index=False))
