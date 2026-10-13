"""Инференс: python detect.py [auth.log] [--recent МИНУТ]
Читает auth.log.1 и auth.log: история нужна признаку «новые IP» и правилу входов, иначе после ротации всё «новое».
Окно аномально, если ошибка автоэнкодера выше порога или неудач больше, чем порог счётчика (всплеск объёма).
С --recent — режим cron: только завершённые окна и входы, о которых ещё не сообщали (state.json рядом со скриптом;
при первом запуске — за последние МИНУТ); если заданы TG_TOKEN и TG_CHAT, находки уходят в Telegram."""
import json, os, sys, urllib.parse, urllib.request
import numpy as np, onnxruntime as ort, pandas as pd
from ssh import parse, windows, prep, FEATS

RU = os.environ.get("ANOMALY_LANG", "ru").lower().startswith("ru")
T = dict(
    summary=("окон {n}, аномальных {m}", "windows {n}, anomalous {m}"),
    bad=("аномальных окон {n}, пик score {peak:.2f}, неудач {f}, IP {ips}",
         "anomalous windows {n}, peak score {peak:.2f}, failed {f}, IPs {ips}"),
    login=("вход {user} с нового IP {ip} в {ts}", "login {user} from new IP {ip} at {ts}"),
    firsthdr=("первый успешный вход с каждого IP:\n", "first successful login from each IP:\n"),
    tgfail=("не удалось отправить в Telegram: {e}", "failed to send to Telegram: {e}"),
)
tr = lambda k, **kw: T[k][0 if RU else 1].format(**kw)

args = sys.argv[1:]
recent = int(args.pop(args.index("--recent") + 1)) if "--recent" in args else None
if recent is not None: args.remove("--recent")
path = args[0] if args else "/var/log/auth.log"

here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "model_ssh.json")))
prev = path + ".1"
d = parse(*([prev] if os.path.exists(prev) else []), path); w = windows(d)
if len(w):
    x = prep(w, (np.array(cfg["mean"]), np.array(cfg["std"])))
    w["score"] = ((ort.InferenceSession(os.path.join(here, "model_ssh.onnx")).run(None, {"x": x})[0] - x) ** 2).mean(1)
else:
    w["score"] = np.zeros(0)
bad = w[(w.score > cfg["threshold"]) | (w.fail_count > cfg.get("count_threshold", np.inf))]
ok = d[d.ok == 1]; first = ok.drop_duplicates("ip")   # первый успешный вход с каждого IP в доступной истории

if recent is not None:
    state = os.path.join(here, "state.json")
    st = json.load(open(state)) if os.path.exists(state) else {}
    until = pd.Timestamp.now().floor("5min")          # окна до этой границы уже закончились
    since = pd.Timestamp(st["last"]) if "last" in st else (until - pd.Timedelta(minutes=recent)).floor("5min")
    bad = bad[(bad.index >= since) & (bad.index < until)]
    new = first[(first.ts >= since) & (first.ts < until) & ~first.ip.isin(st.get("known", []))]
    out = []
    if len(bad): out.append(tr("bad", n=len(bad), peak=bad.score.max(), f=int(bad.fail_count.sum()), ips=int(bad.uniq_ips.max())))
    for r in new.itertuples(): out.append(tr("login", user=r.user, ip=r.ip, ts=r.ts))
    if out:
        msg = "[anomaly-net] " + "; ".join(out); print(msg)
        if os.environ.get("TG_TOKEN") and os.environ.get("TG_CHAT"):
            try:
                urllib.request.urlopen(f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
                                       urllib.parse.urlencode({"chat_id": os.environ["TG_CHAT"], "text": msg}).encode(), timeout=15)
            except OSError as e:   # состояние не сохраняем: в следующий раз отправим снова
                print(tr("tgfail", e=e)); sys.exit(1)
    # known — IP, с которых уже были входы: так адрес не станет «новым», когда его вход уйдёт из ротированных логов
    json.dump({"last": str(until), "known": sorted(set(st.get("known", [])) | set(ok.ip[ok.ts < until]))}, open(state, "w"))
    sys.exit(0)

print(tr("summary", n=len(w), m=len(bad)))
print(bad.sort_values("score", ascending=False).head(10)[FEATS + ["score"]].round(2).to_string())
print(tr("firsthdr"), first[["ts", "ip", "user"]].to_string(index=False))
