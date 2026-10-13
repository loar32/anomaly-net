"""SSH-модель: признаки из auth.log, автоэнкодер на нормальном периоде, порог и проверка на следующих днях.
Обучение: python ssh.py [--from ГГГГ-ММ-ДД] — логи берутся из logs/auth.log* (вместе с .gz),
норма — окна начиная с --from (например, со дня после включения fail2ban), всё раньше — прежний режим."""
import glob, gzip, json, os, re, sys
from datetime import datetime
import numpy as np, pandas as pd
from logs import SSHD, entropy, stamp

FEATS = ["fail_count", "uniq_ips", "invalid_user_ratio", "user_entropy", "new_ip_ratio", "top_ip_share"]
FAIL = re.compile(SSHD + r"Failed \S+ for (invalid user )?(\S*) from (\S+)")   # логин бывает пустым
OK = re.compile(SSHD + r"Accepted \S+ for (\S+) from (\S+)")
EPS = 0.1
Q = 99.5   # перцентиль порога: на ~270 окнах отложенных суток 99-й определяют 2–3 окна, и он скачет


def parse(*paths):
    """Неудачные и успешные входы из файлов по порядку (сначала старые, .gz тоже)."""
    ev = []
    for p in paths:
        ref = datetime.fromtimestamp(os.path.getmtime(p))
        with (gzip.open if p.endswith(".gz") else open)(p, "rt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = FAIL.match(ln)
                if m: ev += [(stamp(m[1], m[2], ref), m[6], m[5], bool(m[4]), 0)] * int(m[3] or 1); continue
                m = OK.match(ln)
                if m: ev.append((stamp(m[1], m[2], ref), m[5], m[4], False, 1))
    d = pd.DataFrame(ev, columns=["ts", "ip", "user", "invalid", "ok"]); d["ts"] = pd.to_datetime(d.ts)
    return d.sort_values("ts", kind="stable").reset_index(drop=True)


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
    return pd.DataFrame(rows, columns=["t"] + FEATS + ["accepted_count"]).set_index("t")


def prep(w, stats):
    mean, std = stats
    z = (w[FEATS].values - mean) / (std + EPS)
    return (np.sign(z) * np.log1p(np.abs(z))).astype(np.float32)


def files(folder="logs"):
    """auth.log.4.gz … auth.log.1, auth.log — от старых к новым."""
    age = lambda p: int(m[1]) if (m := re.search(r"\.(\d+)(?:\.gz)?$", p)) else 0
    return sorted(glob.glob(os.path.join(folder, "auth.log*")), key=age, reverse=True)


if __name__ == "__main__":
    import torch, torch.nn as nn
    torch.manual_seed(0); rng = np.random.default_rng(0)
    args = sys.argv[1:]
    since = pd.Timestamp(args[args.index("--from") + 1]) if "--from" in args else None
    d = parse(*files()); w = windows(d)
    old, norm = (w[w.index < since], w[w.index >= since]) if since is not None else (w.iloc[:0], w)
    a, b = int(len(norm) * 0.6), int(len(norm) * 0.8)
    tr, va, te = norm.iloc[:a], norm.iloc[a:b], norm.iloc[b:]   # по времени: обучение / порог / проверка
    stats = (tr[FEATS].mean().values, tr[FEATS].std().values)
    span = lambda x: f"{len(x)} окон, {x.index[0]:%d.%m %H:%M} – {x.index[-1]:%d.%m %H:%M}" if len(x) else "нет"
    print(f"прежний режим: {span(old)}\nобучение: {span(tr)}\nпорог: {span(va)}\nпроверка: {span(te)}")
    print(pd.DataFrame({"прежний режим": old[FEATS].mean(), "норма": norm[FEATS].mean()}).round(2).to_string())

    dim = len(FEATS)
    ae = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, 3), nn.ReLU(), nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, dim))
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    X = torch.tensor(prep(tr, stats))
    for ep in range(400):
        for bt in torch.randperm(len(X)).split(64):
            opt.zero_grad(); nn.functional.mse_loss(ae(X[bt]), X[bt]).backward(); opt.step()
    def score(w_):
        with torch.no_grad(): x = torch.tensor(prep(w_, stats)); return ((ae(x) - x) ** 2).mean(1).numpy()
    # порог — перцентиль Q на днях, которых сеть не видела; счётчик неудач с таким же правилом — и для сравнения,
    # и как вторая часть детектора: всплеск «как обычно, только больше» сеть восстанавливает и пропускает
    thr, cnt = float(np.percentile(score(va), Q)), float(np.percentile(va.fail_count, Q))

    def flags(w_):   # доля помеченных окон: автоэнкодер, счётчик, вместе
        a, c = score(w_) > thr, w_.fail_count.values > cnt
        return a.mean(), c.mean(), (a | c).mean()

    # искусственные атаки: события вписываются в случайные окна проверки, признаки пересчитываются заново
    def inject(kind, n=200):
        picks = te.index[rng.choice(len(te), n, replace=False)]
        add, drop = [], []
        for k, t in enumerate(picks):
            at = lambda: t + pd.Timedelta(seconds=float(rng.uniform(0, 300)))
            if kind == "brute_1ip":                # 20 паролей к root с нового адреса, пока fail2ban не забанил
                add += [(at(), f"198.51.{k}.1", "root", False, 0) for _ in range(20)]
            if kind == "botnet":                   # 50 новых адресов по 2 попытки (ниже порога fail2ban), словарь логинов
                add += [(at(), f"203.{k}.{i}.1", u, True, 0) for i in range(50) for u in rng.choice(["admin", "test", "user", "oracle", "ubuntu", "git", "postgres", "ftp"], 2)]
            if kind == "targeted":                 # объём как обычно, но всё — один новый IP по root
                drop.append(t); add += [(at(), f"192.0.{k}.1", "root", False, 0) for _ in range(max(int(te.fail_count[t]), 8))]
            if kind == "slow":                     # одна попытка с нового IP — неотличима от фона
                add.append((at(), f"100.64.{k}.1", "root", False, 0))
        keep = d[~(d.ts.dt.floor("5min").isin(drop) & (d.ok == 0))]
        w2 = windows(pd.concat([keep, pd.DataFrame(add, columns=d.columns)]).sort_values("ts", kind="stable"))
        return w2.loc[picks]

    names = dict(false_alarms="ложные тревоги (дни проверки)", old_regime="прежний режим (до --from)",
                 brute_1ip="перебор с одного IP", botnet="ботнет", targeted="целевой подбор", slow="медленный перебор")
    rows = {"false_alarms": flags(te)}
    if len(old): rows["old_regime"] = flags(old)
    for k in ["brute_1ip", "botnet", "targeted", "slow"]:
        rows[k] = flags(inject(k))
    res = pd.DataFrame(rows, index=["автоэнкодер", "счётчик неудач", "вместе"]).T.rename(index=names)
    print(f"порог автоэнкодера {thr:.3f}, порог счётчика: больше {cnt:.0f} неудач за 5 минут")
    print(res.round(3).to_string())
    print("топ-5 аномальных окон проверки:")
    print(te.assign(score=score(te)).sort_values("score", ascending=False).head(5)[FEATS + ["score"]].round(2).to_string())

    torch.onnx.export(ae, torch.zeros(1, dim), "model_ssh.onnx", input_names=["x"], output_names=["recon"],
                      dynamic_axes={"x": {0: "n"}, "recon": {0: "n"}}, dynamo=False)
    json.dump({"threshold": thr, "features": FEATS, "mean": stats[0].tolist(), "std": stats[1].tolist(), "eps": EPS,
               "normal_from": str(norm.index[0]), "count_threshold": cnt,
               "metrics": {k: dict(zip(["autoencoder", "fail_count", "combined"], np.round(v, 4).tolist())) for k, v in rows.items()}},
              open("model_ssh.json", "w"), indent=1)
