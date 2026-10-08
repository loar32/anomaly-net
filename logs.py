"""Парсер nginx access.log (combined) и auth.log -> окна признаков в схеме data.FEATURES."""
import re
from datetime import datetime
from collections import Counter
import numpy as np, pandas as pd
from data import FEATURES

NGINX = re.compile(r'(\S+) \S+ \S+ \[([^\]]+)\] "(?:\S+) (\S+)[^"]*" (\d{3}) \S+ "[^"]*" "([^"]*)"')
AUTH = re.compile(r'^(?:(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S*|(\w{3}\s+\d+ \d\d:\d\d:\d\d)) \S+ sshd(?:-session)?\[\d+\]: (?:Failed \S+|Invalid user)')


def entropy(items):
    c = np.array(list(Counter(items).values()), float)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def read_nginx(lines):
    rows = []
    for ln in lines:
        m = NGINX.match(ln)
        if m:
            ip, ts, path, st, ua = m.groups()
            rows.append((datetime.strptime(ts.split()[0], "%d/%b/%Y:%H:%M:%S"), ip, path.split("?")[0], int(st), ua))
    return pd.DataFrame(rows, columns=["ts", "ip", "path", "status", "ua"])


def read_auth(lines, year):
    ts = []
    for ln in lines:
        m = AUTH.match(ln)
        if m:
            ts.append(datetime.fromisoformat(m.group(1)) if m.group(1)
                      else datetime.strptime(f"{year} {m.group(2)}", "%Y %b %d %H:%M:%S"))
    return pd.Series(ts, dtype="datetime64[ns]")


def windows(access, failed_ts=None, freq="5min"):
    """Одно окно = freq. new_ip_ratio: доля IP, которых не было ни в одном прошлом окне."""
    seen, out = set(), []
    for t, g in access.groupby(pd.Grouper(key="ts", freq=freq)):
        if g.empty:
            continue
        ips = set(g.ip)
        fl = 0 if failed_ts is None else int(((failed_ts >= t) & (failed_ts < t + pd.Timedelta(freq))).sum())
        out.append(dict(timestamp=t, req_count=len(g), uniq_ips=len(ips),
                        error_rate_4xx=g.status.between(400, 499).mean(), error_rate_5xx=(g.status >= 500).mean(),
                        path_entropy=entropy(g.path), ua_entropy=entropy(g.ua),
                        new_ip_ratio=len(ips - seen) / len(ips), failed_login_count=float(fl),
                        distinct_ports_per_ip=1.0))  # в логах веб-сервера портов нет
        seen |= ips
    return pd.DataFrame(out)[["timestamp"] + FEATURES]
