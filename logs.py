"""Парсер nginx access.log (combined) и auth.log -> окна признаков в схеме data.FEATURES."""
import re
from datetime import datetime, timedelta
from collections import Counter
import numpy as np, pandas as pd
from data import FEATURES

NGINX = re.compile(r'(\S+) \S+ \S+ \[([^\]]+)\] "(?:\S+) (\S+)[^"]*" (\d{3}) \S+ "[^"]*" "([^"]*)"')
# Начало строки sshd: метка времени ISO 8601 (rsyslog в Debian 12+, Ubuntu 24.04+) или классическая «Oct  8 10:02:11»;
# «message repeated N times: [ …]» — rsyslog свернул N одинаковых строк подряд в одну.
SSHD = (r'^(?:(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S*|(\w{3} [ \d]\d \d\d:\d\d:\d\d)) \S+ sshd(?:-session)?\[\d+\]: '
        r'(?:message repeated (\d+) times: \[ )?')
AUTH = re.compile(SSHD + r'Failed \S+')   # как в ssh.py: «Invalid user» не считаем, за ним идёт та же неудача


def stamp(iso, classic, ref):
    """Время строки syslog. У классической метки нет года: берём год файла (ref — время его изменения),
    а строки «из будущего» (декабрь в январском файле) относим к прошлому году."""
    if iso:
        return datetime.fromisoformat(iso)
    t = datetime.strptime(f"{ref.year} {classic}", "%Y %b %d %H:%M:%S")
    return t.replace(year=ref.year - 1) if t > ref + timedelta(days=1) else t


def entropy(items):
    c = np.array(list(Counter(items).values()), float)
    p = c / c.sum()
    return abs(float((p * np.log2(p)).sum()))   # abs: у одного значения вышло бы -0.0


def read_nginx(lines):
    rows = []
    for ln in lines:
        m = NGINX.match(ln)
        if m:
            ip, ts, path, st, ua = m.groups()
            rows.append((datetime.strptime(ts.split()[0], "%d/%b/%Y:%H:%M:%S"), ip, path.split("?")[0], int(st), ua))
    return pd.DataFrame(rows, columns=["ts", "ip", "path", "status", "ua"])


def read_auth(lines, ref):
    ts = []
    for ln in lines:
        m = AUTH.match(ln)
        if m:
            ts += [stamp(m[1], m[2], ref)] * int(m[3] or 1)
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
                        new_ip_ratio=len(ips - seen) / len(ips), failed_login_count=float(fl)))
        seen |= ips
    return pd.DataFrame(out)[["timestamp"] + FEATURES]
