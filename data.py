"""Синтетические окна трафика + z-score признаки относительно истории сервера."""
import numpy as np
import pandas as pd

FEATURES = ["req_count", "uniq_ips", "error_rate_4xx", "error_rate_5xx", "path_entropy",
            "ua_entropy", "new_ip_ratio", "failed_login_count", "distinct_ports_per_ip"]
ANOMALIES = ["port_scan", "brute_force", "traffic_spike", "path_scanning",
             "credential_stuffing", "slow_exfiltration"]
EPS = 0.1


def make_profiles(n, seed):
    """Выдуманные серверы: у каждого свой масштаб трафика и своя 'норма'."""
    r = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        req = float(np.exp(r.uniform(np.log(200), np.log(20000))))
        out[f"srv{seed}_{i}"] = dict(
            req_mean=req, req_std=req * r.uniform(0.1, 0.3),
            ip_mean=req * r.uniform(0.05, 0.3), ip_std=req * r.uniform(0.01, 0.04),
            path_entropy_mean=r.uniform(2.5, 4.5), ua_entropy_mean=r.uniform(1.5, 3.0),
            e4=r.uniform(0.02, 0.08), e5=r.uniform(0.003, 0.02), new_ip=r.uniform(0.05, 0.2),
            logins=r.uniform(0.2, 1.5))
    return out


def normal_window(p, r):
    load = r.normal()  # общий фактор нагрузки: запросы, IP, ошибки и энтропия растут вместе
    ip = max(1.0, p["ip_mean"] + p["ip_std"] * (0.9 * load + 0.44 * r.normal()))
    return dict(
        req_count=max(1.0, p["req_mean"] + p["req_std"] * load), uniq_ips=ip,
        error_rate_4xx=np.clip(p["e4"] + 0.02 * (0.6 * load + 0.8 * r.normal()), 0, 1),
        error_rate_5xx=np.clip(r.normal(p["e5"], 0.005), 0, 1),
        path_entropy=p["path_entropy_mean"] + 0.3 * (0.7 * load + 0.71 * r.normal()), ua_entropy=r.normal(p["ua_entropy_mean"], 0.2),
        new_ip_ratio=np.clip(p["new_ip"] + 0.05 * (0.7 * load + 0.71 * r.normal()), 0, 1),
        failed_login_count=float(r.poisson(p["logins"])), distinct_ports_per_ip=1.0)


def inject(w, kind, r):
    u = r.uniform
    if kind == "port_scan":
        w["distinct_ports_per_ip"] = u(15, 200); w["uniq_ips"] = max(1, w["uniq_ips"] * u(0.1, 0.3))
        w["new_ip_ratio"] = u(0.7, 1.0)
    elif kind == "brute_force":
        w["failed_login_count"] = u(50, 500); w["uniq_ips"] = max(1, w["uniq_ips"] * u(0.05, 0.2))
        w["error_rate_4xx"] = min(1, w["error_rate_4xx"] + u(0.3, 0.6))
    elif kind == "traffic_spike":
        m = u(5, 50); w["req_count"] *= m; w["uniq_ips"] *= u(1, m * 0.3)
    elif kind == "path_scanning":
        w["path_entropy"] = u(4.0, 7.0); w["error_rate_4xx"] = min(1, w["error_rate_4xx"] + u(0.4, 0.8))
        w["ua_entropy"] = u(0.0, 0.5)
    elif kind == "credential_stuffing":
        w["failed_login_count"] = u(30, 300); w["uniq_ips"] *= u(2, 10); w["new_ip_ratio"] = u(0.6, 0.95)
    elif kind == "slow_exfiltration":
        w["req_count"] *= u(1.3, 1.8); w["new_ip_ratio"] = min(1, w["new_ip_ratio"] + u(0.1, 0.2))
    return w


def build(profiles, n=5000, anomaly_ratio=0.15, seed=0):
    r = np.random.default_rng(seed)
    rows = []
    for sid, p in profiles.items():
        for i in range(n):
            w = normal_window(p, r)
            kind = "none"
            if i >= n * (1 - anomaly_ratio):
                kind = ANOMALIES[r.integers(len(ANOMALIES))]
                w = inject(w, kind, r)
            rows.append({**w, "server_id": sid, "label": int(kind != "none"), "anomaly_type": kind})
    return pd.DataFrame(rows)


def server_stats(normal_df):
    """Среднее/std по нормальным окнам сервера (калибровочный период)."""
    return normal_df[FEATURES].mean().values, normal_df[FEATURES].std().values


def zfeatures(df, stats):
    """z-score относительно сервера + signed-log, чтобы хвосты не взрывали сеть."""
    mean, std = stats
    z = (df[FEATURES].values - mean) / (std + EPS)
    return np.sign(z) * np.log1p(np.abs(z))
