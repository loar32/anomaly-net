# anomaly-net

[Русский](README.md) | **English**

Log anomaly detector: an autoencoder (PyTorch → ONNX) over features expressed as z-scores against the server's own history. Two models: **SSH** (`auth.log`, ready to use; the autoencoder works together with a simple failed-login counter) and **web traffic** (`model_v1`, synthetic data only so far).

## Install (SSH detector on a Linux server)

Requirements: Linux, root, `python3` with the `venv` module, and `/var/log/auth.log` (Debian/Ubuntu; if logs go only to journald, install rsyslog).

```bash
git clone https://github.com/loar32/anomaly-net.git
cd anomaly-net
sudo ./install.sh                                 # without Telegram
# or with Telegram alerts and a language for messages (ru or en, default ru):
sudo TG_TOKEN=<bot token> TG_CHAT=<chat id> ANOMALY_LANG=ru ./install.sh
```

`install.sh` copies the files to `/opt/anomaly-net`, creates a venv, installs `numpy pandas onnxruntime`, adds a cron job that runs every 15 minutes, and finishes with a trial run over the current log. It is safe to re-run: the cron line is not duplicated and an existing `.env` (Telegram settings) is kept. Findings are appended to `/var/log/anomaly-net.log`.

The detector reads `auth.log.1` and `auth.log`, so after the weekly rotation known IPs do not become "new". In cron mode it only looks at finished 5-minute windows and remembers what it has already reported (`state.json` in the install directory): the same window or login is never sent to Telegram twice. If Telegram is unreachable, the findings go out on the next run.

Manual check: `cd /opt/anomaly-net && .venv/bin/python detect.py /var/log/auth.log` should print a line like `windows N, anomalous M`.

**Important:** `model_ssh.onnx` was calibrated on one server with fail2ban (ban after 4 failures in 10 minutes) and a background of about 8 failed logins per 5 minutes. On a server with a different background, retrain on your own logs (at least 4 days of normal traffic): put `auth.log`, `auth.log.1` and any `auth.log.*.gz` into `logs/`, run `pip install -r requirements-train.txt`, then `python ssh.py --from <start of normal>` (e.g. `--from 2026-10-08T18:00`), and copy the new `model_ssh.onnx` and `model_ssh.json` to the install directory. Without `--from` the whole log is treated as normal.

Parser tests: `python test_logs.py` (needs only numpy and pandas).

## SSH model (`ssh.py`, `detect.py`)

Data: `auth.log` of one server from 13 Sep to 13 Oct 2026. Until 8 October the server was under constant brute force (about 32 failed logins per 5 minutes). fail2ban was enabled on 8 October at 17:36, and the background dropped to 8. **Normal is the period after fail2ban** (from 8 Oct 18:00); everything earlier is the "old regime", which the model does not treat as normal.

- 5-minute windows, 6 features: failed attempts, unique IPs, share of non-existent users, username entropy, share of new IPs, top-IP share.
- Normalisation: `z = (x − mean) / (std + 0.1)`, then `sign(z)·log(1+|z|)`; mean and std come from the training period.
- Autoencoder 6→32→3→32→6 (ReLU).
- The normal period is split by time: training 8 Oct 18:00 – 11 Oct 14:15 (820 windows), threshold 11 Oct 14:20 – 12 Oct 13:00 (273 windows), test 12 Oct 13:05 – 13 Oct 11:50 (274 windows). The threshold and the test use days the network has not seen.
- The autoencoder threshold is the 99.5th percentile of the reconstruction error on the threshold day. The counter uses the same rule for the number of failures (more than 19 per 5 minutes). **A window is anomalous if either the autoencoder or the counter fires.**
- Successful logins are not part of the model: a separate "login from a new IP" rule reports the first login from an address that is absent from the history (`auth.log.1`, `auth.log`) and from `state.json`.

`python ssh.py --from 2026-10-08T18:00` trains the model, prints the table below and writes `model_ssh.onnx` and `model_ssh.json` (threshold, counter threshold, mean/std, metrics). `python detect.py /var/log/auth.log` runs a log through the model; `--recent 20` is the cron mode (the first run looks at the last 20 minutes, later runs continue where the previous one stopped).

### Results (share of flagged windows)

| | autoencoder | failure counter | combined |
|---|---|---|---|
| false alarms (test day) | 0.004 | 0.000 | 0.004 |
| old brute-force regime (13 Sep – 8 Oct) | 0.44 | 0.48 | 0.51 |
| brute force from one IP (20 passwords for root) | 1.00 | 1.00 | 1.00 |
| botnet (50 new IPs, 2 attempts each, dictionary usernames) | 0.00 | 1.00 | 1.00 |
| targeted guessing (usual volume, all from one new IP for root) | 0.42 | 0.00 | 0.42 |
| slow brute force (1 attempt from a new IP) | 0.025 | 0.005 | 0.030 |

The last four rows are injected attacks: events are added to 200 random windows of the test day and the features are recomputed from scratch.

- The autoencoder and the counter complement each other. To the network a botnet looks like "the usual picture, only bigger": in normal traffic attempts and IPs grow together, so it reconstructs such a window well. The counter catches it. Targeted guessing does not change the volume, so the counter misses it; the network catches it by the unusual structure of the window (one IP, one username).
- About half of the old regime is flagged: during quiet hours the brute force looks like the new normal.
- Over the whole period after fail2ban the detector flags 0–5 windows a day, mostly through the counter; these are real bursts of more than 19 failures in 5 minutes.

### Limitations

- **The threshold is a percentile**, so about 0.5% of normal windows are flagged by construction, even with no attacks. On the test day it was 0.4%. This is an alarm rate, not an attack rate.
- **There are no labeled real attacks.** The "old regime" is a different traffic regime, not a label; the attacks in the table are injected. This does not measure real recall.
- **One server and about 4.5 days of normal traffic.** If an attack happened during the normal period, the model learned it as normal; this is not checked.
- **The background after fail2ban is still bot attempts** (1–3 attempts per address, below the ban threshold). The model treats them as normal; only key-only login or moving SSH to another port would remove them.
- **Slow brute force** (one attempt each from different IPs) is indistinguishable from the background.
- **No drift handling**: the threshold is fixed. After changing SSH/fail2ban settings or a noticeable change in traffic, retrain the model.
- Only sshd `Failed …` and `Accepted …` lines are parsed (ISO 8601 and classic syslog timestamps, repeats collapsed by rsyslog); lines in other formats are skipped.

## Web model (`train.py`)

Trained on synthetic data: `pip install -r requirements-train.txt`, then `python train.py` (writes `model_v1.onnx` and `model_v1.json`). It learns the normal state only and is evaluated on servers it has not seen.

8 features: requests, unique IPs, 4xx and 5xx shares, path and user-agent entropy, share of new IPs, failed SSH logins. There is no "ports per IP" feature and no port-scan scenario: web server logs have no ports, so on real logs that feature would always be a constant. Autoencoder 8→64→4→64→8: with a bottleneck of 8 the network can simply copy its input.

### Metrics (held-out servers)

The threshold is the 99th percentile of the error on held-out normal windows of the training servers; FPR and recall are measured on servers the model has not seen (so FPR is not 1% by construction).

| | autoencoder | isolation forest |
|---|---|---|
| FPR | 0.012 | 0.011 |
| brute_force | 1.00 | 1.00 |
| traffic_spike | 1.00 | 0.18 |
| path_scanning | 1.00 | 0.89 |
| credential_stuffing | 1.00 | 0.82 |
| slow_exfiltration | 0.89 | 0.08 |
| recall (all) | 0.98 | 0.60 |

Tested on synthetic scenarios only. In the synthetic normal traffic, features share a common load factor (requests, IPs, errors and entropy rise together); without such structure the autoencoder has nothing to compress. Real logs will look different; this is still to be verified.

## License

MIT
