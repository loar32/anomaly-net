# anomaly-net

[Русский](README.md) | **English**

Log anomaly detector: an autoencoder (PyTorch → ONNX) over features expressed as z-scores against the server's own history. Two models: **SSH** (`auth.log`, ready to use) and **web traffic** (`model_v1`, synthetic data only so far).

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

Manual check: `cd /opt/anomaly-net && .venv/bin/python detect.py /var/log/auth.log` should print a line like `windows N, anomalous M`.

**Important:** `model_ssh.onnx` was calibrated on one server with constant background SSH noise (~35 failed logins per 5 minutes). On a quiet server it will raise many false alarms. Retrain on your own log (at least 4 days): put it in `logs/auth.log`, run `pip install -r requirements-train.txt`, then `python ssh.py`, and copy the new `model_ssh.onnx` and `model_ssh.json` to the install directory.

Parser tests: `python test_logs.py` (needs only numpy and pandas).

## SSH model (`ssh.py`, `detect.py`)

Trained on a real `auth.log` from a server under constant SSH brute force (~15k failed logins per day); the normal state is that background noise. 5-minute windows, 6 features (attempts, unique IPs, share of non-existent users, username entropy, new IPs, top-IP share), z-scores from the first 3 days, autoencoder 6→3→6. Successful logins are not part of the model: they are a separate rule in `detect.py` ("login from a new IP").

`python ssh.py` trains and writes `model_ssh.onnx` and `model_ssh.json` (threshold, mean/std). `python detect.py /var/log/auth.log` runs a log through the ONNX model; with `--recent 20` it reports only the last 20 minutes (this is what cron uses).

Result: over the whole log (1345 windows) 17 were flagged (1.3%). Synthetic bursts injected into real windows (brute force from 1 IP, a 200-IP botnet) are caught 100% of the time. Some alerts are quiet windows with a single attempt from a new IP — false alarms. There are no labeled real attacks in the log, so this does not measure real recall; one server only, transfer to others is untested.

## Web model (`train.py`)

Trained on synthetic data: `pip install -r requirements-train.txt`, then `python train.py` (writes `model_v1.onnx` and `model_v1.json`). It learns the normal state only and is evaluated on servers it has not seen.

### Metrics v1 (held-out servers, threshold = 99th percentile of normal, bottleneck 8)

| | autoencoder | isolation forest |
|---|---|---|
| FPR | 0.010 | 0.010 |
| port_scan | 1.00 | 0.26 |
| brute_force | 1.00 | 1.00 |
| traffic_spike | 1.00 | 0.16 |
| path_scanning | 1.00 | 0.72 |
| credential_stuffing | 1.00 | 1.00 |
| slow_exfiltration | 0.80 | 0.12 |
| recall (all) | 0.97 | 0.55 |

Tested on synthetic scenarios only: port scan, brute force, traffic spike, path scanning, credential stuffing, slow exfiltration. In the synthetic normal traffic, features share a common load factor (requests, IPs, errors and entropy rise together); without such structure the autoencoder has nothing to compress — it either copies its input (bottleneck 16) or is unstable (recall 0.73–0.75). Real logs will look different; this is still to be verified.

## License

MIT
