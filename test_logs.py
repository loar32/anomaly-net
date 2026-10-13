import os, tempfile
from datetime import datetime
from logs import *
from ssh import parse, windows as ssh_windows

acc = ['1.1.1.1 - - [08/Oct/2026:10:00:01 +0000] "GET /a?x=1 HTTP/1.1" 200 5 "-" "curl/8"',
       '2.2.2.2 - - [08/Oct/2026:10:01:00 +0000] "GET /.env HTTP/1.1" 404 5 "-" "Mozilla/5.0"',
       '1.1.1.1 - - [08/Oct/2026:10:06:00 +0000] "POST /login HTTP/1.1" 502 5 "-" "curl/8"',
       'garbage line']
auth = ['Oct  8 10:02:11 h sshd[1]: Failed password for root from 3.3.3.3 port 22 ssh2',
        'Oct  8 10:03:11 h sshd[1]: Invalid user admin from 3.3.3.3 port 22',
        'Oct  8 10:03:12 h sshd[1]: Failed password for invalid user admin from 3.3.3.3 port 22 ssh2',
        'Oct  8 10:03:12 h sshd[1]: Accepted password for root']
w = windows(read_nginx(acc), read_auth(auth, datetime(2026, 10, 9)))
print(w.T)
a, b = w.iloc[0], w.iloc[1]
assert (a.req_count, a.uniq_ips, a.error_rate_4xx, a.failed_login_count, a.new_ip_ratio) == (2, 2, 0.5, 2, 1.0)
assert (b.req_count, b.error_rate_5xx, b.new_ip_ratio, b.failed_login_count) == (1, 1.0, 0.0, 0)

# auth.log для SSH-модели: классические метки через Новый год, свёрнутые повторы, пустой логин, ISO-метки
log = ['Dec 31 23:59:58 h sshd[2]: Failed password for root from 4.4.4.4 port 22 ssh2',
       'Jan  1 00:00:01 h sshd[2]: message repeated 3 times: [ Failed password for root from 4.4.4.4 port 22 ssh2]',
       '2027-01-01T00:01:00.5+00:00 h sshd-session[3]: Failed none for invalid user  from 5.5.5.5 port 22 ssh2',
       '2027-01-01T00:02:00.5+00:00 h sshd-session[4]: Accepted publickey for root from 6.6.6.6 port 22 ssh2: ED25519']
p = os.path.join(tempfile.mkdtemp(), "auth.log")
open(p, "w").write("\n".join(log) + "\n"); os.utime(p, (datetime(2027, 1, 2).timestamp(),) * 2)
d = parse(p)
assert d.ts.dt.year.tolist() == [2026, 2027, 2027, 2027, 2027, 2027] and d.ok.sum() == 1
assert d.invalid.sum() == 1 and d.user[4] == ""
s = ssh_windows(d)
assert s.fail_count.sum() == 5 and s.accepted_count.sum() == 1
assert len(ssh_windows(d.iloc[:0])) == 0   # пустой лог сразу после ротации не роняет детектор
print("ok")
