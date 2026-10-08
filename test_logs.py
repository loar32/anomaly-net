from logs import *
acc = ['1.1.1.1 - - [08/Oct/2026:10:00:01 +0000] "GET /a?x=1 HTTP/1.1" 200 5 "-" "curl/8"',
       '2.2.2.2 - - [08/Oct/2026:10:01:00 +0000] "GET /.env HTTP/1.1" 404 5 "-" "Mozilla/5.0"',
       '1.1.1.1 - - [08/Oct/2026:10:06:00 +0000] "POST /login HTTP/1.1" 502 5 "-" "curl/8"',
       'garbage line']
auth = ['Oct  8 10:02:11 h sshd[1]: Failed password for root from 3.3.3.3 port 22 ssh2',
        'Oct  8 10:03:11 h sshd[1]: Invalid user admin from 3.3.3.3 port 22',
        'Oct  8 10:03:12 h sshd[1]: Accepted password for root']
w = windows(read_nginx(acc), read_auth(auth, 2026))
print(w.T)
a, b = w.iloc[0], w.iloc[1]
assert (a.req_count, a.uniq_ips, a.error_rate_4xx, a.failed_login_count, a.new_ip_ratio) == (2, 2, 0.5, 2, 1.0)
assert (b.req_count, b.error_rate_5xx, b.new_ip_ratio, b.failed_login_count) == (1, 1.0, 0.0, 0)
print("ok")
