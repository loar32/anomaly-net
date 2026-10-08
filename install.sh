#!/usr/bin/env bash
# Установка детектора на Linux-сервер: ./install.sh [каталог, по умолчанию /opt/anomaly-net]
# Нужны: python3 с модулем venv, root (cron и чтение /var/log/auth.log).
# Необязательно: TG_TOKEN и TG_CHAT (тревоги в Telegram), ANOMALY_LANG=ru|en (язык сообщений, по умолчанию ru).
set -euo pipefail
DEST="${1:-/opt/anomaly-net}"
SRC="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$DEST"
cp "$SRC"/{data.py,logs.py,ssh.py,detect.py,model_ssh.onnx,model_ssh.json,requirements.txt} "$DEST"/
python3 -m venv "$DEST/.venv"
"$DEST/.venv/bin/pip" install -q -r "$DEST/requirements.txt"

LANG_UI="${ANOMALY_LANG:-ru}"
if [ -n "${TG_TOKEN:-}" ] && [ -n "${TG_CHAT:-}" ]; then
  printf 'TG_TOKEN=%s\nTG_CHAT=%s\nANOMALY_LANG=%s\n' "$TG_TOKEN" "$TG_CHAT" "$LANG_UI" > "$DEST/.env"; chmod 600 "$DEST/.env"
elif [ ! -f "$DEST/.env" ]; then
  printf 'ANOMALY_LANG=%s\n' "$LANG_UI" > "$DEST/.env"; chmod 600 "$DEST/.env"
fi

CRON="*/15 * * * * cd $DEST && set -a && . ./.env && set +a && .venv/bin/python detect.py /var/log/auth.log --recent 20 >> /var/log/anomaly-net.log 2>&1"
( crontab -l 2>/dev/null | grep -vF "$DEST && set -a" || true; echo "$CRON" ) | crontab -

case "$LANG_UI" in
  ru*) M1="Проверка:"; M2="Готово. Находки пишутся в /var/log/anomaly-net.log (проверка раз в 15 минут).";;
  *)   M1="Trial run:"; M2="Done. Findings are written to /var/log/anomaly-net.log (checked every 15 minutes).";;
esac
echo "$M1"
ANOMALY_LANG="$LANG_UI" "$DEST/.venv/bin/python" "$DEST/detect.py" /var/log/auth.log | head -5
echo "$M2"
