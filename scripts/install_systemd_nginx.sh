#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SYSTEMD_SRC_DIR="$ROOT_DIR/deploy/systemd"
NGINX_SRC_CONF="$ROOT_DIR/deploy/nginx/neurx-model.conf"
NGINX_SRC_LOCATIONS="$ROOT_DIR/deploy/nginx/neurx-model.locations.conf"
NGINX_SRC_FORWARD_LOCATIONS="$ROOT_DIR/deploy/nginx/neurx-model-forwarded-8080.locations.conf"

echo "[1/6] Install systemd unit files"
install -m 0644 "$SYSTEMD_SRC_DIR/neurx-model-backend.service" /etc/systemd/system/neurx-model-backend.service
install -m 0644 "$SYSTEMD_SRC_DIR/neurx-model-frontend.service" /etc/systemd/system/neurx-model-frontend.service

echo "[2/6] Reload systemd daemon"
systemctl daemon-reload

echo "[3/6] Enable and restart backend/frontend services"
if command -v fuser >/dev/null 2>&1; then
	fuser -k 3100/tcp 2>/dev/null || true
	fuser -k 8000/tcp 2>/dev/null || true
	fuser -k 8080/tcp 2>/dev/null || true
fi
systemctl enable neurx-model-backend.service neurx-model-frontend.service
systemctl restart neurx-model-backend.service neurx-model-frontend.service

echo "[4/6] Install nginx reverse proxy config"
install -d /etc/nginx/sites-available /etc/nginx/sites-enabled /etc/nginx/snippets

if [[ -f /etc/nginx/sites-available/tiku-admin.conf ]]; then
	install -m 0644 "$NGINX_SRC_LOCATIONS" /etc/nginx/snippets/neurx-model.locations.conf
	if ! grep -q "include /etc/nginx/snippets/neurx-model.locations.conf;" /etc/nginx/sites-available/tiku-admin.conf; then
		sed -i '/client_max_body_size 50m;/a\
\
		include /etc/nginx/snippets/neurx-model.locations.conf;' /etc/nginx/sites-available/tiku-admin.conf
	fi
	rm -f /etc/nginx/sites-enabled/neurx-model.conf /etc/nginx/sites-available/neurx-model.conf /etc/nginx/conf.d/neurx-model.conf
else
	install -m 0644 "$NGINX_SRC_CONF" /etc/nginx/sites-available/neurx-model.conf
	ln -sfn /etc/nginx/sites-available/neurx-model.conf /etc/nginx/sites-enabled/neurx-model.conf
	rm -f /etc/nginx/conf.d/neurx-model.conf
fi

# Optional compatibility: some upstream environments forward external :8080 to local :80.
# If aistudy.conf exists, inject forwarding rules for /neurx and API endpoints there.
if [[ -f /etc/nginx/sites-available/aistudy.conf ]]; then
	install -m 0644 "$NGINX_SRC_FORWARD_LOCATIONS" /etc/nginx/snippets/neurx-model-forwarded-8080.locations.conf
	if ! grep -q "include /etc/nginx/snippets/neurx-model-forwarded-8080.locations.conf;" /etc/nginx/sites-available/aistudy.conf; then
		sed -i '/server_name 111.202.231.146;/a\
\
		include /etc/nginx/snippets/neurx-model-forwarded-8080.locations.conf;' /etc/nginx/sites-available/aistudy.conf
	fi
fi

echo "[5/6] Test and reload nginx"
nginx -t
systemctl enable nginx
systemctl restart nginx

echo "[6/6] Verify endpoints"
echo "Backend health:"
curl -sS http://127.0.0.1:8000/health || true
echo
echo "Nginx /neurx:"
curl -sS -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8080/neurx || true

echo "Done. External URL: http://111.202.231.146:8080/neurx"