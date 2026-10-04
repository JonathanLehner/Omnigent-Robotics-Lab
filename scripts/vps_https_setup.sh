#!/bin/bash
# Opal Labs: Omnigent server behind HTTPS on the VPS. Run as root on the server:
#   ssh root@217.216.75.252 'bash -s' < scripts/vps_https_setup.sh
# - Omnigent server as a systemd service (user "lab"), bound to 127.0.0.1:6767, login enabled
# - Caddy reverse proxy with an automatic Let's Encrypt certificate for HOSTN
# - firewall: only SSH, HTTP and HTTPS
set -euo pipefail
HOSTN="${HOSTN:-217-216-75-252.sslip.io}"   # free hostname that resolves to the server's IP
PW="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"

export DEBIAN_FRONTEND=noninteractive
apt-get install -y -qq caddy >/dev/null

cat > /home/lab/omnigent.env <<EOF
OMNIGENT_AUTH_ENABLED=1
OMNIGENT_ACCOUNTS_BASE_URL=https://$HOSTN
OMNIGENT_ACCOUNTS_INIT_ADMIN_PASSWORD=$PW
LAB_ROOT=/home/lab/Omnigent-Robotics-Lab
MUJOCO_GL=osmesa
PYOPENGL_PLATFORM=osmesa
PATH=/home/lab/.npm-global/bin:/home/lab/.local/bin:/home/lab/.pixi/bin:/usr/local/bin:/usr/bin:/bin
EOF
chown lab:lab /home/lab/omnigent.env && chmod 600 /home/lab/omnigent.env

cat > /etc/systemd/system/omnigent.service <<'EOF'
[Unit]
Description=Omnigent server (Opal Labs)
After=network-online.target
[Service]
User=lab
WorkingDirectory=/home/lab/Omnigent-Robotics-Lab
EnvironmentFile=/home/lab/omnigent.env
ExecStart=/home/lab/.local/bin/omnigent server --host 127.0.0.1 --no-open
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF

cat > /etc/caddy/Caddyfile <<EOF
$HOSTN {
    reverse_proxy 127.0.0.1:6767
}
EOF

systemctl daemon-reload
systemctl enable --now omnigent
systemctl restart caddy
ufw allow 22/tcp && ufw allow 80/tcp && ufw allow 443/tcp && ufw --force enable

sleep 8
systemctl is-active omnigent caddy
echo "Open https://$HOSTN  (user: admin, password: $PW)"
echo "Store this password; it is also in /home/lab/omnigent.env (readable only by lab and root)."
