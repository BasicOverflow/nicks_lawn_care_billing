#!/usr/bin/env bash
# Bootstrap Nick's billing inside the LXC. Invoked by install.sh via pct exec.
set -euo pipefail

APP_ROOT="${APP_ROOT:-/opt/nicks_lawn_care_billing}"
APP_USER="${APP_USER:-billing}"
ENV_FILE="${ENV_FILE:-/etc/nicks-billing.env}"
RAY_HIVE_ROOT="${RAY_HIVE_ROOT:-/opt/ray-hive}"
RAY_HIVE_CLONE_URL="${RAY_HIVE_CLONE_URL:-https://github.com/BasicOverflow/ray-hive.git}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

export DEBIAN_FRONTEND=noninteractive
export LANG=C.UTF-8

apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates curl git \
  build-essential \
  libglib2.0-0 libgomp1 libgl1 \
  libpq5

if ! id -u "$APP_USER" >/dev/null 2>&1; then
  useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"
fi

mkdir -p "$APP_ROOT" "$(dirname "$ENV_FILE")" "$(dirname "$RAY_HIVE_ROOT")"
if [[ ! -f "$APP_ROOT/run.py" ]]; then
  echo "bootstrap: missing $APP_ROOT/run.py" >&2
  exit 1
fi
if [[ ! -f "$ENV_FILE" ]]; then
  echo "bootstrap: missing $ENV_FILE" >&2
  exit 1
fi
ln -sfn "$ENV_FILE" "$APP_ROOT/.env"

if [[ ! -d "$RAY_HIVE_ROOT/.git" && ! -f "$RAY_HIVE_ROOT/pyproject.toml" ]]; then
  echo "bootstrap: cloning ray-hive (job working_dir, not the GPU runtime)"
  rm -rf "$RAY_HIVE_ROOT"
  git clone --depth 1 "$RAY_HIVE_CLONE_URL" "$RAY_HIVE_ROOT"
fi

export PATH="/root/.local/bin:/usr/local/bin:${PATH}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/opt/uv-python}"
mkdir -p "$UV_PYTHON_INSTALL_DIR"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
if [[ -x /root/.local/bin/uv ]]; then
  ln -sfn /root/.local/bin/uv /usr/local/bin/uv
fi
if [[ -x /root/.cargo/bin/uv ]]; then
  ln -sfn /root/.cargo/bin/uv /usr/local/bin/uv
fi
hash -r || true
command -v uv >/dev/null 2>&1 || { echo "bootstrap: uv not on PATH" >&2; exit 1; }

uv python install "$PYTHON_VERSION"
chmod -R a+rX "$UV_PYTHON_INSTALL_DIR"
rm -rf "$APP_ROOT/.venv"
uv venv --python "$PYTHON_VERSION" "$APP_ROOT/.venv"
uv pip install --python "$APP_ROOT/.venv/bin/python" -r "$APP_ROOT/requirements.txt"
# Client imports only. Workers already have vLLM; do not pull torch into this CT.
uv pip install --python "$APP_ROOT/.venv/bin/python" --no-deps -e "$RAY_HIVE_ROOT"
uv pip install --python "$APP_ROOT/.venv/bin/python" pydantic requests rich

install -m 0644 "$APP_ROOT/deploy/nicks-billing.service" /etc/systemd/system/nicks-billing.service
sed -i "s#^WorkingDirectory=.*#WorkingDirectory=${APP_ROOT}#" /etc/systemd/system/nicks-billing.service
sed -i "s#^EnvironmentFile=.*#EnvironmentFile=${ENV_FILE}#" /etc/systemd/system/nicks-billing.service
sed -i "s#^Environment=PYTHONPATH=.*#Environment=PYTHONPATH=${APP_ROOT}:${RAY_HIVE_ROOT}#" /etc/systemd/system/nicks-billing.service
sed -i "s#^Environment=RAY_HIVE=.*#Environment=RAY_HIVE=${RAY_HIVE_ROOT}#" /etc/systemd/system/nicks-billing.service
sed -i "s#^ExecStart=.*#ExecStart=${APP_ROOT}/.venv/bin/python -u run.py#" /etc/systemd/system/nicks-billing.service
sed -i "s#^User=.*#User=${APP_USER}#" /etc/systemd/system/nicks-billing.service
sed -i "s#^Group=.*#Group=${APP_USER}#" /etc/systemd/system/nicks-billing.service

chown -R "$APP_USER:$APP_USER" "$APP_ROOT" "$RAY_HIVE_ROOT"
chmod 640 "$ENV_FILE"
chown root:"$APP_USER" "$ENV_FILE"

systemctl daemon-reload
systemctl enable nicks-billing.service
systemctl restart nicks-billing.service

echo "bootstrap: nicks-billing.service started"
systemctl --no-pager --full status nicks-billing.service || true
