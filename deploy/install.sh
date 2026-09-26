#!/usr/bin/env bash
# Run as root on home-media (Proxmox): create LXC, install Nick's billing, start systemd.
#
# Same shape as lexis-markets: Debian LXC + uv Python + systemd. No Docker inside the CT.
# App listens on :8787. OCR still runs on ray-hive; Postgres/MinIO stay on the NAS.
#
# After Gitea has mirrored github.com/BasicOverflow/nicks_lawn_care_billing:
#   bash -c "$(curl -fsSL http://10.0.0.52:3000/admin/nicks_lawn_care_billing/raw/branch/main/deploy/install.sh)"
#
# Before that, from a copy already on the Proxmox host:
#   SRC_DIR=/root/nini_billing_scratch bash /root/nini_billing_scratch/deploy/install.sh
#
# Secrets (never committed): /root/nicks-billing.env
#   DATABASE_URL, S3_ACCESS_KEY, S3_SECRET_KEY, SMTP_*, RAY_ADDRESS, RAY_DASHBOARD, RAY_SERVE
set -euo pipefail

REPO_CLONE_URL="${REPO_CLONE_URL:-http://10.0.0.52:3000/admin/nicks_lawn_care_billing.git}"
WORK_DIR="${WORK_DIR:-/tmp/nicks-billing-deploy}"
SRC_DIR="${SRC_DIR:-}"

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing command: $1" >&2; exit 1; }
}

need_cmd pct

if [[ "$(id -u)" -ne 0 ]]; then
  echo "run as root on a Proxmox node (home-media)" >&2
  exit 1
fi

rm -rf "$WORK_DIR"
mkdir -p "$WORK_DIR"

# One app tree. Secrets, sheet photos, and gold labels stay off the guest.
tar_app() {
  local src="$1"
  tar -C "$src" -cf - \
    --exclude .git \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'bench_results' \
    --exclude 'ground_truth' \
    --exclude '_archive_old_photos' \
    --exclude 'ocr_bench' \
    --exclude '.tmp' \
    --exclude 'data' \
    --exclude 'update_new_sheet_gold.py' \
    .
}

if [[ -n "$SRC_DIR" ]]; then
  echo "install: using local tree ${SRC_DIR}"
  if [[ ! -f "${SRC_DIR}/run.py" ]]; then
    echo "install: ${SRC_DIR}/run.py not found" >&2
    exit 1
  fi
  tar_app "$SRC_DIR" | tar -C "$WORK_DIR" -xf -
else
  need_cmd curl
  if ! command -v git >/dev/null 2>&1; then
    echo "install: installing git on Proxmox host"
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq git
  fi
  echo "install: cloning ${REPO_CLONE_URL}"
  git clone --depth 1 "$REPO_CLONE_URL" "$WORK_DIR"
fi
find "$WORK_DIR" -type f -exec sed -i 's/\r$//' {} +

__OVERRIDES=$(mktemp)
for v in CTID CT_HOSTNAME CT_IP CT_GW CT_CIDR BRIDGE STORAGE TEMPLATE_STORAGE OSTEMPLATE MEMORY CORES DISK_GB UNPRIVILEGED FEATURES START_ON_BOOT HA_GROUP APP_ROOT APP_USER ENV_FILE RAY_HIVE_ROOT RAY_HIVE_CLONE_URL REPO_CLONE_URL SECRETS_FILE SRC_DIR; do
  if [[ -n "${!v+x}" ]]; then
    printf '%s=%q\n' "$v" "${!v}" >>"$__OVERRIDES"
  fi
done
# shellcheck disable=SC1091
source "${WORK_DIR}/deploy/defaults.env"
# shellcheck disable=SC1090
source "$__OVERRIDES"
rm -f "$__OVERRIDES"

echo "install: CTID=${CTID} hostname=${CT_HOSTNAME} ip=${CT_IP}/${CT_CIDR} storage=${STORAGE}"

if pct status "$CTID" >/dev/null 2>&1; then
  echo "install: CT ${CTID} already exists — skipping pct create (will re-bootstrap)"
  pct set "$CTID" --hostname "$CT_HOSTNAME" || true
else
  TEMPLATE_PATH="${TEMPLATE_STORAGE}:vztmpl/${OSTEMPLATE}"
  if ! pveam list "$TEMPLATE_STORAGE" 2>/dev/null | grep -q "$OSTEMPLATE"; then
    echo "install: downloading template ${OSTEMPLATE}"
    pveam update
    pveam download "$TEMPLATE_STORAGE" "$OSTEMPLATE"
  fi

  pct create "$CTID" "$TEMPLATE_PATH" \
    --hostname "$CT_HOSTNAME" \
    --cores "$CORES" \
    --memory "$MEMORY" \
    --swap 512 \
    --rootfs "${STORAGE}:${DISK_GB}" \
    --net0 "name=eth0,bridge=${BRIDGE},ip=${CT_IP}/${CT_CIDR},gw=${CT_GW}" \
    --unprivileged "$UNPRIVILEGED" \
    --features "$FEATURES" \
    --onboot "$START_ON_BOOT" \
    --start 1

  sleep 5
fi

pct start "$CTID" >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if pct exec "$CTID" -- true >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

SECRETS_SRC=""
if [[ -n "${SECRETS_FILE:-}" && -f "$SECRETS_FILE" ]]; then
  SECRETS_SRC="$SECRETS_FILE"
elif [[ -f /root/nicks-billing.env ]]; then
  SECRETS_SRC=/root/nicks-billing.env
elif [[ -f "${WORK_DIR}/.env" ]]; then
  SECRETS_SRC="${WORK_DIR}/.env"
fi
if [[ -z "$SECRETS_SRC" ]]; then
  echo "install: no secrets file" >&2
  echo "  write /root/nicks-billing.env (DATABASE_URL, S3_*, SMTP_*, RAY_*)" >&2
  exit 1
fi

echo "install: pushing ${SECRETS_SRC} -> ${ENV_FILE}"
CREDS_TMP=$(mktemp)
cp "$SECRETS_SRC" "$CREDS_TMP"
sed -i 's/\r$//' "$CREDS_TMP"
# Windows checkout path must not survive into the CT.
sed -i '/^RAY_HIVE=/d' "$CREDS_TMP"
grep -q '^HOST=' "$CREDS_TMP" || echo 'HOST=0.0.0.0' >>"$CREDS_TMP"
pct exec "$CTID" -- bash -lc "mkdir -p $(dirname "$ENV_FILE") $(dirname "$APP_ROOT")"
pct push "$CTID" "$CREDS_TMP" "$ENV_FILE"
rm -f "$CREDS_TMP"

echo "install: syncing the whole app tree into CT ${APP_ROOT}"
pct exec "$CTID" -- bash -lc "rm -rf ${APP_ROOT} && mkdir -p ${APP_ROOT}"
tar_app "$WORK_DIR" | pct exec "$CTID" -- tar -C "$APP_ROOT" -xf -

echo "install: bootstrap inside CT"
pct exec "$CTID" -- env \
  APP_ROOT="$APP_ROOT" \
  APP_USER="$APP_USER" \
  ENV_FILE="$ENV_FILE" \
  RAY_HIVE_ROOT="$RAY_HIVE_ROOT" \
  RAY_HIVE_CLONE_URL="$RAY_HIVE_CLONE_URL" \
  bash "${APP_ROOT}/deploy/bootstrap-ct.sh"

if [[ -n "${HA_GROUP:-}" ]]; then
  echo "install: adding to HA"
  ha-manager add "ct:${CTID}" || true
fi

echo
echo "Done."
echo "  CT ${CTID} (${CT_HOSTNAME}) @ http://${CT_IP}:8787"
echo "  logs: pct exec ${CTID} -- journalctl -u nicks-billing -f"
echo "  status: pct exec ${CTID} -- systemctl status nicks-billing"
echo "  enter: pct enter ${CTID}"
