# Nick's Lawn Care Billing

FastAPI + static UI for OCR sheet ingest and monthly PDF invoices. OCR uses a fixed **qwen25-vl-3b** novel pipeline on ray-hive. Data lives in homelab **Postgres Fast** and **MinIO Fast**. Email uses self-hosted SMTP ([docker-mailserver](https://github.com/docker-mailserver/docker-mailserver)) on your domain.

## Layout

```
app/                 # FastAPI (db, storage, emailer, routes, billing)
ocr/                 # novel OCR package (extract_sheet / load_model)
static/              # data.html, billing.html, css, js
deploy/
  Dockerfile
  docker-compose.yml
  install.sh              # Proxmox paste-install (home-media LXC)
  bootstrap-ct.sh
  defaults.env
  nicks-billing.service
  mailserver/             # compose profile + DNS notes
ground_truth/        # sample sheets (optional)
run.py
requirements.txt
.env.example
```

## Pages

- **Data** (`/data`) — upload photos → OCR → review / one-shot correction → conflict check → confirm into knowledge for a month.
- **Billing** (`/billing`) — generate PDFs → download zip / tax TSV → **Email all** via SMTP.
- Header: model status, **Load model**, global progress bar.

## Environment

Copy `.env.example` → `.env` and fill secrets:

| Variable | Purpose |
|----------|---------|
| `DATABASE_URL` | Postgres Fast, DB `nicks_billing` |
| `S3_*` | MinIO Fast endpoint + keys + bucket `nicks-lawn-billing` |
| `RAY_*` / `RAY_HIVE` | ray-hive for OCR deploy |
| `SMTP_*` | Submission to docker-mailserver (`mail.basicoverflow.com` or LAN IP) |

## Local (Windows)

```powershell
py -3 -m pip install -r requirements.txt
# editable ray-hive if not already installed:
py -3 -m pip install -e C:\Users\Peter\Desktop\ray-hive

py -3 scripts\ensure_infra.py   # create DB + schema + S3 bucket
py -3 run.py                    # http://127.0.0.1:8787
```

1. Open **Data** → **Load model** (waits for Serve).
2. Upload a sheet for month `YYYY-MM` → review → Confirm.
3. **Billing** → Generate PDFs → Download zip. Email all when SMTP is configured.

## Docker Compose

From the repo root (or `deploy/`):

```bash
cp .env.example .env   # edit
docker compose -f deploy/docker-compose.yml up -d --build
```

App listens on **8787**. The container must reach Postgres `:5433`, MinIO `:9000`, and Ray Serve on the LAN.

Optional lab mailstack:

```bash
docker compose -f deploy/docker-compose.yml --profile mail up -d
```

See [deploy/mailserver/README.md](deploy/mailserver/README.md) for MX/SPF/DKIM and creating `billing@…`.

## LXC on home-media

Same pattern as Lexis Markets CT **117**: root on **home-media** (`10.0.0.20`) runs `deploy/install.sh`, which `pct create`s an unprivileged Debian 12 CT and starts systemd inside it. No Docker in the guest. Defaults: CT **118**, hostname `nicks-billing`, `10.0.0.60:8787`.

Put secrets on the Proxmox host first (not in git):

```bash
# /root/nicks-billing.env  — copy from .env.example and fill S3 / SMTP
```

Redeploy replaces the whole app in CT 118 and restarts the service. It does not copy files one by one. Re-running skips `pct create`.

From a tree already on the Proxmox host:

```bash
SRC_DIR=/root/nicks-billing-src bash /root/nicks-billing-src/deploy/install.sh
```

From the Gitea mirror of [BasicOverflow/nicks_lawn_care_billing](https://github.com/BasicOverflow/nicks_lawn_care_billing):

```bash
bash -c "$(curl -fsSL http://10.0.0.52:3000/admin/nicks_lawn_care_billing/raw/branch/main/deploy/install.sh)"
```

The guest keeps `/etc/nicks-billing.env`. Sheet photos, gold labels, and `.env` are not copied in. OCR still uses ray-hive on the cluster; the CT only submits jobs. It clones `ray-hive` to `/opt/ray-hive` as that working directory and does not install vLLM.

```bash
pct exec 118 -- systemctl status nicks-billing
pct exec 118 -- journalctl -u nicks-billing -f
```

## SMTP / DNS checklist

On Cloudflare (or your DNS for `basicoverflow.com`):

1. **A** `mail.basicoverflow.com` → mail host public IP  
2. **MX** → `mail.basicoverflow.com`  
3. **SPF** TXT `v=spf1 mx a:mail.basicoverflow.com ~all`  
4. DKIM from docker-mailserver → TXT  
5. **DMARC** soft-fail to start  

App env: `SMTP_HOST`, `SMTP_PORT=587`, `SMTP_USER` / `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TLS=1`. Downloads still work if SMTP is down; Email all skips clients without email.

## OCR package

Photos are work-completed sheets only. Mowing prices, hedge prices, and contact
details are typed on the Data page. Knowledge-guided OCR sends the full photo
many times in parallel; each request asks the model for a small chunk of
Postgres-known clients (default 4 names per request). A follow-up pass fills
missing names and blank date cells. No tile/crop splitting.

The OCR benches score `ground_truth/work_completed/` only.

Seed the knowledge base from Nick’s office files (spreadsheet, hedges list, phones,
mailing labels). Gold JSON under `ground_truth/` is for scoring tests only and
is not imported.

```powershell
py -3 scripts\import_tmp_knowledge.py
```

```python
import ocr
ocr.load_model()
sheet = ocr.extract_sheet("photo.jpg", sheet_kind="work")
fixed = ocr.apply_correction("photo.jpg", sheet, "Fix MOSHER price to $80")
ocr.unload_model()
```

Model id is fixed: `qwen25-vl-3b`. Chunk size: `NINI_OCR_CHUNK` (default 4).
Legacy tile multipass: `ocr.extract_sheet_legacy_multipass(...)`.

