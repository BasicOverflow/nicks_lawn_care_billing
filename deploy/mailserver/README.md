# docker-mailserver setup (self-hosted on basicoverflow.com)

## DNS (Cloudflare)

1. **A** `mail.basicoverflow.com` → public IP of the mail host (or DDNS target).
2. **MX** `@` or billing subdomain → `mail.basicoverflow.com` (priority 10).
3. **SPF** TXT: `v=spf1 mx a:mail.basicoverflow.com ~all`
4. After first start, copy DKIM from the container and add as TXT (`mail._domainkey`).
5. **DMARC** TXT `_dmarc`: `v=DMARC1; p=none; rua=mailto:postmaster@basicoverflow.com`

## Create mailbox

```bash
docker exec -it nicks-mailserver setup email add billing@basicoverflow.com 'your-password'
```

Point the app `.env` at this SMTP:

```
SMTP_HOST=mail.basicoverflow.com   # or LAN IP
SMTP_PORT=587
SMTP_USER=billing@basicoverflow.com
SMTP_PASSWORD=…
SMTP_FROM=billing@basicoverflow.com
SMTP_TLS=1
```

Open ports 25/587 (and 465 if used) on the firewall; residential ISP blocks on 25 are common — use a VPS if needed.
