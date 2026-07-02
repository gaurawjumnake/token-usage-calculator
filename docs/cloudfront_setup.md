# EC2 Deployment Guide — Token Usage Calculator

## Infrastructure Overview

```
Frontend (Cloudflare Pages) → CloudFront (HTTPS) → EC2 t2.micro → Docker (FastAPI)
```

| Resource | Value |
|---|---|
| EC2 Instance | t2.micro — ap-southeast-2 (Sydney) |
| EC2 Public IP | [REDACTED_IPV4_ADDRESS_1] |
| EC2 Public DNS | ec2-54-66-252-127.ap-southeast-2.compute.amazonaws.com |
| CloudFront ID | E3OPIH3F8WDZR2 |
| CloudFront URL | https://d2eqlq4nl7lhkr.cloudfront.net |
| SSH Key | token-calc-key.pem |

---

## SSH Access (from local Windows PowerShell)

```powershell
ssh -i "C:\Users\gauraw.jumnake\Downloads\token-calc-key.pem" ubuntu@54.66.252.127
```

> If permission error on first use:
> ```powershell
> icacls "C:\Users\gauraw.jumnake\Downloads\token-calc-key.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
> ```

---

## App Management (run inside SSH terminal)

### Start / Rebuild after code change
```bash
cd /home/ubuntu/app
git pull
sudo docker compose up -d --build
```

### Restart after .env change only
```bash
cd /home/ubuntu/app
sudo docker compose restart
```

### Stop the app
```bash
cd /home/ubuntu/app
sudo docker compose down
```

### Check app status
```bash
sudo docker compose ps
```

### Check app logs
```bash
sudo docker compose logs -f
```

### Health check
```bash
curl http://localhost:8000/health
```

---

## Deployment Workflow

### When you push code changes to GitHub:

1. **Local** — push to GitHub:
   ```powershell
   git add .
   git commit -m "your message"
   git push
   ```

2. **EC2** — pull and rebuild:
   ```bash
   cd /home/ubuntu/app && git pull && sudo docker compose up -d --build
   ```

### What changed → which command:

| Changed | Command |
|---|---|
| Code files (`main.py`, `backend/`) | `sudo docker compose up -d --build` |
| Only `.env` | `sudo docker compose restart` |

---

## File Transfer (from local PowerShell to EC2)

### Copy single file
```powershell
scp -i "C:\Users\gauraw.jumnake\token-calc-key.pem" "d:\token_usage_calculator\.env" ubuntu@[REDACTED_IPV4_ADDRESS_2]:/home/ubuntu/app/.env
```

### Copy entire folder (excluding .venv)
Use `git clone` on EC2 instead — much cleaner.

---

## First-Time Setup (reference only)

### 1. Install Docker on EC2
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install docker.io -y
sudo docker compose up  # uses docker-compose-v2 already installed
sudo usermod -aG docker ubuntu
newgrp docker
```

### 2. Clone app
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git /home/ubuntu/app
```

### 3. Copy .env
```powershell
scp -i "C:\Users\gauraw.jumnake\token-calc-key.pem" "d:\token_usage_calculator\.env" ubuntu@[REDACTED_IPV4_ADDRESS_2]:/home/ubuntu/app/.env
```

### 4. Build and run
```bash
cd /home/ubuntu/app && sudo docker compose up -d --build
```

---

## CloudFront (AWS CLI)

### Check deployment status
```powershell
aws cloudfront get-distribution --id E3OPIH3F8WDZR2 --query "Distribution.Status" --output text
```

### Invalidate cache (force CloudFront to refresh)
```powershell
aws cloudfront create-invalidation --distribution-id E3OPIH3F8WDZR2 --paths "/*"
```

### Test CloudFront health
```powershell
Invoke-WebRequest -Uri "https://d2eqlq4nl7lhkr.cloudfront.net/health" -UseBasicParsing
```

---

## Security Notes

- Port 22 (SSH) open to your IP only — update security group if your IP changes
- Port 8000 open to anywhere (required for CloudFront to reach EC2)
- All requests validated by `X-Origin-Secret` header (set in CloudFront, checked by FastAPI)
- Direct access to `http://[REDACTED_IPV4_ADDRESS_2]:8000` is blocked by the middleware unless the secret header is present

### If your IP changes and SSH is locked out:
1. Go to AWS Console → EC2 → Security Groups
2. Edit inbound rule for port 22
3. Update source to your new IP

---

## Environment Variables (.env on EC2)

Edit directly on EC2:
```bash
nano /home/ubuntu/app/.env
```

Add/update a variable:
```bash
echo 'KEY=VALUE' >> /home/ubuntu/app/.env
```

View current .env:
```bash
cat /home/ubuntu/app/.env
```

After editing .env, restart:
```bash
cd /home/ubuntu/app && sudo docker compose restart
```

---

## Costs

| Period | Cost |
|---|---|
| Now → 12 months | $0 (free tier) |
| After 12 months | ~$8.60/month (t2.micro 24/7) |

CloudFront: Free tier — 1TB/month, 10M requests/month.