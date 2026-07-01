# Exposing the local Docker container via a Cloudflare Quick Tunnel

This sets up temporary public HTTPS access to the API running in Docker on this
machine, so it can be reached from a machine on a different network. It uses a
Cloudflare **quick tunnel** — no Cloudflare login or domain required, but the
URL is random and has no uptime guarantee (fine for demos/testing, not for
production — see "Limitations" below).

## Prerequisites

- Docker Desktop running, with the app started:
  ```powershell
  docker compose -f d:\token_usage_calculator\compose.yaml up -d
  ```
- Confirm it's listening locally:
  ```powershell
  Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing
  ```

## 1. Download cloudflared

A single static binary, no installer needed:

```powershell
$dir = "d:\token_usage_calculator\tools"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$dir\cloudflared.exe" -UseBasicParsing
& "$dir\cloudflared.exe" --version
```

## 2. Start the quick tunnel

Points the tunnel at the local container's port (8000):

```powershell
cd d:\token_usage_calculator\tools
.\cloudflared.exe tunnel --url http://localhost:8000
```

Keep this terminal open — closing it kills the tunnel. Look for output like:

```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://<random-words>.trycloudflare.com                                                  |
+--------------------------------------------------------------------------------------------+
```

## 3. Verify from anywhere

```powershell
Invoke-WebRequest -Uri "https://<random-words>.trycloudflare.com/health" -UseBasicParsing
```

Should return `{"status":200}`. It may take a few seconds to propagate after
the tunnel first comes up (expect a transient `502` until it does).

Other useful routes once it's up:
- `https://<random-words>.trycloudflare.com/docs` — Swagger UI
- `https://<random-words>.trycloudflare.com/api/v1/...` — API routes

## Limitations

- **No uptime guarantee** — Cloudflare's ToS frames quick tunnels as a
  "try it out" feature, not for production traffic.
- **Random URL** — changes every time `cloudflared` is restarted.
- **Depends on this machine** — the tunnel only works while this machine is
  on, Docker is running, and the `cloudflared` process is alive. Sleep,
  reboot, or closing the terminal all kill it.

## Upgrading to a stable URL (named tunnel)

If a domain is added to your Cloudflare account, a **named tunnel** gives a
fixed hostname (e.g. `api.yourdomain.com`) that survives restarts:

```powershell
.\cloudflared.exe tunnel login          # opens browser, pick the zone/domain
.\cloudflared.exe tunnel create token-usage-calculator
.\cloudflared.exe tunnel route dns token-usage-calculator api.yourdomain.com
```

Then create `config.yml` next to `cloudflared.exe`:

```yaml
tunnel: token-usage-calculator
credentials-file: <path printed by 'tunnel create'>
ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

Run it with:

```powershell
.\cloudflared.exe tunnel run token-usage-calculator
```

This still requires this machine to stay on — for a fully machine-independent
deployment, host the container on a cloud VM instead.
