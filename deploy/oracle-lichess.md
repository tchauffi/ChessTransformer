# Deploy the lichess bot on Oracle Cloud (Always Free ARM)

Runs `ct-bot lichess` as an always-on systemd service on a free Oracle Ampere
(ARM64) VM. The bot is **outbound-only** (it connects out to lichess), so no
inbound ports, public IP, domain, or TLS are needed.

Prereqs: an Oracle Cloud account, a BOT lichess account + token with the
`bot:play` scope, and the int8 model built locally
(`scripts/export_onnx.py data/models/pos2move_v2.1 --ema --quantize`).

---

## 1. Provision the instance (Oracle web console)

- **Compute → Instances → Create instance**
- **Image**: Canonical Ubuntu 24.04
- **Shape**: `VM.Standard.A1.Flex` (Ampere/ARM, Always Free). 1 OCPU / 6 GB is
  plenty; 2 OCPU / 12 GB builds faster.
- Add your SSH public key.
- Networking: defaults are fine — **no ingress rules needed** (outbound only).
- Create, then note the public IP for SSH.

```bash
ssh ubuntu@<INSTANCE_IP>
```

## 2. Install build tools (on the instance)

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev git curl
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

## 3. Build ct-bot natively (ARM64)

```bash
git clone https://github.com/tchauffi/ChessTransformer.git
cd ChessTransformer
cargo build --release --manifest-path rust/Cargo.toml -p ct-bot
```

`ort` downloads the linux-aarch64 ONNX Runtime automatically. The CPU build is
self-contained (no shared libs to ship).

## 4. Install binary + model into /opt/ct-bot

```bash
sudo mkdir -p /opt/ct-bot
sudo cp rust/target/release/ct-bot /opt/ct-bot/
```

The model is not in git — copy it up **from your local machine**:

```bash
# (run locally, not on the instance)
scp data/models/pos2move_v2.1/model_ema.int8.onnx ubuntu@<INSTANCE_IP>:/tmp/
# back on the instance:
sudo mv /tmp/model_ema.int8.onnx /opt/ct-bot/
```

## 5. Token (kept out of the unit file)

```bash
sudo mkdir -p /etc/ct-bot
echo 'LICHESS_BOT_TOKEN=lip_your_token_here' | sudo tee /etc/ct-bot/lichess.env >/dev/null
sudo chmod 600 /etc/ct-bot/lichess.env
```

## 6. Install and start the service

```bash
sudo cp deploy/ct-bot-lichess.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now ct-bot-lichess
```

## 7. Verify

```bash
sudo systemctl status ct-bot-lichess --no-pager
journalctl -u ct-bot-lichess -f       # expect: "connected as <bot> (BOT)"
```

Challenge the bot from your main lichess account (Standard · Real-time · 5+3 ·
Casual). It auto-accepts blitz/rapid.

---

## 8. (Optional) CI/CD: auto-build + deploy on push

`.github/workflows/deploy-lichess.yml` builds the aarch64 binary on a free
GitHub arm64 runner and ships it to the instance on every push to `rust/**`
(or via "Run workflow"). The model + token stay on the box (steps 4–5); CI only
updates the binary, which is what changes with code.

**One-time box prep so CI can deploy without an interactive sudo password:**

```bash
# let the deploy user own the binary dir (no sudo needed to replace the binary)
sudo chown -R "$USER":"$USER" /opt/ct-bot
# allow exactly one passwordless command: restarting the service
echo "$USER ALL=(root) NOPASSWD: /usr/bin/systemctl restart ct-bot-lichess" \
  | sudo tee /etc/sudoers.d/ct-bot >/dev/null
sudo chmod 440 /etc/sudoers.d/ct-bot
```

**Add the repo secrets** (GitHub → Settings → Secrets and variables → Actions):

| Secret | Value |
|---|---|
| `ORACLE_HOST` | the instance public IP |
| `ORACLE_USER` | ssh user (e.g. `ubuntu`) |
| `ORACLE_SSH_KEY` | a private key authorized on the instance |

CI needs **inbound SSH (port 22)** to the instance — that's the one ingress
exception (key-based auth only). Push to `main` (touching `rust/**`) or run the
workflow manually, and it builds → scp → `systemctl restart` → checks the
service is active.

To rebuild the **model** (only when the weights change): run
`scripts/export_onnx.py data/models/pos2move_v2.1 --ema --quantize` and `scp`
the new `model_ema.int8.onnx` to `/opt/ct-bot/`, then restart.

## Notes

- **Stop any other running instance of the bot** (e.g. your dev machine) before
  starting this one — two clients on the same account fight over the event
  stream.
- Manual update after a code change (if not using CI): `git pull && cargo build --release -m rust/Cargo.toml -p ct-bot && cp rust/target/release/ct-bot /opt/ct-bot/ && sudo systemctl restart ct-bot-lichess`.
- Tuning: edit `ExecStart` in the unit (`--speeds`, `--max-games`, `--move-temp`
  for opening variety). Sims are clock-budgeted and capped at 1200 (see
  timeman.rs).
- Logs: `journalctl -u ct-bot-lichess` (per-move lines at RUST_LOG=info).
