#!/bin/zsh
# Public demo: passcode-gated, ingestion off, capped usage. Binds to localhost only;
# Tailscale Funnel is the sole way in from outside.
#   ./run_demo.sh                                  start the app
#   tailscale funnel --bg --https=8443 8502        publish it
#   tailscale funnel --https=8443 off              unpublish (kill switch)
set -euo pipefail
cd "${0:A:h}"
[[ -f .env.demo ]] || { echo "missing .env.demo (needs DEMO_PASSCODE=...)"; exit 1; }
set -a; source .env.demo; set +a
export PUBLIC_MODE=1
exec .venv/bin/streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8502 --server.headless true \
  --client.toolbarMode viewer --browser.gatherUsageStats false
