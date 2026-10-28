#!/usr/bin/env bash
# Start (or restart) the LiteLLM gateway in Docker, reachable only from this machine.
#   LITELLM_MASTER_KEY must be set in your environment or in .env (any random string).
#   Generate one:  python3 -c "import secrets; print('sk-' + secrets.token_urlsafe(24))"
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${LITELLM_MASTER_KEY:-}" ] && [ -f .env ]; then
  # read just this one variable; never print it
  LITELLM_MASTER_KEY="$(grep -E '^LITELLM_MASTER_KEY=' .env | head -1 | cut -d= -f2-)"
fi
: "${LITELLM_MASTER_KEY:?Set LITELLM_MASTER_KEY (see the comment at the top of this file)}"
export LITELLM_MASTER_KEY
docker rm -f litellm-gateway >/dev/null 2>&1 || true
# --network host: the container sees the same localhost as WSL, which is how it reaches Ollama.
# --host 127.0.0.1: with host networking, the default 0.0.0.0 would open the port to your whole network.
docker run -d --name litellm-gateway --network host \
  --memory 1g --cap-drop ALL --security-opt no-new-privileges \
  -v "$PWD/gateway/config.yaml:/app/config.yaml:ro" \
  -e LITELLM_MASTER_KEY \
  litellm/litellm:v1.106.5 --config /app/config.yaml --host 127.0.0.1 --port 4000
echo "gateway starting on http://127.0.0.1:4000 (docker logs litellm-gateway)"
