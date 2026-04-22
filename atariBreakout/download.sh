#!/bin/bash
set -euo pipefail

REMOTE_USER="jacobgar"
REMOTE_HOST="nova-login-1.its.iastate.edu"

# Build a list of episode checkpoint filenames that already exist locally.
# The cluster will skip sending these — saves gigabytes of transfer.
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"
EXISTING_CHECKPOINTS=""
if ls "$LOCAL_DIR"/models/dqn_episode_*.keras >/dev/null 2>&1; then
    EXISTING_CHECKPOINTS=$(ls "$LOCAL_DIR"/models/dqn_episode_*.keras \
        | xargs -n1 basename \
        | tr '\n' ' ')
fi

echo "Downloading models and logs from the cluster..."
echo "Skipping $(echo "$EXISTING_CHECKPOINTS" | wc -w | tr -d ' ') episode checkpoints already present locally."

TMP_DIR=$(mktemp -d)
cleanup() {
    rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${TMP_DIR}/models" "${TMP_DIR}/logs"

# ── 1. Download all logs ──────────────────────────────────────────────────────
echo "Syncing logs..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail
cd ~/atariBreakout
mkdir -p logs
ls logs/ >/dev/null 2>&1 || true
tar -czf - logs
' | tar -xzf - -C "${TMP_DIR}"

# ── 2. Download critical model files (best + state) ──────────────────────────
echo "Downloading critical model files (dqn_best.keras, training_state.json)..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" '
set -euo pipefail
cd ~/atariBreakout/models
FILES=""
[ -f dqn_best.keras ]        && FILES="$FILES dqn_best.keras"
[ -f training_state.json ]   && FILES="$FILES training_state.json"
[ -f dqn_final.keras ]       && FILES="$FILES dqn_final.keras"
if [ -n "$FILES" ]; then
    tar -czf - $FILES
else
    tar -czf - --files-from /dev/null
fi
' | tar -xzf - -C "${TMP_DIR}/models" 2>/dev/null || true

# ── 3. Download only NEW episode checkpoints ──────────────────────────────────
echo "Downloading new episode checkpoints (skipping ones you already have)..."
EXISTING_JSON=$(echo "$EXISTING_CHECKPOINTS" | tr ' ' '\n' \
    | grep -v '^$' \
    | jq -Rsc 'split("\n") | map(select(length > 0))' 2>/dev/null \
    || python3 -c "
import sys, json
names = '''${EXISTING_CHECKPOINTS}'''.split()
print(json.dumps(names))
")

ssh -o ServerAliveInterval=30 "${REMOTE_USER}@${REMOTE_HOST}" \
    "EXISTING='${EXISTING_CHECKPOINTS}'" '
set -euo pipefail
cd ~/atariBreakout/models
NEW_FILES=""
for f in dqn_episode_*.keras; do
    [ -f "$f" ] || continue
    # Skip if already present locally
    case " $EXISTING " in
        *" $f "*) continue ;;
    esac
    NEW_FILES="$NEW_FILES $f"
done
COUNT=$(echo $NEW_FILES | wc -w)
echo "Sending $COUNT new episode checkpoint(s)..." >&2
if [ -n "$NEW_FILES" ]; then
    tar -czf - $NEW_FILES
else
    tar -czf - --files-from /dev/null
fi
' | tar -xzf - -C "${TMP_DIR}/models" 2>/dev/null || true

# ── 4. Atomic swap — only touch local files after a successful download ───────
mkdir -p models logs

# Merge new models in (never delete existing local files, only add/overwrite)
if ls "${TMP_DIR}/models/"* >/dev/null 2>&1; then
    cp -f "${TMP_DIR}"/models/* ./models/ 2>/dev/null || true
fi

# Replace logs entirely with remote copy
if ls "${TMP_DIR}/logs/"* >/dev/null 2>&1; then
    rm -rf logs
    mv "${TMP_DIR}/logs" ./logs
fi

echo ""
echo "Latest local training logs after sync:"
ls -lt logs/training_log_*.out 2>/dev/null | head -10 || echo "No local training_log_*.out files found."

echo ""
MODEL_COUNT=$(ls models/dqn_episode_*.keras 2>/dev/null | wc -l | tr -d ' ')
echo "Download complete! Local episode checkpoints: ${MODEL_COUNT}"
echo "Best model and state synced to models/"
