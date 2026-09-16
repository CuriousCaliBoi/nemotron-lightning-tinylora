#!/usr/bin/env bash
# Show the loop's state: pointers, last commits, evaluations, live status.
set -uo pipefail
STATE_DIR="${STATE_DIR:-/home/nimitz/cl-state/nemotron35-gsm8k}"
echo "HEAD: $(cat "$STATE_DIR/HEAD" 2>/dev/null || echo none)   BEST: $(cat "$STATE_DIR/BEST" 2>/dev/null || echo none)"
echo "--- status.json ---"
python3 - "$STATE_DIR/status.json" <<'EOF' 2>/dev/null || echo "(no status yet)"
import json, sys
s = json.load(open(sys.argv[1]))
keys = ("phase", "step", "head", "served", "updated_at", "error", "step_seconds")
print({k: s[k] for k in keys if k in s})
m = s.get("last_metrics") or {}
if m:
    print({k: round(m[k], 4) for k in ("reward_mean", "loss", "grad_norm", "adapter_norm", "tis_truncated_fraction", "clip_fraction") if k in m})
EOF
echo "--- last commits ---"
tail -n 5 "$STATE_DIR/history.jsonl" 2>/dev/null | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line); m = r.get("metrics", {})
    print(r["release_id"], r.get("operation"), "step", r.get("step"), "reward", m.get("reward_mean"), "loss", m.get("loss"), "step_s", m.get("update_seconds"))' 2>/dev/null
echo "--- evaluations ---"
tail -n 6 "$STATE_DIR/evals.jsonl" 2>/dev/null | python3 -c '
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(r["release_id"], "step", r["step"], r["reason"], f"{r[\"correct\"]}/{r[\"rows\"]}", f"{100*r[\"accuracy\"]:.1f}%", "clipped", round(r["clipped_fraction"], 3), f"{r[\"seconds\"]}s")' 2>/dev/null
echo "--- vLLM ---"
curl -s http://127.0.0.1:30000/v1/models | python3 -c 'import json,sys; print([m["id"] for m in json.load(sys.stdin)["data"]])' 2>/dev/null || echo "(vLLM not answering)"
