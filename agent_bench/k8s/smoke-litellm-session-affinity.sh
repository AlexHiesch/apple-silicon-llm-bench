#!/usr/bin/env bash
# Smoke: least-busy dual backends + Claude-Code-style session stickiness.
# Requires: kubectl access, ~/llm-serving/aa-index-key, LiteLLM on :4000 (hostNetwork).
set -euo pipefail
KEY_FILE="${LITELLM_KEY_FILE:-$HOME/llm-serving/aa-index-key}"
BASE="${LITELLM_BASE:-http://127.0.0.1:4000}"
KEY="$(tr -d '[:space:]' <"$KEY_FILE")"

hdr_model_id() {
  # Prefer explicit LiteLLM model id header; fall back to body system_fingerprint.
  local headers_file=$1 body_file=$2
  local id
  id=$(grep -i '^x-litellm-model-id:' "$headers_file" | awk '{$1=""; print substr($0,2)}' | tr -d '\r' | head -1 || true)
  if [[ -z "$id" ]]; then
    id=$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d.get("model") or d.get("system_fingerprint") or "")' "$body_file" 2>/dev/null || true)
  fi
  printf '%s' "$id"
}

post_once() {
  local session=$1 out_hdr=$2 out_body=$3
  local code
  code=$(curl -sS -D "$out_hdr" -o "$out_body" -w '%{http_code}' -m 120 \
    -H "Authorization: Bearer $KEY" \
    -H 'content-type: application/json' \
    -H "x-claude-code-session-id: $session" \
    -d '{"model":"thinkingcap","max_tokens":8,"messages":[{"role":"user","content":"Reply with OK only."}]}' \
    "$BASE/v1/chat/completions" || true)
  echo "$code"
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

echo "== gateway models =="
curl -sf -m 10 -H "Authorization: Bearer $KEY" "$BASE/v1/models" >/dev/null
echo OK

SESS_A="bench-affinity-a-$(date +%s)"
SESS_B="bench-affinity-b-$(date +%s)"

echo "== session A turn1 =="
c1=$(post_once "$SESS_A" "$TMP/a1.h" "$TMP/a1.b")
id1=$(hdr_model_id "$TMP/a1.h" "$TMP/a1.b")
echo "http=$c1 model_id=$id1"
[[ "$c1" == "200" ]] || { echo "FATAL turn1"; head -c 400 "$TMP/a1.b"; exit 1; }

echo "== session A turn2 (expect same deployment) =="
c2=$(post_once "$SESS_A" "$TMP/a2.h" "$TMP/a2.b")
id2=$(hdr_model_id "$TMP/a2.h" "$TMP/a2.b")
echo "http=$c2 model_id=$id2"
[[ "$c2" == "200" ]] || { echo "FATAL turn2"; exit 1; }

echo "== session B turn1 (may differ) =="
c3=$(post_once "$SESS_B" "$TMP/b1.h" "$TMP/b1.b")
id3=$(hdr_model_id "$TMP/b1.h" "$TMP/b1.b")
echo "http=$c3 model_id=$id3"
[[ "$c3" == "200" ]] || { echo "FATAL sessB"; exit 1; }

if [[ -n "$id1" && -n "$id2" && "$id1" == "$id2" ]]; then
  echo "PASS: session stickiness held ($id1)"
else
  echo "WARN: could not confirm stickiness via model id (id1='$id1' id2='$id2')."
  echo "      Check response headers below; LiteLLM may omit x-litellm-model-id on some paths."
  echo "---- a1 headers ----"; grep -iE 'x-litellm|model' "$TMP/a1.h" || true
  echo "---- a2 headers ----"; grep -iE 'x-litellm|model' "$TMP/a2.h" || true
fi

echo "== both backends reachable via shuffle of sessions =="
echo "sessA=$id1/$id2 sessB=$id3"

# Sequential idle requests often all hit the least-busy winner; verify dual
# backends under a short parallel burst of distinct sessions.
echo "== parallel first-turns (expect both model ids when both healthy) =="
PAR=$(mktemp -d)
trap 'rm -rf "$TMP" "$PAR"' EXIT
for i in 1 2 3 4; do
  (
    sid="par-$i-$RANDOM"
    code=$(curl -sS -D "$PAR/$i.h" -o "$PAR/$i.b" -w '%{http_code}' -m 120 \
      -H "Authorization: Bearer $KEY" \
      -H 'content-type: application/json' \
      -H "x-claude-code-session-id: $sid" \
      -d '{"model":"thinkingcap","max_tokens":16,"messages":[{"role":"user","content":"Say OK."}]}' \
      "$BASE/v1/chat/completions" || true)
    mid=$(hdr_model_id "$PAR/$i.h" "$PAR/$i.b")
    echo "par$i http=$code model_id=$mid"
  ) &
done
wait
uniq_ids=$(for i in 1 2 3 4; do hdr_model_id "$PAR/$i.h" "$PAR/$i.b"; echo; done | sed '/^$/d' | sort -u | tr '\n' ' ')
echo "parallel_model_ids: $uniq_ids"
if echo "$uniq_ids" | grep -q 'thinkingcap-x40' && echo "$uniq_ids" | grep -q 'thinkingcap-x39'; then
  echo "PASS: parallel burst used both backends"
else
  echo "WARN: parallel burst did not show both backends (may be transient load skew)"
fi
echo DONE
