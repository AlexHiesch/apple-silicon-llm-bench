# Claude Code WebFetch on corp / ThinkingCap

## Symptoms

- `Claude Code is unable to fetch from <host>` while `curl` to the same URL returns HTTP 200.
- Or, after the fetch succeeds: `403 key not allowed to access model … Tried to access claude-haiku-4.5`.

## Root causes

1. **Domain preflight blocklist** — before fetching, Claude Code calls
   `https://claude.ai/api/web/domain_info?domain=<host>`. Some public sites
   return `{"can_fetch":false}` (observed: `ard.de`, `tagesschau.de`,
   `spiegel.de`; `example.com` / `zdf.de` are allowed). Claude then refuses
   the fetch. The same call fails hard when corp policy blocks `claude.ai`.

2. **Haiku summarizer model** — after HTML is downloaded, WebFetch asks the
   Haiku-tier model to extract content. Against LiteLLM ThinkingCap keys that
   only allow `thinkingcap`, an unmapped `claude-haiku-4.5` request 403s.

## Fixes

### Local Claude Code (Mac)

In `~/.claude/settings.json`:

```json
{
  "skipWebFetchPreflight": true
}
```

Already applied for this workstation. When pointing at ThinkingCap, also set:

```bash
export ANTHROPIC_DEFAULT_HAIKU_MODEL=thinkingcap
export ANTHROPIC_SMALL_FAST_MODEL=thinkingcap
export ANTHROPIC_DEFAULT_SONNET_MODEL=thinkingcap
export ANTHROPIC_DEFAULT_OPUS_MODEL=thinkingcap
```

(`ANTHROPIC_SMALL_FAST_MODEL` wins over `ANTHROPIC_DEFAULT_HAIKU_MODEL`.)

### Harbor / Pier AA Index

- `run_harbor.py` / `run_pier.py` pass the ThinkingCap model on every Anthropic
  tier env var (including `ANTHROPIC_SMALL_FAST_MODEL`).
- `scripts/patch_harbor_claude_webfetch.py` (run from
  `run_aa_index_workstation.sh`) writes
  `{"skipWebFetchPreflight":true}` into the agent `CLAUDE_CONFIG_DIR` and pins
  `ANTHROPIC_SMALL_FAST_MODEL` inside Harbor’s Claude adapter.

## Verify

```bash
curl -s 'https://claude.ai/api/web/domain_info?domain=www.ard.de'
# {"domain":"www.ard.de","can_fetch":false}  ← preflight alone blocks

# With skipWebFetchPreflight + ThinkingCap remap, WebFetch should return:
# title ≈ "ARD.de – TV, Radio, Streams, News, Sport und alles zur ARD | ard.de"
```
