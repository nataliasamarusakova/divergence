## v10.4.2

Scheduler persistence restored to the v9 model: `recent_event_cache.json` and `timeframe_scan_state.json` remain in `data/` across runs. Workflow no longer deletes runtime data or runs per-cycle test/preflight stages. Commit step merges remote/local scheduler watermarks to protect against queued-run races.

# v10.3.8 Audited: fixed 7% SL + Ajay R5.41 S/R room integration

> **Current release status (2026-10-04): temporarily disabled in the VST workflow.** The S/R implementation remains intact for later re-enablement; it is not applied to entry admission or post-fill protection while `AJAY_SR_ROOM_ENABLED=false` / `AJAY_SR_ROOM_MODE=off`.

## Current behavior

The SR layer is deliberately **lazy**: it is not calculated for the whole universe.
It is requested only after a symbol has already survived the existing event, HTF,
15m trigger, funding, entry-quality and conflict checks, and immediately before the
BingX market order path.

The current source is **Binance SPOT 1H** (`https://data-api.binance.vision`) through the
existing Binance WireProxy in the VST workflow. This is a deliberate temporary choice. The signal
engine itself still uses Binance Futures for market data. Before switching the SR source to futures,
the same levels must be compared against the TradingView instrument/feed to ensure the source is correct.

## SR geometry policy

- The supplied Ajay R5.41 calculation remains standalone and is not mixed into
  `signals.py`.
- `cwidth` is treated as a conservative cluster-band half-width. This is a trading
  interpretation of the supplied Pine calculation, not a claim that it is the exact
  visual rectangle thickness used by an unseen Pine drawing block.
- LONG: resistance/supply above the current executable price is the opposing zone;
  support/demand below is supporting context.
- SHORT: support/demand below is the opposing zone; resistance/supply above is
  supporting context.
- Entry inside an opposing zone is rejected.
- An opposing zone that reaches or crosses TP1 is rejected.
- An opposing zone after TP1 is allowed; it does not block the entry and does not mutate
  TP2/TP3. This intentionally follows the current rule: only failure to have room for
  the first take-profit invalidates the trade.
- A supporting zone aligned with the trade direction is recorded as confirmation context
  only: LONG inside Demand/Support and SHORT inside Supply/Resistance. It never adds
  arbitrary legacy-score points and never overrides other entry conditions.
- Multiple candidate events for one symbol reuse one cached SR snapshot per latest
  closed 1H bar.
- S/R data is based only on closed 1H candles; no developing candle is used.
- `AJAY_SR_HTTP_PROXY` points to the same explicit Binance WireProxy used by the Futures
  market-data path in the VST workflow.
- Spot S/R requests use the same explicit HTTP proxy as the Binance Futures market-data path
  in the VST workflow; the provider is kept isolated so it can later be replaced with Futures.
- Spot symbols are resolved from live Binance Spot `exchangeInfo`, not by a hard-coded cross-venue
  alias table. Exact active `USDT` Spot symbols win; known quantity-prefixed logical symbols such as
  `1000SHIB`, `1000PEPE`, `1000BONK` and similar quantity-prefixed Binance perpetual symbols may
  use the corresponding unprefixed Spot pair when that pair exists. The numeric prefix changes the
  contract/index unit: for `1000SHIBUSDT`, Binance defines a 1,000-SHIB contract and its index/quoted
  price is 1,000x the SHIB/USDT Spot index. Therefore the Spot source uses `price_scale=1000` and
  `volume_scale=0.001` so S/R levels are in the same logical price/quantity unit as the signal/execution
  instrument. Ambiguous
  or missing Spot sources fail closed as `SR_SYMBOL_UNAVAILABLE`. A Spot HTTP 400 causes one fresh
  `exchangeInfo` resolution and one retry; the BingX execution catalog is never used as an S/R alias source.

## Safety / fallback

`AJAY_SR_REQUIRE_DATA=true` in the shipped VST workflow. If the SPOT S/R snapshot cannot be
obtained or is stale, only that candidate is blocked and logged as `SR_DATA_UNAVAILABLE`; the
whole engine is not stopped. Retryable provider/data failures are persisted in `trades.jsonl`
and exhausted after three attempts for the same event, so a workflow restart cannot reset the
budget. An unambiguous missing/ambiguous Spot symbol is terminal immediately as
`SR_SYMBOL_UNAVAILABLE`. The S/R gate is therefore fail-closed for each candidate rather
than silently bypassed. The Spot provider uses the same Binance WireProxy as the Futures
market-data path in the GitHub Actions workflow.

`AJAY_SR_ROOM_MODE=enforce` is the shipped VST policy. The new room rule can be moved
back to `shadow` without changing event detectors or execution code.

## Future futures feed

If/when Futures becomes the preferred SR source, add a validated Futures kline provider
behind the same `get_cached_sr_snapshot()` interface. Do not change the room/gate rules
at the same time as changing the market feed; validate the feed first, then compare the
TradingView levels, then run shadow, then enforce.

## Post-fill safety

When SR enforcement is active, the same closed-1H snapshot is evaluated again after the
confirmed BingX average fill and before protective orders are installed. If the worse fill
makes TP1 unreachable, the candidate is flattened and terminalized. An opposing zone that
starts after TP1 remains allowed, and the existing TP ladder is preserved without capping TP3.
Shadow mode remains observational and never mutates the TP ladder.
## Regression-audit note
The final release also isolates the tracker SL-fill regression test from the live BingX price endpoint; tests must remain deterministic and read-only.

## v10.3 fixed stop policy

All new entries use a deterministic fixed stop-loss distance of 7.00% from the actual
entry reference. ATR remains available only as diagnostic telemetry; it no longer
determines the new-entry stop and does not clip/reject the 7% policy. Orphan-position
reconciliation also falls back to the same 7% stop when no historical trade profile exists.

The 7% stop means the current normal target ladder is 5.25% / 8.75% / 14.00%
(0.75R / 1.25R / 2.00R) and squeeze targets are 7% / 10.50% / 14.00%
(1R / 1.5R / 2R). The partial-exit fractions are unchanged.

## Cross-exchange drift retry guard

A `CROSS_EXCHANGE_DRIFT_EXCEEDED` pre-order rejection is retryable for up to 3
rejections of the same event. After the third rejection the event is terminalized
with `CROSS_EXCHANGE_DRIFT_EXHAUSTED`, and the persisted `EXECUTION_ATTEMPT` history
prevents a workflow restart from resetting that budget.

## v10.3.8 final audit and execution-notification correction

A quantity-prefixed perpetual such as `1000SHIBUSDT` uses the numeric prefix as a contract quantity multiplier.
For the Binance 1000-contract index, the corresponding quoted price is 1,000x the unprefixed Spot pair;
therefore a Spot `SHIBUSDT` source is converted with `price_scale=1000` and `volume_scale=0.001` so the S/R snapshot is expressed in the logical 1000SHIB unit.

The Binance Futures client refreshes its live symbol catalog once and retries after an HTTP 400 that can indicate a stale symbol mapping.

The VST workflow removes test-generated runtime/cache state before the unconditional state-commit step, preventing future CI test artifacts from entering persistent runtime telemetry.

A BingX contract that disappears or becomes unavailable after the candidate preflight is terminalized as `BINGX_CONTRACT_UNAVAILABLE` rather than retried blindly.


## v10.3.8 TP and Telegram policy

New entries use the TP ladder defined centrally in `run_once.py`: normal 0.65R/1.25R/2.00R
and squeeze 1.00R/1.50R/2.00R, with the configured partial-exit fractions preserved. Protection
adapts after the confirmed fill: use 3 TP legs when the actual quantity is executable, otherwise
fall back to TP1/TP2, and finally to a single terminal TP3 when only one leg is supported.
Restart/reconciliation fallback uses the same current ladder only when no persisted TP profile exists;
persisted historical TP profiles are preserved.
BE remains an execution/risk-management mechanism, but the successful `BE_ACTIVATED` transition is no longer
queued for Telegram. Local `[TRACKER_BE_ACTIVATED]` logging and `BE_FAILED` Telegram alerts remain intact.

## v10.4.0 runtime-scan audit note

The 2026-10-02 second run showed a full 1H+4H universe rescan again. With the
production `BAR_CLOSE_GRACE_MIN=2`, the 13:02 and 13:12 UTC cycles should map to
the same completed 1H/4H buckets. Therefore a second full rescan proves that the
persisted per-symbol `timeframe_scan_state.json` was not available/matched at the
start of that run; the engine did not intentionally need to rescan those buckets.

Additional same-cycle optimization: when a 1H/4H frame has already been fetched
for event detection in the current cycle, its raw candles are reused for the
same-cycle risk/HTF checks instead of issuing a duplicate Binance kline request.
This does not change detector logic or closed-bar selection.

Runtime telemetry: event `Age` is measured from the event's `detected_at_ts`, which
for the latest-bar strategy engines is the close time of the same latest completed
candle. Multiple events derived from that candle therefore legitimately share the
same age; subsequent cycles should show that age increasing until the event expires.


## v10.4.1 scheduler persistence hardening

The workflow now force-refreshes `origin/main` before tests and restores the remote
`data/timeframe_scan_state.json` into the runner workspace before the engine starts.
The engine emits explicit missing/invalid-state diagnostics instead of silently
turning every persistence failure into `symbols=0`.

The runtime-state commit step now validates the scheduler state schema before staging,
force-stages the state file, pushes with retry/rebase handling, then fetches `origin/main`
and verifies that the pushed scheduler-state hash and symbol count match the local state.
This keeps per-symbol 1H/4H watermarks persistent without changing detector or S/R logic.
