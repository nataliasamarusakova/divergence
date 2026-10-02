# v10.3 Fixed 7% SL + Ajay R5.41 S/R room integration

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

## Safety / fallback

`AJAY_SR_REQUIRE_DATA=true` in the shipped VST workflow. If the SPOT S/R snapshot cannot be
obtained or is stale, only that candidate is blocked and logged as `SR_DATA_UNAVAILABLE`; the
whole engine is not stopped. The S/R gate is therefore fail-closed for each candidate rather
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

The 7% stop means the normal target ladder is 5.25% / 10.50% / 17.50% (0.75R / 1.50R /
2.50R) and squeeze targets are 7% / 14% / 21% (1R / 2R / 3R).

## Cross-exchange drift retry guard

A `CROSS_EXCHANGE_DRIFT_EXCEEDED` pre-order rejection is retryable for up to 3
rejections of the same event. After the third rejection the event is terminalized
with `CROSS_EXCHANGE_DRIFT_EXHAUSTED`, and the persisted `EXECUTION_ATTEMPT` history
prevents a workflow restart from resetting that budget.
