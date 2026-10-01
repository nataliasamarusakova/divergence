# v10.2 Ajay R5.41 S/R room integration

## Current behavior

The SR layer is deliberately **lazy**: it is not calculated for the whole universe.
It is requested only after a symbol has already survived the existing event, HTF,
15m trigger, funding, entry-quality and conflict checks, and immediately before the
BingX market order path.

The current source is **Binance SPOT 1H** (`https://data-api.binance.vision`) through the
existing Binance WireProxy in the VST workflow. This is a deliberate temporary choice. The signal engine itself still uses Binance Futures
for market data. Before switching the SR source to futures, the same levels must be
compared against the TradingView instrument/feed to ensure the source is correct.

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
- Opposing zone before TP1 or TP2 is rejected.
- Opposing zone between TP2 and TP3 is allowed, but TP3 is capped just before the
  zone with a price/R buffer. This avoids targeting through the opposing zone.
- Opposing zone beyond TP3 leaves the full target ladder unchanged.
- Supporting zones never add arbitrary legacy-score points. They are recorded as
  context only so their real forward value can be measured later.
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
makes TP2 unreachable, the candidate is flattened and terminalized. If the fill remains valid
but only TP3 is blocked, TP3 is capped from the actual fill. Shadow mode remains observational
and never mutates the TP ladder.
## Regression-audit note
The final release also isolates the tracker SL-fill regression test from the live BingX price endpoint; tests must remain deterministic and read-only.
