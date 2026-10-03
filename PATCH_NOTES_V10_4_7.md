# v10.4.7 safe infrastructure patch

Base: `divergence-main_v10.4.6_SIGNAL_RECOVERY_FIXED(1).zip`

## Changed
- Trigger stale age is measured from the actual closed 15M `trigger_bar_close_ts`; `trigger_observed_at_ts` remains telemetry only.
- First-seen symbol/timeframe replay is bounded to a recent contiguous window (`NEW_SYMBOL_BACKFILL_MIN`, default 120 minutes).
- Same-direction opportunity deduplication prefers the newest event; score only breaks exact timestamp ties.
- Historical HTF validation receives only candles that had closed by the event timestamp, preventing future HTF leakage.
- Execution telemetry no longer fabricates `actual_entry_price` from the requested signal price when no confirmed fill exists.

## Intentionally unchanged
- All detector/event formulas and event construction in `event_engine/signals.py`.
- S/R geometry and evaluation semantics in `event_engine/sr_context.py`.
- Trade lifecycle/execution implementation in `event_engine/tracker.py` and `event_engine/bingx.py`.
- Entry-quality shadow/enforcement settings.
- Portfolio/per-cycle research limits.
- Existing TP/SL configuration.

## Verification
- `pytest -q`: 369 passed.
- `python -m compileall -q .`: OK.
- SHA-256 of `event_engine/signals.py`, `sr_context.py`, `tracker.py`, `bingx.py` is unchanged from the source archive.
- Runtime `data/` is intentionally excluded from the clean release archive.
## v10.4.8 presentation/telemetry patch

- User-facing numeric formatting is normalized for prices, SL/TP values, percentages and per-leg R multipliers; raw floating-point artifacts are no longer shown in Telegram/tracker messages.
- Open-position Telegram setup now shows TP1/TP2/TP3 as absolute prices with favorable percentage, per-leg R and configured close fraction.
- Removed the user-facing `R:R (Effective Weighted)` and `TP Mode` fields from the open-position Telegram card. Internal persisted fields remain unchanged because they are required for lifecycle/reconciliation.
- Added `CONFIRMATION` field: `2/2` when the event direction is aligned with an in-zone supporting Demand/Supply zone (`LONG + DEMAND` or `SHORT + SUPPLY`); otherwise `1/2` with the reason. This is presentation-only and does not gate entries.
- Existing detector/event, S/R geometry, protection and execution calculations are unchanged.
