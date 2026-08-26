from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import (
    refresh_contracts,
    get_contract,
    fetch_klines,
    open_market,
)
from event_engine.signals import (
    add_cvd,
    detect_divergences,
    detect_squeeze_release,
    build_15m_trigger,
)
from event_engine.telegram import send as send_tg, format_signal


logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA = Path("data")
DATA.mkdir(exist_ok=True)

EVENTS = DATA / "events.jsonl"
TRADES = DATA / "trades.jsonl"
ACTIONS = DATA / "actions.jsonl"

MAX_CANDIDATES = int(
    os.environ.get("MAX_CANDIDATES", "40")
)

MIN_VOL = float(
    os.environ.get("MIN_VOLUME_24H", "1000000")
)

MIN_OI = float(
    os.environ.get("MIN_OPEN_INTEREST", "500000")
)

EXECUTION_ENABLED = (
    os.environ.get("EXECUTION_ENABLED", "false").lower()
    == "true"
)

REQUIRE_CVD = (
    os.environ.get("REQUIRE_CVD_CONFIRMATION", "false").lower()
    == "true"
)

REQUIRE_TRIGGER = (
    os.environ.get("REQUIRE_15M_TRIGGER", "true").lower()
    == "true"
)

# 1H event -> 15m execution trigger.
# 90m is intentionally used instead of the old 45m.
MAX_AGE = int(
    os.environ.get("MAX_EVENT_AGE_MIN", "90")
)

MAX_TRADES = int(
    os.environ.get("MAX_TRADES_PER_CYCLE", "1")
)


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        try:
            value = json.loads(line).get(
                "event_id"
            )

            if value:
                ids.add(value)

        except Exception:
            continue

    return ids


def append_jsonl(
    path: Path,
    obj: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
    ) as f:

        f.write(
            json.dumps(
                obj,
                ensure_ascii=False,
            )
            + "\n"
        )


def emit_event(ev: dict) -> None:
    append_jsonl(EVENTS, ev)


def record_trade(obj: dict) -> None:
    append_jsonl(TRADES, obj)


def record_action(obj: dict) -> None:
    append_jsonl(ACTIONS, obj)


def _event_label(direction: str) -> str:
    if str(direction).upper() == "LONG":
        return "🚨 LONG SIGNAL"

    return "🔻 SHORT SIGNAL"


def _safe_price(value) -> str:
    try:
        return f"{float(value):.8g}"
    except Exception:
        return "n/a"


def _print_trigger_debug(
    symbol: str,
    direction: str,
    df15: pd.DataFrame,
) -> None:

    try:
        if len(df15) < 2:
            print(
                f"[TRIGGER_DEBUG] "
                f"symbol={symbol} "
                f"direction={direction} "
                f"15m_bars={len(df15)} < 2"
            )
            return

        prev_bar = df15.iloc[-2]
        last_bar = df15.iloc[-1]

        print(
            f"[TRIGGER_DEBUG] "
            f"symbol={symbol} "
            f"direction={direction} "
            f"prev_high={_safe_price(prev_bar.get('high'))} "
            f"prev_low={_safe_price(prev_bar.get('low'))} "
            f"last_open={_safe_price(last_bar.get('open'))} "
            f"last_high={_safe_price(last_bar.get('high'))} "
            f"last_low={_safe_price(last_bar.get('low'))} "
            f"last_close={_safe_price(last_bar.get('close'))}"
        )

        if "close_time" in df15.columns:
            try:
                print(
                    f"[TRIGGER_TIME] "
                    f"symbol={symbol} "
                    f"prev_close_time={int(prev_bar['close_time'])} "
                    f"last_close_time={int(last_bar['close_time'])}"
                )
            except Exception:
                pass

    except Exception as exc:
        print(
            f"[TRIGGER_DEBUG_ERROR] "
            f"symbol={symbol} "
            f"error={exc}"
        )


def main() -> None:

    stats = {
        "coinalyze_rows": 0,
        "candidates_before_limit": 0,
        "candidates": 0,
        "bingx_mapped": 0,
        "bingx_unmapped": 0,
        "klines_1h_ok": 0,
        "klines_15m_ok": 0,
        "rsi_events": 0,
        "cvd_events": 0,
        "squeeze_events": 0,
        "events_total": 0,
        "events_recent": 0,
        "events_duplicate": 0,
        "events_cvd_gate_rejected": 0,
        "trigger_pass": 0,
        "trigger_rejected": 0,
        "setups": 0,
        "execution_attempts": 0,
        "trades": 0,
        "telegram_attempts": 0,
        "telegram_sent": 0,
        "scan_errors": 0,
    }

    # ==============================================================
    # 1. COINALYZE DISCOVERY
    # ==============================================================

    rows = fetch_data()

    stats["coinalyze_rows"] = len(rows)

    print(
        f"[ENGINE] Coinalyze rows={len(rows)}"
    )

    # ==============================================================
    # 2. BINGX CONTRACTS
    # ==============================================================

    try:

        refresh_contracts()

    except Exception as exc:

        stats["scan_errors"] += 1

        print(
            f"[BINGX] contracts refresh error={exc}"
        )

    # ==============================================================
    # 3. CANDIDATE BUILDING + BINGX MAPPING
    # ==============================================================

    candidates = []

    for r in rows:

        if r.price is None or r.price <= 0:
            continue

        if (
            r.volume24 is None
            or r.volume24 < MIN_VOL
        ):
            continue

        if (
            r.oi is None
            or r.oi < MIN_OI
        ):
            continue

        contract = get_contract(
            r.symbol
        )

        if not contract:

            stats["bingx_unmapped"] += 1

            print(
                f"[MAP] NO_BINGX "
                f"coinalyze={r.symbol} "
                f"name={getattr(r, 'name', '')}"
            )

            continue

        stats["bingx_mapped"] += 1

        print(
            f"[MAP] OK "
            f"coinalyze={r.symbol} "
            f"bingx={contract.get('symbol', r.symbol)} "
            f"displayName={contract.get('displayName', '')}"
        )

        candidates.append(r)

    stats["candidates_before_limit"] = len(
        candidates
    )

    candidates = candidates[
        :MAX_CANDIDATES
    ]

    stats["candidates"] = len(
        candidates
    )

    print(
        f"[ENGINE] Coinalyze candidates={len(candidates)} "
        f"execution={EXECUTION_ENABLED} "
        f"env={os.environ.get('EXECUTION_MODE', os.environ.get('BINGX_ENV', 'vst'))}"
    )

    # ==============================================================
    # 4. EVENT / EXECUTION STATE
    # ==============================================================

    seen_events = load_ids(
        EVENTS
    )

    executed_event_ids = load_ids(
        TRADES
    )

    trades = 0

    # ==============================================================
    # 5. PROCESS CANDIDATES
    # ==============================================================

    for r in candidates:

        symbol = r.symbol

        try:

            # ------------------------------------------------------
            # 5.1 OHLCV
            # ------------------------------------------------------

            k1 = fetch_klines(
                symbol,
                "1h",
                int(
                    os.environ.get(
                        "KLINE_LIMIT_1H",
                        "250",
                    )
                ),
            )

            k15 = fetch_klines(
                symbol,
                "15m",
                int(
                    os.environ.get(
                        "KLINE_LIMIT_15M",
                        "250",
                    )
                ),
            )

            if len(k1) < 60:

                print(
                    f"[DATA_REJECT] "
                    f"{symbol} "
                    f"1H bars={len(k1)} < 60"
                )

                continue

            if len(k15) < 10:

                print(
                    f"[DATA_REJECT] "
                    f"{symbol} "
                    f"15m bars={len(k15)} < 10"
                )

                continue

            stats["klines_1h_ok"] += 1
            stats["klines_15m_ok"] += 1

            # ------------------------------------------------------
            # 5.2 CVD
            # ------------------------------------------------------

            d1 = add_cvd(
                pd.DataFrame(k1)
            )

            # ------------------------------------------------------
            # 5.3 1H DIVERGENCES
            # ------------------------------------------------------

            rsi_events_all = detect_divergences(
                d1,
                symbol,
                "1h",
            )

            # ------------------------------------------------------
            # 5.4 1H SQUEEZE
            # ------------------------------------------------------

            squeeze_events = (
                detect_squeeze_release(
                    d1,
                    symbol,
                    "1h",
                )
            )

            rsi_events = [
                e
                for e in rsi_events_all
                if "_RSI" in e.get(
                    "event_type",
                    "",
                )
            ]

            cvd_events = [
                e
                for e in rsi_events_all
                if "BINGX_CVD" in e.get(
                    "event_type",
                    "",
                )
            ]

            stats["rsi_events"] += len(
                rsi_events
            )

            stats["cvd_events"] += len(
                cvd_events
            )

            stats["squeeze_events"] += len(
                squeeze_events
            )

            stats["events_total"] += (
                len(rsi_events_all)
                + len(squeeze_events)
            )

            print(
                f"[EVENT_SCAN] "
                f"{symbol} "
                f"RSI={len(rsi_events)} "
                f"CVD={len(cvd_events)} "
                f"SQUEEZE={len(squeeze_events)}"
            )

            all_events = (
                rsi_events_all
                + squeeze_events
            )

            # ------------------------------------------------------
            # 5.5 EVENT PROCESSING
            # ------------------------------------------------------

            for ev in all_events:

                event_id = ev.get(
                    "event_id"
                )

                if not event_id:

                    print(
                        f"[EVENT_REJECT] "
                        f"{symbol} "
                        f"event without event_id"
                    )

                    continue

                detected_at = int(
                    ev.get(
                        "timestamps",
                        {},
                    ).get(
                        "detected_at_ts",
                        0,
                    )
                )

                latest_close = int(
                    d1.close_time.iloc[-1]
                )

                age = (
                    latest_close
                    - detected_at
                ) / 60000.0

                # --------------------------------------------------
                # EVENT AGE
                # --------------------------------------------------

                if (
                    age < 0
                    or age > MAX_AGE
                ):

                    print(
                        f"[EVENT_REJECT_AGE] "
                        f"{symbol} "
                        f"type={ev.get('event_type')} "
                        f"age={age:.1f}m "
                        f"max={MAX_AGE}m"
                    )

                    continue

                stats["events_recent"] += 1

                # --------------------------------------------------
                # PERSIST EVENT
                # --------------------------------------------------

                if event_id in seen_events:

                    stats["events_duplicate"] += 1

                else:

                    emit_event(
                        ev
                    )

                    seen_events.add(
                        event_id
                    )

                # --------------------------------------------------
                # OPTIONAL CVD CONFIRMATION
                # --------------------------------------------------

                if (
                    REQUIRE_CVD
                    and "_RSI"
                    in ev.get(
                        "event_type",
                        "",
                    )
                ):

                    matched_cvd = any(
                        other.get(
                            "direction"
                        )
                        == ev.get(
                            "direction"
                        )
                        and other.get(
                            "timestamps",
                            {},
                        ).get(
                            "pivot_1_ts"
                        )
                        == ev.get(
                            "timestamps",
                            {},
                        ).get(
                            "pivot_1_ts"
                        )
                        and other.get(
                            "timestamps",
                            {},
                        ).get(
                            "pivot_2_ts"
                        )
                        == ev.get(
                            "timestamps",
                            {},
                        ).get(
                            "pivot_2_ts"
                        )
                        for other
                        in cvd_events
                    )

                    if not matched_cvd:

                        stats[
                            "events_cvd_gate_rejected"
                        ] += 1

                        print(
                            f"[EVENT_REJECT_CVD] "
                            f"{symbol} "
                            f"type={ev.get('event_type')}"
                        )

                        continue

                # --------------------------------------------------
                # 15M TRIGGER
                # --------------------------------------------------

                df15 = pd.DataFrame(
                    k15
                )

                trigger = build_15m_trigger(
                    df15,
                    ev["direction"],
                )

                if (
                    REQUIRE_TRIGGER
                    and not trigger
                ):

                    stats[
                        "trigger_rejected"
                    ] += 1

                    print(
                        f"[TRIGGER_REJECT] "
                        f"symbol={symbol} "
                        f"direction={ev.get('direction')} "
                        f"event={ev.get('event_type')} "
                        f"reason=15m_trigger_failed"
                    )

                    _print_trigger_debug(
                        symbol=symbol,
                        direction=ev.get(
                            "direction",
                            "",
                        ),
                        df15=df15,
                    )

                    continue

                stats[
                    "trigger_pass"
                ] += 1

                print(
                    f"[TRIGGER_PASS] "
                    f"symbol={symbol} "
                    f"direction={ev.get('direction')} "
                    f"event={ev.get('event_type')}"
                )

                # --------------------------------------------------
                # REFERENCE ENTRY PRICE
                # --------------------------------------------------

                try:

                    price = float(
                        ev[
                            "event_fact"
                        ][
                            "detection_close_price"
                        ]
                    )

                except Exception:

                    print(
                        f"[EVENT_REJECT] "
                        f"{symbol} "
                        f"missing detection_close_price"
                    )

                    continue

                # --------------------------------------------------
                # SETUP
                # --------------------------------------------------

                setup = {
                    "entry_reference": price,
                    "invalidation_price": None,
                    "target_price": None,
                    "rr": None,
                    "trigger_ok": True,
                }

                stats[
                    "setups"
                ] += 1

                # --------------------------------------------------
                # EXECUTION
                # --------------------------------------------------

                execution_result = None

                if event_id in executed_event_ids:

                    execution_result = {
                        "status": "ALREADY_EXECUTED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                    }

                    print(
                        f"[EXECUTION_SKIP_DUPLICATE] "
                        f"{symbol} "
                        f"event={event_id}"
                    )

                elif (
                    EXECUTION_ENABLED
                    and trades < MAX_TRADES
                ):

                    stats[
                        "execution_attempts"
                    ] += 1

                    trade_id = (
                        event_id.replace(
                            "EVT_",
                            "",
                        )
                    )

                    execution_result = open_market(
                        symbol,
                        ev["direction"],
                        price,
                        trade_id,
                    )

                    record_trade(
                        {
                            "event_id": event_id,
                            "symbol": symbol,
                            "direction": ev[
                                "direction"
                            ],
                            "price": price,
                            "event_type": ev.get(
                                "event_type"
                            ),
                            "ts": int(
                                pd.Timestamp.utcnow().timestamp()
                                * 1000
                            ),
                            "result": execution_result,
                        }
                    )

                    # IMPORTANT:
                    # Mark event as processed after order attempt
                    # so the same event isn't repeatedly submitted.
                    executed_event_ids.add(
                        event_id
                    )

                    if (
                        execution_result.get(
                            "status"
                        )
                        == "opened"
                    ):

                        trades += 1

                        stats[
                            "trades"
                        ] += 1

                        print(
                            f"[EXECUTION_OPENED] "
                            f"{symbol} "
                            f"direction={ev['direction']} "
                            f"order={execution_result.get('order_id')}"
                        )

                    else:

                        print(
                            f"[EXECUTION_RESULT] "
                            f"{symbol} "
                            f"status={execution_result.get('status')} "
                            f"message={execution_result.get('msg')}"
                        )

                elif not EXECUTION_ENABLED:

                    execution_result = {
                        "status": "DISABLED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                    }

                else:

                    execution_result = {
                        "status": "TRADE_LIMIT_REACHED",
                        "mode": os.environ.get(
                            "EXECUTION_MODE",
                            "vst",
                        ),
                        "order_id": None,
                    }

                # --------------------------------------------------
                # TELEGRAM
                # --------------------------------------------------

                label = _event_label(
                    ev["direction"]
                )

                try:

                    msg = format_signal(
                        ev,
                        setup=setup,
                        coinalyze_row=r,
                        execution=execution_result,
                    )

                except Exception as exc:

                    status = (
                        execution_result.get(
                            "status"
                        )
                        if execution_result
                        else "NOT_ATTEMPTED"
                    )

                    msg = (
                        f"{label}\n"
                        f"<b>"
                        f"{getattr(r, 'name', None) or symbol}"
                        f"</b> "
                        f"(<code>{symbol}</code>)\n"
                        f"Event: "
                        f"<code>{ev.get('event_type')}</code>\n"
                        f"TF: "
                        f"1H + trigger 15m\n"
                        f"Price: "
                        f"<code>{price:.8g}</code>\n"
                        f"Execution: "
                        f"<code>{status}</code>"
                    )

                    print(
                        f"[TELEGRAM_FORMAT_FALLBACK] "
                        f"{symbol}: {exc}"
                    )

                stats[
                    "telegram_attempts"
                ] += 1

                try:

                    sent = send_tg(
                        msg
                    )

                except Exception as exc:

                    sent = False

                    print(
                        f"[TELEGRAM_ERROR] "
                        f"{symbol}: {exc}"
                    )

                if sent:

                    stats[
                        "telegram_sent"
                    ] += 1

                    print(
                        f"[TELEGRAM_SENT] "
                        f"{symbol} "
                        f"direction={ev['direction']}"
                    )

                record_action(
                    {
                        "event_id": event_id,
                        "symbol": symbol,
                        "direction": ev[
                            "direction"
                        ],
                        "event_type": ev.get(
                            "event_type"
                        ),
                        "telegram_sent": bool(
                            sent
                        ),
                        "execution_status": (
                            execution_result.get(
                                "status"
                            )
                            if execution_result
                            else None
                        ),
                        "ts": int(
                            pd.Timestamp.utcnow().timestamp()
                            * 1000
                        ),
                    }
                )

        except Exception as exc:

            stats[
                "scan_errors"
            ] += 1

            print(
                f"[SCAN_ERROR] "
                f"{symbol}: {exc}"
            )

    # ==============================================================
    # 6. SUMMARY
    # ==============================================================

    print(
        f"[ENGINE] "
        f"trades_this_cycle={trades}"
    )

    print(
        "[ENGINE_SUMMARY] "
        + " ".join(
            f"{key}={value}"
            for key, value
            in stats.items()
        )
    )


if __name__ == "__main__":
    main()
