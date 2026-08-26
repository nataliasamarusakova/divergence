from __future__ import annotations
import json, logging, os
from pathlib import Path
import pandas as pd

from event_engine.coinalyze import fetch_data
from event_engine.bingx import refresh_contracts, get_contract, to_bx_symbol, fetch_klines, open_market
from event_engine.signals import add_cvd, detect_divergences, detect_squeeze_release, build_15m_trigger
from event_engine.telegram import send as send_tg

logging.basicConfig(level=logging.INFO, format="%(message)s")
DATA=Path("data"); DATA.mkdir(exist_ok=True)
EVENTS=DATA/"events.jsonl"; TRADES=DATA/"trades.jsonl"
MAX_CANDIDATES=int(os.environ.get("MAX_CANDIDATES","40"))
MIN_VOL=float(os.environ.get("MIN_VOLUME_24H","1000000"))
MIN_OI=float(os.environ.get("MIN_OPEN_INTEREST","500000"))
EXECUTION_ENABLED=os.environ.get("EXECUTION_ENABLED","false").lower()=="true"
REQUIRE_CVD=os.environ.get("REQUIRE_CVD_CONFIRMATION","false").lower()=="true"
REQUIRE_TRIGGER=os.environ.get("REQUIRE_15M_TRIGGER","true").lower()=="true"
MAX_AGE=int(os.environ.get("MAX_EVENT_AGE_MIN","45"))
MAX_TRADES=int(os.environ.get("MAX_TRADES_PER_CYCLE","1"))


def load_ids(path:Path):
    if not path.exists(): return set()
    ids=set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try: ids.add(json.loads(line).get("event_id"))
        except Exception: pass
    return ids


def emit_event(ev):
    with EVENTS.open("a",encoding="utf-8") as f: f.write(json.dumps(ev,ensure_ascii=False)+"\n")


def record_trade(x):
    with TRADES.open("a",encoding="utf-8") as f: f.write(json.dumps(x,ensure_ascii=False)+"\n")


def main():
    rows=fetch_data()
    print(f"[ENGINE] Coinalyze rows={len(rows)}")
    try: refresh_contracts()
    except Exception as exc: print(f"[BINGX] contracts refresh error={exc}")
    candidates=[]
    for r in rows:
        if r.price is None or r.price<=0: continue
        if r.volume24 is None or r.volume24<MIN_VOL: continue
        if r.oi is None or r.oi<MIN_OI: continue
        if not get_contract(r.symbol): continue
        candidates.append(r)
    print(f"[ENGINE] Coinalyze candidates={len(candidates[:MAX_CANDIDATES])} execution={EXECUTION_ENABLED} env={os.environ.get('EXECUTION_MODE',os.environ.get('BINGX_ENV','vst'))}")
    seen=load_ids(EVENTS); trades=0
    for r in candidates[:MAX_CANDIDATES]:
        try:
            k1=fetch_klines(r.symbol,"1h",int(os.environ.get("KLINE_LIMIT_1H","250")))
            k15=fetch_klines(r.symbol,"15m",int(os.environ.get("KLINE_LIMIT_15M","250")))
            if len(k1)<60 or len(k15)<10: continue
            d1=add_cvd(pd.DataFrame(k1))
            e_div=detect_divergences(d1,r.symbol,"1h")
            e_sq=detect_squeeze_release(d1,r.symbol,"1h")
            all_events=e_div+e_sq
            cvd_recent=[e for e in e_div if "BINGX_CVD" in e["event_type"]]
            rsi_recent=[e for e in e_div if "_RSI" in e["event_type"]]
            print(f"[EVENT_SCAN] {r.symbol} RSI={len(rsi_recent)} CVD={len(cvd_recent)} SQUEEZE={len(e_sq)}")
            for ev in all_events:
                age=(int(d1.close_time.iloc[-1])-int(ev['timestamps']['detected_at_ts']))/60000
                if age<0 or age>MAX_AGE: continue
                if ev["event_id"] in seen: continue
                if REQUIRE_CVD and "_RSI" in ev["event_type"] and not cvd_recent: continue
                emit_event(ev); seen.add(ev["event_id"])
                trigger=build_15m_trigger(pd.DataFrame(k15),ev["direction"])
                if REQUIRE_TRIGGER and not trigger: continue
                price=float(ev["event_fact"]["detection_close_price"])
                label="🚨 LONG SIGNAL" if ev["direction"]=="LONG" else "🔻 SHORT SIGNAL"
                reason=ev["event_type"]
                msg=(f"{label}\n<b>{r.name}</b> ({r.symbol})\n"
                     f"Event: <code>{reason}</code>\nTF: 1H + trigger 15m\n"
                     f"Price: <code>{price:.8g}</code>\n"
                     f"Vol24H: <code>{r.volume24:,.0f}</code>\nOI: <code>{r.oi:,.0f}</code>")
                execution_result = None
                if EXECUTION_ENABLED and trades < MAX_TRADES:
                    trade_id = ev["event_id"].replace("EVT_", "")
                    execution_result = open_market(r.symbol, ev["direction"], price, trade_id)
                    record_trade({
                        "event_id": ev["event_id"],
                        "symbol": r.symbol,
                        "direction": ev["direction"],
                        "price": price,
                        "result": execution_result,
                    })
                    if execution_result.get("status") == "opened":
                        trades += 1

                # Telegram is sent after the execution attempt so the message
                # clearly states whether VST actually opened the position.
                try:
                    from event_engine.telegram import format_signal
                    msg = format_signal(
                        ev,
                        setup={
                            "entry_reference": price,
                            "invalidation_price": None,
                            "target_price": None,
                            "rr": None,
                        },
                        coinalyze_row=r,
                        execution=execution_result,
                    )
                except Exception:
                    # Keep the notification path alive even if optional formatting fails.
                    status = execution_result.get("status") if execution_result else "NOT_ATTEMPTED"
                    msg = (
                        f"{label}\n"
                        f"<b>{r.name}</b> ({r.symbol})\n"
                        f"Event: <code>{reason}</code>\n"
                        f"TF: 1H + trigger 15m\n"
                        f"Price: <code>{price:.8g}</code>\n"
                        f"Vol24H: <code>{r.volume24:,.0f}</code>\n"
                        f"OI: <code>{r.oi:,.0f}</code>\n"
                        f"Execution: <code>{status}</code>"
                    )
                send_tg(msg)
        except Exception as exc:
            print(f"[SCAN_ERROR] {r.symbol}: {exc}")
    print(f"[ENGINE] trades_this_cycle={trades}")

if __name__=="__main__": main()
