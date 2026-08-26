from __future__ import annotations
import os
import requests

TOKEN=os.environ.get("TG_BOT_TOKEN","").strip()
IDS=[x.strip() for x in os.environ.get("TG_CHAT_IDS","").replace(";",",").split(",") if x.strip()]

def send(text: str) -> bool:
    if not TOKEN or not IDS: return False
    ok=True
    url=f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for chat_id in IDS:
        try:
            r=requests.post(url,data={"chat_id":chat_id,"text":text,"parse_mode":"HTML","disable_web_page_preview":True},timeout=15)
            ok = ok and r.ok
        except Exception:
            ok=False
    return ok
