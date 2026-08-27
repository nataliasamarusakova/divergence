# coinalyze.py

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

log = logging.getLogger("event_engine.coinalyze")

COINALYZE_URL = os.environ.get("COINALYZE_URL", "https://coinalyze.net/").strip()
COINALYZE_P_SID = os.environ.get("COINALYZE_P_SID", "").strip()
COINALYZE_CHAT_SID = os.environ.get("COINALYZE_CHAT_SID", "").strip()
MAX_PAGES = int(os.environ.get("MAX_PAGES", "20"))
DEBUG_HTML_FILE = os.environ.get("DEBUG_HTML_FILE", "debug_page.html")


@dataclass(frozen=True)
class CoinalyzeRow:
    symbol: str
    name: str
    price: float | None
    price_chg24: float | None
    volume24: float | None
    oi: float | None
    oi_chg24_pct: float | None
    oi_chg4h_pct: float | None
    oi_vol_ratio: float | None
    oi_mktcap_ratio: float | None
    fr_oiw: float | None
    pfr_oiw: float | None
    liq_short24: float | None
    liq_long24: float | None
    ls_accounts: float | None
    btc_corr7d: float | None
    cvd24: float | None
    lls24: float | None
    raw: dict[str, Any]


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    if not s or s.lower() in {"nan", "none", "-", "—", "n/a"}:
        return None
    mult = 1.0
    if s[-1:].upper() in {"K", "M", "B", "T"}:
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}[s[-1].upper()]
        s = s[:-1].strip()
    try:
        return float(s) * mult
    except ValueError:
        return None


def _setup_browser_context(p):
    """Create a Coinalyze browser context compatible with playwright-stealth 2.x and 1.x."""
    browser = p.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage", "--no-sandbox"],
    )
    ctx = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/150.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="Europe/Berlin",
    )

    cookies = []
    if COINALYZE_P_SID:
        cookies.append({
            "name": "p_sid",
            "value": COINALYZE_P_SID,
            "domain": "coinalyze.net",
            "path": "/",
            "secure": True,
        })
    if COINALYZE_CHAT_SID:
        cookies.append({
            "name": "chat_sid",
            "value": COINALYZE_CHAT_SID,
            "domain": "coinalyze.net",
            "path": "/",
            "secure": True,
        })
    cookies.append({
        "name": "cookies_accepted",
        "value": "1",
        "domain": "coinalyze.net",
        "path": "/",
        "secure": True,
    })
    if cookies:
        ctx.add_cookies(cookies)

    page = ctx.new_page()

    # playwright-stealth >=2 uses Stealth().apply_stealth_sync(page),
    # whereas old 1.x exposed stealth_sync(page).
    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
    except (ImportError, AttributeError):
        try:
            from playwright_stealth import stealth_sync
        except ImportError as exc:
            raise RuntimeError(
                "playwright-stealth is installed but exposes neither "
                "Stealth.apply_stealth_sync nor stealth_sync."
            ) from exc
        stealth_sync(page)

    return browser, page


def _load_page(page, url: str) -> str:
    page.goto(url, wait_until="networkidle", timeout=60_000)
    page.wait_for_timeout(3_000)
    content = page.content()
    if "Attention Required" in content:
        log.warning("Cloudflare challenge detected, waiting")
        page.wait_for_timeout(10_000)
        page.wait_for_load_state("networkidle", timeout=30_000)
    page.wait_for_selector("tbody tr", timeout=25_000)
    prev = len(page.query_selector_all("tbody tr"))
    for attempt in range(15):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(700)
        cur = len(page.query_selector_all("tbody tr"))
        if page.query_selector(".pagination") is not None:
            log.info("Coinalyze pagination detected after scroll %d; rows=%d", attempt + 1, cur)
            break
        if cur != prev:
            log.info("Coinalyze scroll %d: rows %d -> %d", attempt + 1, prev, cur)
        if cur == prev and attempt >= 3:
            break
        prev = cur
    try:
        page.wait_for_load_state("networkidle", timeout=10_000)
    except Exception:
        pass
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(500)
    final_count = len(page.query_selector_all("tbody tr"))
    log.info("Coinalyze page rows=%d pagination=%s", final_count, bool(page.query_selector(".pagination")))
    return page.content()


def click_next_page(page, current_page_num: int) -> bool:
    pag = page.query_selector(".pagination")
    if not pag:
        return False
    first_row = page.query_selector("tbody tr")
    before = first_row.get_attribute("data-coin") if first_row else None
    target = None
    for el in pag.query_selector_all("a, button, li"):
        if (el.inner_text() or "").strip() == str(current_page_num + 1):
            target = el
            break
    if target is None:
        target = pag.query_selector("[aria-label='Next'], .next, a[rel='next']")
    if target is None:
        return False
    target.click()
    page.wait_for_timeout(1500)
    for _ in range(10):
        row = page.query_selector("tbody tr")
        after = row.get_attribute("data-coin") if row else None
        if after and after != before:
            return True
        page.wait_for_timeout(500)
    return False


def get_page_urls(html_text: str) -> list[str]:
    soup = BeautifulSoup(html_text, "lxml")
    pagination = soup.select_one(".pagination")
    if not pagination:
        return [COINALYZE_URL]
    urls = [COINALYZE_URL]
    for a in pagination.select("a[href]"):
        href = a.get("href", "")
        if not href:
            continue
        full = f"https://coinalyze.net{href}" if href.startswith("/") else href
        if full not in urls:
            urls.append(full)
    return urls[:MAX_PAGES]


def parse_table(html_text: str) -> list[CoinalyzeRow]:
    soup = BeautifulSoup(html_text, "lxml")
    rows = soup.select("tbody tr")
    out: list[CoinalyzeRow] = []

    # Map column headers dynamically if thead exists, otherwise fallback to positional indices
    header_map = {}
    headers = [th.get_text(" ", strip=True).lower() for th in soup.select("thead th")]
    for idx, h in enumerate(headers):
        if "price" in h and "chg" not in h:
            header_map["price"] = idx
        elif "24h chg" in h or "price chg" in h or "24h %" in h:
            header_map["price_chg24"] = idx
        elif "market cap" in h or "mkt cap" in h:
            header_map["mktcap"] = idx
        elif "24h vol" in h or "volume" in h:
            header_map["volume24"] = idx
        elif "open int" in h or h == "oi":
            header_map["oi"] = idx
        elif "24h oi" in h or "oi chg 24h" in h:
            header_map["oi_chg24_pct"] = idx
        elif "4h oi" in h or "oi chg 4h" in h:
            header_map["oi_chg4h_pct"] = idx
        elif "oi / vol" in h:
            header_map["oi_vol_ratio"] = idx
        elif "oi / mkt" in h:
            header_map["oi_mktcap_ratio"] = idx
        elif "fr" in h and "pred" not in h:
            header_map["fr_oiw"] = idx
        elif "pred fr" in h or "pfr" in h:
            header_map["pfr_oiw"] = idx
        elif "liq short" in h:
            header_map["liq_short24"] = idx
        elif "liq long" in h:
            header_map["liq_long24"] = idx
        elif "l/s" in h or "accounts" in h:
            header_map["ls_accounts"] = idx
        elif "btc corr" in h:
            header_map["btc_corr7d"] = idx
        elif "cvd" in h:
            header_map["cvd24"] = idx
        elif "lls" in h:
            header_map["lls24"] = idx

    def _get_val(tds: list, field_name: str, fallback_idx: int) -> float | None:
        idx = header_map.get(field_name, fallback_idx)
        if idx < len(tds):
            return parse_number(tds[idx].get_text(" ", strip=True))
        return None

    for tr in rows:
        symbol = (tr.get("data-coin") or "").strip().upper()
        tds = tr.find_all("td")
        if not symbol or len(tds) < 18:
            continue
        spans = tds[1].find_all("span") if len(tds) > 1 else []
        name = spans[0].get_text(strip=True) if spans else symbol

        raw = {
            "price": _get_val(tds, "price", 2),
            "price_chg24": _get_val(tds, "price_chg24", 3),
            "mktcap": _get_val(tds, "mktcap", 4),
            "volume24": _get_val(tds, "volume24", 5),
            "oi": _get_val(tds, "oi", 6),
            "oi_chg24_pct": _get_val(tds, "oi_chg24_pct", 7),
            "oi_chg4h_pct": _get_val(tds, "oi_chg4h_pct", 9),
            "oi_vol_ratio": _get_val(tds, "oi_vol_ratio", 11),
            "oi_mktcap_ratio": _get_val(tds, "oi_mktcap_ratio", 12),
            "fr_oiw": _get_val(tds, "fr_oiw", 15),
            "pfr_oiw": _get_val(tds, "pfr_oiw", 16),
            "liq_short24": _get_val(tds, "liq_short24", 17),
            "liq_long24": _get_val(tds, "liq_long24", 18),
            "ls_accounts": _get_val(tds, "ls_accounts", 19),
            "btc_corr7d": _get_val(tds, "btc_corr7d", 20),
            "cvd24": _get_val(tds, "cvd24", 21),
            "lls24": _get_val(tds, "lls24", 22),
        }

        out.append(CoinalyzeRow(
            symbol=symbol,
            name=name,
            price=raw["price"],
            price_chg24=raw["price_chg24"],
            volume24=raw["volume24"],
            oi=raw["oi"],
            oi_chg24_pct=raw["oi_chg24_pct"],
            oi_chg4h_pct=raw["oi_chg4h_pct"],
            oi_vol_ratio=raw["oi_vol_ratio"],
            oi_mktcap_ratio=raw["oi_mktcap_ratio"],
            fr_oiw=raw["fr_oiw"],
            pfr_oiw=raw["pfr_oiw"],
            liq_short24=raw["liq_short24"],
            liq_long24=raw["liq_long24"],
            ls_accounts=raw["ls_accounts"],
            btc_corr7d=raw["btc_corr7d"],
            cvd24=raw["cvd24"],
            lls24=raw["lls24"],
            raw=raw,
        ))
    return out


def fetch_data() -> list[CoinalyzeRow]:
    all_rows: list[CoinalyzeRow] = []
    seen: set[str] = set()
    page_errors = []
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser, page = _setup_browser_context(p)
        try:
            html = _load_page(page, COINALYZE_URL)
            try:
                with open(DEBUG_HTML_FILE, "w", encoding="utf-8") as f:
                    f.write(html)
            except OSError:
                pass
            rows = parse_table(html)
            for r in rows:
                if r.symbol not in seen:
                    all_rows.append(r); seen.add(r.symbol)
            log.info("Coinalyze page 1: %d rows", len(rows))
            page_urls = get_page_urls(html)
            log.info("Coinalyze pagination: %d pages", len(page_urls))
            if len(page_urls) > 1:
                for i, url in enumerate(page_urls[1:], start=2):
                    try:
                        html = _load_page(page, url)
                        rows = parse_table(html)
                        added = 0
                        for r in rows:
                            if r.symbol not in seen:
                                all_rows.append(r); seen.add(r.symbol); added += 1
                        log.info("Coinalyze page %d: +%d new rows", i, added)
                    except Exception as exc:
                        page_errors.append({"page": i, "url": url, "error": str(exc)[:200]})
                        log.exception("Coinalyze page %d failed", i)
            elif page.query_selector(".pagination") is not None:
                page_num = 1
                while page_num < MAX_PAGES and click_next_page(page, page_num):
                    page.wait_for_selector("tbody tr", timeout=15_000)
                    page.wait_for_timeout(500)
                    html = page.content()
                    rows = parse_table(html)
                    added = 0
                    for r in rows:
                        if r.symbol not in seen:
                            all_rows.append(r); seen.add(r.symbol); added += 1
                    page_num += 1
                    log.info("Coinalyze click page %d: +%d new rows", page_num, added)
        finally:
            browser.close()
    if page_errors:
        log.warning("Coinalyze scrape incomplete: %d page errors", len(page_errors))
    log.info("Coinalyze total unique rows=%d", len(all_rows))
    return all_rows
