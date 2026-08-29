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

    s = str(value).strip()
    s = (
        s.replace(",", "")
        .replace("$", "")
        .replace("%", "")
        .replace("\u2212", "-")
        .strip()
    )

    if not s or s.lower() in {"nan", "none", "-", "—", "n/a"}:
        return None

    mult = 1.0
    suffix = s[-1:].upper()

    if suffix in {"K", "M", "B", "T"}:
        mult = {
            "K": 1e3,
            "M": 1e6,
            "B": 1e9,
            "T": 1e12,
        }[suffix]
        s = s[:-1].strip()

    try:
        return float(s) * mult
    except (TypeError, ValueError):
        return None


def _normalize_header(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("&", "and")
    value = value.replace("\xa0", " ")
    value = value.replace("\u2212", "-")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _header_text(th) -> str:
    title_span = th.select_one("span[title]")
    if title_span:
        title = title_span.get("title")
        if title:
            return str(title).strip()
    return th.get_text(" ", strip=True)


def _build_header_map(soup: BeautifulSoup) -> dict[str, int]:
    ths = soup.select("thead tr th")
    if not ths:
        raise ValueError("[COINALYZE] Table schema error: <thead><th> not found")

    normalized_headers: list[tuple[int, str]] = []
    for idx, th in enumerate(ths):
        raw_header = _header_text(th)
        normalized = _normalize_header(raw_header)
        normalized_headers.append((idx, normalized))

    log.info(
        "[COINALYZE] Table headers: %s",
        " | ".join(f"{idx}:{header}" for idx, header in normalized_headers),
    )

    aliases: dict[str, set[str]] = {
        "price": {"price"},
        "price_chg24": {"price change % 24h", "price change 24h", "chg 24h"},
        "mktcap": {"market capitalisation", "market capitalization", "mkt cap"},
        "volume24": {"volume 24h", "vol 24h"},
        "oi": {"open interest"},
        "oi_chg24_pct": {"open interest change % 24h"},
        "oi_chg24_abs": {"open interest change 24h"},
        "oi_chg4h_pct": {"open interest change % 4h"},
        "oi_chg4h_abs": {"open interest change 4h"},
        "oi_vol_ratio": {"open interest / volume 24h", "open interest / volume24h", "oi / vol24h"},
        "oi_mktcap_ratio": {"open interest / market capitalization", "open interest / market capitalisation", "oi / mktcap"},
        "fr_avg": {"funding rate average", "fr avg"},
        "pfr_avg": {"predicted funding rate average", "pfr avg"},
        "fr_oiw": {"funding rate average, oi weighted", "funding rate average, oi_weighted", "fr avg oi_w"},
        "pfr_oiw": {"predicted funding rate average, oi weighted", "predicted funding rate average, oi_weighted", "pfr avg oi_w"},
        "liq_total24": {"liquidations 24h", "liqs. 24h"},
        "liq_short24": {"short liquidations 24h", "short liqs. 24h"},
        "liq_long24": {"long liquidations 24h", "long liqs. 24h"},
        "liq_vol_ratio": {"liquidations 24h / volume 24h", "liqs. 24h / vol 24h"},
        "liq_oi_ratio": {"liquidations 24h / open interest", "liqs. 24h / oi"},
        "nr_contracts": {"number of contracts", "nr contrs"},
        "ls_accounts_1h": {"long/short accounts ratio (1h)", "l/s ratio (1h)"},
        "long_1h": {"long accounts % (1h)", "long (1h)"},
        "short_1h": {"short accounts % (1h)", "short (1h)"},
        "ls_accounts": {"long/short accounts ratio (1d)", "l/s ratio (1d)"},
        "long_1d": {"long accounts % (1d)", "long (1d)"},
        "short_1d": {"short accounts % (1d)", "short (1d)"},
        "btc_corr30d": {"btc correlation 30 days", "btc corr. 30d"},
        "btc_corr7d": {"btc correlation 7 days", "btc corr. 7d"},
        "eth_corr30d": {"eth correlation 30 days", "eth corr. 30d"},
        "eth_corr7d": {"eth correlation 7 days", "eth corr. 7d"},
        "cvd24": {"cvd24"},
        "lls24": {"long liquidation share 24h", "lls24"},
    }

    header_map: dict[str, int] = {}
    for field_name, field_aliases in aliases.items():
        normalized_aliases = {_normalize_header(alias) for alias in field_aliases}
        matches = [idx for idx, normalized in normalized_headers if normalized in normalized_aliases]

        if len(matches) == 1:
            header_map[field_name] = matches[0]
        elif len(matches) > 1:
            raise ValueError(
                f"[COINALYZE] Schema error: field {field_name!r} matched multiple columns at indices {matches}"
            )

    required_fields = {
        "price", "price_chg24", "volume24", "oi", "oi_chg24_pct", "oi_chg4h_pct",
        "oi_vol_ratio", "oi_mktcap_ratio", "fr_oiw", "pfr_oiw", "liq_short24",
        "liq_long24", "ls_accounts", "btc_corr7d", "cvd24", "lls24",
    }

    missing = sorted(required_fields - header_map.keys())
    if missing:
        raise ValueError("[COINALYZE] Schema error: required columns not found: " + ", ".join(missing))

    log.info(
        "[COINALYZE] Field mapping: %s",
        " | ".join(f"{field}→td[{idx}]" for field, idx in sorted(header_map.items())),
    )
    return header_map


def _get_val(tds: list[Any], header_map: dict[str, int], field_name: str) -> float | None:
    idx = header_map.get(field_name)
    if idx is None or idx < 0 or idx >= len(tds):
        return None
    return parse_number(tds[idx].get_text(" ", strip=True))


def _validate_row_semantics(symbol: str, raw: dict[str, float | None]) -> None:
    finite_fields = {
        "price", "volume24", "oi", "oi_chg24_pct", "oi_chg4h_pct", "oi_vol_ratio",
        "oi_mktcap_ratio", "fr_oiw", "pfr_oiw", "liq_short24", "liq_long24",
        "ls_accounts", "btc_corr7d", "cvd24", "lls24",
    }

    for field in finite_fields:
        value = raw.get(field)
        if value is None:
            continue
        try:
            value_float = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric value for {symbol} field={field}: {value!r}")
        if not (float("-inf") < value_float < float("inf")):
            raise ValueError(f"Non-finite value for {symbol} field={field}: {value!r}")

    if raw.get("price") is not None and raw["price"] <= 0:
        raise ValueError(f"Invalid non-positive price for {symbol}: {raw['price']}")
    if raw.get("volume24") is not None and raw["volume24"] < 0:
        raise ValueError(f"Invalid negative Volume 24H for {symbol}: {raw['volume24']}")
    if raw.get("oi") is not None and raw["oi"] < 0:
        raise ValueError(f"Invalid negative Open Interest for {symbol}: {raw['oi']}")


def _setup_browser_context(p):
    browser = p.chromium.launch(
        headless=True,
        args=["--disable-dev-shm-usage", "--no-sandbox"],
    )

    ctx = browser.new_context(
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/150.0.0.0 Safari/537.36",
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        timezone_id="Europe/Berlin",
    )

    cookies = []
    if COINALYZE_P_SID:
        cookies.append({"name": "p_sid", "value": COINALYZE_P_SID, "domain": "coinalyze.net", "path": "/", "secure": True})
    if COINALYZE_CHAT_SID:
        cookies.append({"name": "chat_sid", "value": COINALYZE_CHAT_SID, "domain": "coinalyze.net", "path": "/", "secure": True})
    cookies.append({"name": "cookies_accepted", "value": "1", "domain": "coinalyze.net", "path": "/", "secure": True})

    ctx.add_cookies(cookies)
    page = ctx.new_page()

    try:
        from playwright_stealth import Stealth
        Stealth().apply_stealth_sync(page)
    except (ImportError, AttributeError):
        try:
            from playwright_stealth import stealth_sync
            stealth_sync(page)
        except ImportError as exc:
            raise RuntimeError("[COINALYZE] playwright-stealth is not installed properly.") from exc

    return browser, page


def _load_page(page, url: str) -> str:
    page.goto(url, wait_until="networkidle", timeout=60_000)
    page.wait_for_timeout(3_000)

    content = page.content()
    if "Attention Required" in content:
        log.warning("[COINALYZE] Cloudflare challenge detected, waiting...")
        page.wait_for_timeout(10_000)
        try:
            page.wait_for_load_state("networkidle", timeout=30_000)
        except Exception:
            pass

    page.wait_for_selector("tbody tr", timeout=25_000)
    prev = len(page.query_selector_all("tbody tr"))

    for attempt in range(15):
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(700)
        cur = len(page.query_selector_all("tbody tr"))

        if page.query_selector(".pagination") is not None:
            log.info("[COINALYZE] Pagination detected after scroll %d; rows=%d", attempt + 1, cur)
            break
        if cur != prev:
            log.info("[COINALYZE] Scroll %d: rows %d -> %d", attempt + 1, prev, cur)
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
    log.info("[COINALYZE] Page rows=%d (pagination=%s)", final_count, bool(page.query_selector(".pagination")))
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
    if not rows:
        raise ValueError("[COINALYZE] Table schema error: no <tbody><tr> rows found")

    header_map = _build_header_map(soup)
    out: list[CoinalyzeRow] = []

    for row_number, tr in enumerate(rows, start=1):
        symbol = (tr.get("data-coin") or "").strip().upper()
        tds = tr.find_all("td")

        if not symbol:
            log.warning("[COINALYZE] Skipping row %d: missing data-coin", row_number)
            continue
        if len(tds) <= 1:
            log.warning("[COINALYZE] Skipping row %d (%s): insufficient cells=%d", row_number, symbol, len(tds))
            continue

        spans = tds[1].find_all("span")
        name = spans[0].get_text(strip=True) if spans else symbol

        raw: dict[str, float | None] = {
            "price": _get_val(tds, header_map, "price"),
            "price_chg24": _get_val(tds, header_map, "price_chg24"),
            "mktcap": _get_val(tds, header_map, "mktcap"),
            "volume24": _get_val(tds, header_map, "volume24"),
            "oi": _get_val(tds, header_map, "oi"),
            "oi_chg24_pct": _get_val(tds, header_map, "oi_chg24_pct"),
            "oi_chg24_abs": _get_val(tds, header_map, "oi_chg24_abs"),
            "oi_chg4h_pct": _get_val(tds, header_map, "oi_chg4h_pct"),
            "oi_chg4h_abs": _get_val(tds, header_map, "oi_chg4h_abs"),
            "oi_vol_ratio": _get_val(tds, header_map, "oi_vol_ratio"),
            "oi_mktcap_ratio": _get_val(tds, header_map, "oi_mktcap_ratio"),
            "fr_avg": _get_val(tds, header_map, "fr_avg"),
            "pfr_avg": _get_val(tds, header_map, "pfr_avg"),
            "fr_oiw": _get_val(tds, header_map, "fr_oiw"),
            "pfr_oiw": _get_val(tds, header_map, "pfr_oiw"),
            "liq_total24": _get_val(tds, header_map, "liq_total24"),
            "liq_short24": _get_val(tds, header_map, "liq_short24"),
            "liq_long24": _get_val(tds, header_map, "liq_long24"),
            "liq_vol_ratio": _get_val(tds, header_map, "liq_vol_ratio"),
            "liq_oi_ratio": _get_val(tds, header_map, "liq_oi_ratio"),
            "nr_contracts": _get_val(tds, header_map, "nr_contracts"),
            "ls_accounts_1h": _get_val(tds, header_map, "ls_accounts_1h"),
            "long_1h": _get_val(tds, header_map, "long_1h"),
            "short_1h": _get_val(tds, header_map, "short_1h"),
            "ls_accounts": _get_val(tds, header_map, "ls_accounts"),
            "long_1d": _get_val(tds, header_map, "long_1d"),
            "short_1d": _get_val(tds, header_map, "short_1d"),
            "btc_corr30d": _get_val(tds, header_map, "btc_corr30d"),
            "btc_corr7d": _get_val(tds, header_map, "btc_corr7d"),
            "eth_corr30d": _get_val(tds, header_map, "eth_corr30d"),
            "eth_corr7d": _get_val(tds, header_map, "eth_corr7d"),
            "cvd24": _get_val(tds, header_map, "cvd24"),
            "lls24": _get_val(tds, header_map, "lls24"),
        }

        try:
            _validate_row_semantics(symbol=symbol, raw=raw)
        except ValueError as exc:
            log.error("[COINALYZE] Validation failed for %s: %s", symbol, exc)
            continue

        out.append(
            CoinalyzeRow(
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
            )
        )

    log.info("[COINALYZE] Parsed rows=%d/%d", len(out), len(rows))
    return out


def fetch_data() -> list[CoinalyzeRow]:
    all_rows: list[CoinalyzeRow] = []
    seen: set[str] = set()
    page_errors: list[dict[str, Any]] = []

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser, page = _setup_browser_context(p)
        try:
            html = _load_page(page, COINALYZE_URL)
            try:
                with open(DEBUG_HTML_FILE, "w", encoding="utf-8") as f:
                    f.write(html)
            except OSError as exc:
                log.warning("[COINALYZE] Cannot write debug HTML %s: %s", DEBUG_HTML_FILE, exc)

            rows = parse_table(html)
            for row in rows:
                if row.symbol not in seen:
                    all_rows.append(row)
                    seen.add(row.symbol)

            log.info("[COINALYZE] Page 1: %d rows", len(rows))
            page_urls = get_page_urls(html)
            log.info("[COINALYZE] Pagination: %d pages", len(page_urls))

            if len(page_urls) > 1:
                for i, url in enumerate(page_urls[1:], start=2):
                    try:
                        html = _load_page(page, url)
                        rows = parse_table(html)
                        added = 0
                        for row in rows:
                            if row.symbol not in seen:
                                all_rows.append(row)
                                seen.add(row.symbol)
                                added += 1
                        log.info("[COINALYZE] Page %d: +%d new rows", i, added)
                    except Exception as exc:
                        page_errors.append({"page": i, "url": url, "error": str(exc)[:200]})
                        log.exception("[COINALYZE] Page %d failed", i)

            elif page.query_selector(".pagination") is not None:
                page_num = 1
                while page_num < MAX_PAGES and click_next_page(page, page_num):
                    page.wait_for_selector("tbody tr", timeout=15_000)
                    page.wait_for_timeout(500)
                    html = page.content()
                    rows = parse_table(html)
                    added = 0
                    for row in rows:
                        if row.symbol not in seen:
                            all_rows.append(row)
                            seen.add(row.symbol)
                            added += 1
                    page_num += 1
                    log.info("[COINALYZE] Click page %d: +%d new rows", page_num, added)
        finally:
            browser.close()

    if page_errors:
        log.warning("[COINALYZE] Scrape incomplete: %d page errors", len(page_errors))

    log.info("[COINALYZE] Total unique rows=%d", len(all_rows))
    return all_rows
