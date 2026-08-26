from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import pandas as pd
import requests


@dataclass
class CoinalyzeRow:
    symbol: str
    price: float | None = None
    price_chg_24h: float | None = None
    volume_24h: float | None = None
    open_interest: float | None = None
    oi_chg_24h_pct: float | None = None
    oi_chg_4h_pct: float | None = None
    funding_oiw: float | None = None
    predicted_funding_oiw: float | None = None
    short_liq_24h: float | None = None
    long_liq_24h: float | None = None
    ls_ratio_1h: float | None = None
    ls_ratio_1d: float | None = None
    oi_vol_ratio: float | None = None


def _num(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip().replace(',', '').replace('%', '')
    if s in {'', '-', '—', 'N/A', 'nan', 'None'}:
        return None
    mult = 1.0
    suffix = s[-1:].upper()
    if suffix in {'K', 'M', 'B'}:
        mult = {'K': 1e3, 'M': 1e6, 'B': 1e9}[suffix]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return None


def _norm(s: Any) -> str:
    x = re.sub(r'[^a-z0-9]+', '_', str(s).strip().lower()).strip('_')
    aliases = {
        'chg_24h': 'price_chg_24h',
        'price_change_24h': 'price_chg_24h',
        'vol_24h': 'volume_24h',
        'open_interest_chg_24h': 'oi_chg_24h_pct',
        'oi_chg_24h': 'oi_chg_24h_pct',
        'oi_chg_4h': 'oi_chg_4h_pct',
        'open_interest_change_4h': 'oi_chg_4h_pct',
        'fr_avg_oi_w': 'funding_oiw',
        'fr_avg_oiw': 'funding_oiw',
        'pfr_avg_oi_w': 'predicted_funding_oiw',
        'pfr_avg_oiw': 'predicted_funding_oiw',
        'short_liqs_24h': 'short_liq_24h',
        'long_liqs_24h': 'long_liq_24h',
        'l_s_ratio_1h': 'ls_ratio_1h',
        'l_s_ratio_1d': 'ls_ratio_1d',
        'oi_vol_24h': 'oi_vol_ratio',
        'oi_volume_24h': 'oi_vol_ratio',
    }
    return aliases.get(x, x)


def _find_symbol(row: pd.Series) -> str | None:
    candidates = [str(v).strip().upper() for v in row.tolist()[:8]]
    for c in candidates:
        c = re.sub(r'\s+', '', c)
        c = re.sub(r'[/_].*$', '', c)
        c = re.sub(r'-USDT$', '', c)
        if re.fullmatch(r'[A-Z0-9][A-Z0-9._-]{1,19}', c) and c not in {
            'PRICE', 'OPENINTEREST', 'VOLUME', 'USDT', 'USD', 'BTC', 'ETH',
        }:
            return c
    return None


def normalize_discovery_symbol(symbol: str) -> str:
    s = symbol.upper().strip()
    # Coinalyze commonly displays size-prefixed contracts such as 1000FLOKI.
    # BingX execution contracts may be the base asset, e.g. FLOKI-USDT.
    for prefix in ('10000', '1000'):
        if s.startswith(prefix) and len(s) > len(prefix) + 2:
            tail = s[len(prefix):]
            if tail.isalpha():
                return tail
    return s


class CoinalyzeClient:
    def __init__(self, url: str):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/150 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def fetch(self) -> list[CoinalyzeRow]:
        html = self.session.get(self.url, timeout=30).text
        tables = []
        try:
            tables = pd.read_html(html)
        except ValueError:
            tables = []

        # Coinalyze can render the table client-side. Use Playwright only when the
        # static HTML does not contain a usable table.
        if not any({'price', 'open_interest'} <= {_norm(c) for c in t.columns} or
                   ('price' in {_norm(c) for c in t.columns} and 'volume_24h' in {_norm(c) for c in t.columns})
                   for t in tables):
            try:
                from playwright.sync_api import sync_playwright
                with sync_playwright() as pw:
                    browser = pw.chromium.launch(headless=True)
                    page = browser.new_page(viewport={'width': 1920, 'height': 1080})
                    page.goto(self.url, wait_until='networkidle', timeout=60000)
                    page.wait_for_timeout(1500)
                    rendered = page.content()
                    browser.close()
                try:
                    tables = pd.read_html(rendered)
                except ValueError:
                    tables = []
            except Exception as exc:
                raise RuntimeError(
                    'Coinalyze table not found in static HTML and Playwright fallback failed: '
                    f'{exc}'
                ) from exc

        table = None
        for t in tables:
            cols = {_norm(c) for c in t.columns}
            if 'price' in cols and ('open_interest' in cols or 'volume_24h' in cols):
                table = t
                break
        if table is None:
            raise RuntimeError(
                'Coinalyze HTML table not found. The site may be browser-rendered. '
                'Use a rendered table adapter or export the table in the repository.'
            )
        table.columns = [_norm(c) for c in table.columns]
        rows: list[CoinalyzeRow] = []
        for _, r in table.iterrows():
            sym = _find_symbol(r)
            if sym:
                sym = normalize_discovery_symbol(sym)
            if not sym:
                continue
            rows.append(CoinalyzeRow(
                symbol=sym,
                price=_num(r.get('price')),
                price_chg_24h=_num(r.get('price_chg_24h')),
                volume_24h=_num(r.get('volume_24h')),
                open_interest=_num(r.get('open_interest')),
                oi_chg_24h_pct=_num(r.get('oi_chg_24h_pct')),
                oi_chg_4h_pct=_num(r.get('oi_chg_4h_pct')),
                funding_oiw=_num(r.get('funding_oiw')),
                predicted_funding_oiw=_num(r.get('predicted_funding_oiw')),
                short_liq_24h=_num(r.get('short_liq_24h')),
                long_liq_24h=_num(r.get('long_liq_24h')),
                ls_ratio_1h=_num(r.get('ls_ratio_1h')),
                ls_ratio_1d=_num(r.get('ls_ratio_1d')),
                oi_vol_ratio=_num(r.get('oi_vol_ratio')),
            ))
        return rows
