from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
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


def _num(value: Any) -> float | None:
    if value is None:
        return None

    s = str(value).strip()

    if not s or s.lower() in {"nan", "none", "n/a", "-", "—"}:
        return None

    s = (
        s.replace(",", "")
         .replace("$", "")
         .replace("%", "")
         .strip()
    )

    multiplier = 1.0

    if s:
        suffix = s[-1].upper()

        if suffix in {"K", "M", "B", "T"}:
            multiplier = {
                "K": 1e3,
                "M": 1e6,
                "B": 1e9,
                "T": 1e12,
            }[suffix]

            s = s[:-1].strip()

    try:
        return float(s) * multiplier
    except (TypeError, ValueError):
        return None


def _norm_column(value: Any) -> str:
    s = str(value).strip().lower()

    s = re.sub(
        r"[^a-z0-9]+",
        "_",
        s,
    )

    s = s.strip("_")

    aliases = {
        "coin": "coin",
        "price": "price",

        "chg_24h": "price_chg_24h",
        "price_change_24h": "price_chg_24h",
        "price_chg_24h": "price_chg_24h",

        "mkt_cap": "market_cap",
        "market_capitalisation": "market_cap",
        "market_capitalization": "market_cap",

        "vol_24h": "volume_24h",
        "volume_24h": "volume_24h",

        "open_interest": "open_interest",

        "oi_chg_24h": "oi_chg_24h_pct",
        "oi_chg_24h_pct": "oi_chg_24h_pct",
        "open_interest_change_24h": "oi_chg_24h_pct",
        "open_interest_change_24h_pct": "oi_chg_24h_pct",

        "oi_chg_4h": "oi_chg_4h_pct",
        "oi_chg_4h_pct": "oi_chg_4h_pct",
        "open_interest_change_4h": "oi_chg_4h_pct",
        "open_interest_change_4h_pct": "oi_chg_4h_pct",

        "oi_vol24h": "oi_vol_ratio",
        "oi_vol_24h": "oi_vol_ratio",
        "oi_volume_24h": "oi_vol_ratio",
        "oi_volume_ratio": "oi_vol_ratio",

        "fr_avg_oi_w": "funding_oiw",
        "fr_avg_oiw": "funding_oiw",
        "funding_rate_average_oi_weighted": "funding_oiw",

        "pfr_avg_oi_w": "predicted_funding_oiw",
        "pfr_avg_oiw": "predicted_funding_oiw",
        "predicted_funding_rate_average_oi_weighted": "predicted_funding_oiw",

        "short_liqs_24h": "short_liq_24h",
        "short_liquidations_24h": "short_liq_24h",

        "long_liqs_24h": "long_liq_24h",
        "long_liquidations_24h": "long_liq_24h",

        "l_s_ratio_1h": "ls_ratio_1h",
        "long_short_accounts_ratio_1h": "ls_ratio_1h",

        "l_s_ratio_1d": "ls_ratio_1d",
        "long_short_accounts_ratio_1d": "ls_ratio_1d",
    }

    return aliases.get(s, s)


def normalize_discovery_symbol(symbol: str) -> str:
    """
    Coinalyze может показывать:
        BTC
        ETH
        SOL
        1000FLOKI
        10000SATS

    Для BingX нам нужен базовый asset:
        BTC
        ETH
        SOL
        FLOKI
        SATS
    """

    s = str(symbol).strip().upper()

    # 1000FLOKI -> FLOKI
    for prefix in ("10000", "1000"):
        if s.startswith(prefix):
            tail = s[len(prefix):]

            if re.fullmatch(r"[A-Z][A-Z0-9._-]{1,20}", tail):
                return tail

    return s


def _extract_ticker_from_text(value: Any) -> str | None:
    """
    Пример:
        'Bitcoin BTC' -> BTC
        'Ethereum ETH' -> ETH
        'Solana SOL' -> SOL
        'Ripple XRP' -> XRP
        '1000FLOKI' -> 1000FLOKI
    """

    if value is None:
        return None

    text = str(value).strip().upper()

    if not text or text in {"NAN", "NONE"}:
        return None

    # Сначала ищем короткий тикер в конце строки:
    # Bitcoin BTC
    # Solana SOL
    # Ripple XRP
    match = re.search(
        r"(?:^|\s)([A-Z0-9]{2,20})$",
        text,
    )

    if match:
        token = match.group(1)

        blacklist = {
            "PRICE",
            "VOLUME",
            "OPEN",
            "INTEREST",
            "COIN",
            "USDT",
            "USD",
            "PERPETUAL",
            "CONTRACT",
        }

        if token not in blacklist:
            return normalize_discovery_symbol(token)

    # Если строка сама является тикером:
    if re.fullmatch(
        r"[A-Z0-9][A-Z0-9._-]{1,20}",
        text,
    ):
        return normalize_discovery_symbol(text)

    return None


def _find_symbol(row: pd.Series) -> str | None:
    """
    В первую очередь используем колонку Coin.
    """

    # Самое надежное:
    if "coin" in row.index:
        symbol = _extract_ticker_from_text(
            row.get("coin")
        )

        if symbol:
            return symbol

    # Fallback: смотрим первые несколько ячеек.
    for value in row.tolist()[:6]:
        symbol = _extract_ticker_from_text(value)

        if symbol:
            return symbol

    return None


class CoinalyzeClient:

    def __init__(self, url: str):
        self.url = url

        self.session = requests.Session()

        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 "
                "(X11; Linux x86_64) "
                "AppleWebKit/537.36 "
                "(KHTML, like Gecko) "
                "Chrome/150.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        })

    def _load_static_html(self) -> str:
        response = self.session.get(
            self.url,
            timeout=30,
        )

        response.raise_for_status()

        return response.text

    def _load_rendered_html(self) -> str:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError(
                "Playwright is required for rendered Coinalyze fallback."
            ) from exc

        with sync_playwright() as pw:

            browser = pw.chromium.launch(
                headless=True
            )

            page = browser.new_page(
                viewport={
                    "width": 1920,
                    "height": 1080,
                }
            )

            page.goto(
                self.url,
                wait_until="networkidle",
                timeout=60_000,
            )

            page.wait_for_timeout(2_000)

            html = page.content()

            browser.close()

        return html

    @staticmethod
    def _select_table(
        tables: list[pd.DataFrame],
    ) -> pd.DataFrame | None:

        for table in tables:

            columns = {
                _norm_column(c)
                for c in table.columns
            }

            has_price = (
                "price" in columns
            )

            has_volume = (
                "volume_24h" in columns
            )

            has_oi = (
                "open_interest" in columns
            )

            if (
                has_price
                and (has_volume or has_oi)
            ):
                return table

        return None

    def fetch(self) -> list[CoinalyzeRow]:

        html = self._load_static_html()

        try:
            tables = pd.read_html(
                StringIO(html)
            )
        except ValueError:
            tables = []

        table = self._select_table(
            tables
        )

        # Если таблица не присутствует в static HTML,
        # пробуем реальный browser rendering.
        if table is None:

            print(
                "[COINALYZE] "
                "static HTML table not found; "
                "using Playwright"
            )

            rendered = self._load_rendered_html()

            try:
                tables = pd.read_html(
                    StringIO(rendered)
                )
            except ValueError as exc:
                raise RuntimeError(
                    "Unable to parse Coinalyze rendered table."
                ) from exc

            table = self._select_table(
                tables
            )

        if table is None:

            print(
                "[COINALYZE] "
                "No suitable table found."
            )

            return []

        # Нормализуем названия колонок.
        table = table.copy()

        table.columns = [
            _norm_column(c)
            for c in table.columns
        ]

        print(
            "[COINALYZE] columns="
            + ",".join(table.columns)
        )

        rows: list[CoinalyzeRow] = []

        for _, row in table.iterrows():

            symbol = _find_symbol(
                row
            )

            if not symbol:
                continue

            item = CoinalyzeRow(
                symbol=symbol,

                price=_num(
                    row.get("price")
                ),

                price_chg_24h=_num(
                    row.get("price_chg_24h")
                ),

                volume_24h=_num(
                    row.get("volume_24h")
                ),

                open_interest=_num(
                    row.get("open_interest")
                ),

                oi_chg_24h_pct=_num(
                    row.get("oi_chg_24h_pct")
                ),

                oi_chg_4h_pct=_num(
                    row.get("oi_chg_4h_pct")
                ),

                funding_oiw=_num(
                    row.get("funding_oiw")
                ),

                predicted_funding_oiw=_num(
                    row.get(
                        "predicted_funding_oiw"
                    )
                ),

                short_liq_24h=_num(
                    row.get("short_liq_24h")
                ),

                long_liq_24h=_num(
                    row.get("long_liq_24h")
                ),

                ls_ratio_1h=_num(
                    row.get("ls_ratio_1h")
                ),

                ls_ratio_1d=_num(
                    row.get("ls_ratio_1d")
                ),

                oi_vol_ratio=_num(
                    row.get("oi_vol_ratio")
                ),
            )

            rows.append(item)

        print(
            f"[COINALYZE] parsed_rows={len(rows)}"
        )

        if rows:
            print(
                "[COINALYZE] symbols="
                + ",".join(
                    row.symbol
                    for row in rows[:30]
                )
            )

            print(
                "[COINALYZE] sample="
                + str([
                    {
                        "symbol": r.symbol,
                        "volume_24h": r.volume_24h,
                        "open_interest": r.open_interest,
                    }
                    for r in rows[:5]
                ])
            )

        return rows
