from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from urllib.parse import urlencode

import pandas as pd
import requests


@dataclass(frozen=True)
class Contract:
    symbol: str
    min_qty: float
    qty_precision: int
    price_precision: int
    status: str
    api_state_open: bool
    tick_size: float | None = None
    min_trade_usdt: float = 0.0


class BingXClient:
    def __init__(self, api_key: str, secret_key: str, environment: str = 'vst', recv_window: int = 5000):
        self.api_key = api_key
        self.secret_key = secret_key
        self.recv_window = recv_window
        if environment.lower() == 'live':
            self.base = 'https://open-api.bingx.com'
            self.fallback = 'https://open-api.bingx.pro'
        else:
            self.base = 'https://open-api-vst.bingx.com'
            self.fallback = 'https://open-api-vst.bingx.pro'
        self.session = requests.Session()
        self.session.headers.update({'X-BX-APIKEY': self.api_key})

    def _request(self, method: str, path: str, params: dict | None = None, signed: bool = True):
        params = dict(params or {})
        if signed:
            params.setdefault('timestamp', int(time.time() * 1000))
            params.setdefault('recvWindow', self.recv_window)
            query = urlencode(sorted(params.items()), doseq=True)
            signature = hmac.new(self.secret_key.encode(), query.encode(), hashlib.sha256).hexdigest()
            url = f'{self.base}{path}?{query}&signature={signature}'
        else:
            query = urlencode(params, doseq=True)
            url = f'{self.base}{path}' + (f'?{query}' if query else '')

        try:
            response = self.session.request(method, url, timeout=20)
        except requests.RequestException:
            fb = self.fallback
            url = f'{fb}{path}?{query}&signature={signature}' if signed else f'{fb}{path}' + (f'?{query}' if query else '')
            response = self.session.request(method, url, timeout=20)

        response.raise_for_status()
        payload = response.json()
        if payload.get('code') not in (0, '0'):
            raise RuntimeError(f"BingX API error {payload.get('code')}: {payload.get('msg')}")
        return payload.get('data')

    def klines(self, symbol: str, interval: str, limit: int = 250) -> pd.DataFrame:
        data = self._request('GET', '/openApi/swap/v3/quote/klines', {
            'symbol': symbol,
            'interval': interval,
            'limit': limit,
        }, signed=True)
        rows = []
        now = int(time.time() * 1000)
        for row in data or []:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                continue
            try:
                if int(row[6]) > now:
                    continue
                rows.append({
                    'open_time': int(row[0]),
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': float(row[5]),
                    'close_time': int(row[6]),
                    'quote_volume': float(row[7]) if len(row) > 7 else None,
                    'trades_count': int(row[8]) if len(row) > 8 else None,
                    'taker_buy_base': float(row[9]) if len(row) > 9 else None,
                    'taker_buy_quote': float(row[10]) if len(row) > 10 else None,
                })
            except (TypeError, ValueError):
                continue

        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df = df.sort_values('close_time').drop_duplicates('close_time').reset_index(drop=True)
        valid = (
            df['quote_volume'].notna()
            & df['taker_buy_base'].notna()
            & df['taker_buy_quote'].notna()
            & (df['taker_buy_base'] <= df['volume'] * 1.001 + 1e-9)
            & (df['taker_buy_quote'] <= df['quote_volume'] * 1.001 + 1e-9)
        )
        df['taker_flow_valid'] = valid
        df['bar_delta_usdt'] = 2 * df['taker_buy_quote'] - df['quote_volume']
        df.loc[~valid, ['bar_delta_usdt']] = float('nan')
        df['cvd_segment_id'] = (~valid).cumsum()
        df['bingx_cvd'] = float('nan')
        for seg, idx in df.groupby('cvd_segment_id').groups.items():
            idx_list = list(idx)
            good = df.loc[idx_list, 'taker_flow_valid'].fillna(False)
            good_idx = [i for i, ok in zip(idx_list, good) if bool(ok)]
            if good_idx:
                df.loc[good_idx, 'bingx_cvd'] = df.loc[good_idx, 'bar_delta_usdt'].cumsum()
        return df

    def contracts(self) -> list[Contract]:
        data = self._request('GET', '/openApi/swap/v2/quote/contracts', {})
        out: list[Contract] = []
        for c in data or []:
            try:
                symbol = str(c.get('symbol', ''))
                if not symbol:
                    continue
                api_open_raw = c.get('apiStateOpen')
                api_open = str(api_open_raw).lower() in {'1', 'true', 'open'} if api_open_raw is not None else True
                status = str(c.get('status') or '')
                out.append(Contract(
                    symbol=symbol,
                    min_qty=float(c.get('tradeMinQuantity') or c.get('minQty') or 0),
                    qty_precision=int(c.get('quantityPrecision') or 0),
                    price_precision=int(c.get('pricePrecision') or 0),
                    status=status,
                    api_state_open=api_open,
                    tick_size=float(c.get('tickSize')) if c.get('tickSize') else None,
                    min_trade_usdt=float(c.get('tradeMinUSDT') or c.get('minNotional') or 0),
                ))
            except (TypeError, ValueError):
                continue
        return out

    def set_leverage(self, symbol: str, leverage: int, side: str) -> dict:
        return self._request('POST', '/openApi/swap/v2/trade/leverage', {
            'symbol': symbol,
            'leverage': leverage,
            'side': side,
        })


    def query_order(self, symbol: str, client_order_id: str) -> dict | None:
        try:
            return self._request('GET', '/openApi/swap/v2/trade/order', {
                'symbol': symbol,
                'clientOrderId': client_order_id,
            })
        except RuntimeError as exc:
            msg = str(exc).lower()
            if 'not found' in msg or 'no order' in msg or 'order does not exist' in msg:
                return None
            raise

    def place_market_with_protection(self, symbol: str, direction: str, quantity: float,
                                     sl_price: float, tp_price: float,
                                     client_order_id: str,
                                     price_precision: int) -> dict:
        side = 'BUY' if direction == 'LONG' else 'SELL'
        position_side = 'LONG' if direction == 'LONG' else 'SHORT'
        stop = f'{sl_price:.{price_precision}f}'
        take = f'{tp_price:.{price_precision}f}'
        payload = {
            'symbol': symbol,
            'side': side,
            'positionSide': position_side,
            'type': 'MARKET',
            'quantity': quantity,
            'clientOrderId': client_order_id[:40],
            'workingType': 'MARK_PRICE',
            'stopLoss': json.dumps({
                'type': 'STOP_MARKET',
                'stopPrice': float(stop),
                'workingType': 'MARK_PRICE',
            }, separators=(',', ':')),
            'takeProfit': json.dumps({
                'type': 'TAKE_PROFIT_MARKET',
                'stopPrice': float(take),
                'workingType': 'MARK_PRICE',
            }, separators=(',', ':')),
        }
        return self._request('POST', '/openApi/swap/v2/trade/order', payload)



def quantize_price(price: float, precision: int, tick_size: float | None = None) -> float:
    if tick_size and tick_size > 0:
        units = Decimal(str(price)) / Decimal(str(tick_size))
        units = units.quantize(Decimal('1'), rounding=ROUND_DOWN)
        return float(units * Decimal(str(tick_size)))
    q = Decimal(1).scaleb(-precision)
    return float(Decimal(str(price)).quantize(q, rounding=ROUND_DOWN))

def quantize_qty(qty: float, precision: int) -> float:
    q = Decimal(1).scaleb(-precision)
    return float(Decimal(str(qty)).quantize(q, rounding=ROUND_DOWN))
