from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .bingx import BingXClient, Contract, quantize_qty, quantize_price


def make_client_order_id(event_id: str) -> str:
    # Deterministic per event: a retry of the same event reuses the same clientOrderId.
    return 'evt' + hashlib.sha256(event_id.encode()).hexdigest()[:36]


def execute_vst(client: BingXClient, contract: Contract, setup: dict,
                margin_usdt: float, leverage: float, position_mode: str = 'HEDGE') -> dict:
    direction = str(setup['direction']).upper()
    side = 'LONG' if direction == 'LONG' else 'SHORT'
    position_side = side if str(position_mode).upper() == 'HEDGE' else 'BOTH'
    lev = max(1, int(round(leverage)))
    client_order_id = make_client_order_id(setup['event_id'] or f"NOEVENT-{setup['symbol']}")

    # Idempotency: check the exchange before attempting a new MARKET order.
    existing = client.query_order(contract.symbol, client_order_id)
    if existing:
        return {
            'mode': 'vst',
            'status': existing.get('status', 'EXISTING'),
            'order_id': existing.get('orderID') or existing.get('orderId'),
            'symbol': contract.symbol,
            'quantity': existing.get('origQty') or existing.get('quantity'),
            'idempotent_reuse': True,
            'raw': existing,
        }

    if not contract.api_state_open:
        raise RuntimeError(f'BingX contract is not open: {contract.symbol}')

    client.set_leverage(contract.symbol, lev, position_side)

    entry = float(setup['entry_reference'])
    raw_qty = (margin_usdt * leverage) / entry
    qty = quantize_qty(raw_qty, contract.qty_precision)
    if qty < contract.min_qty:
        qty = quantize_qty(contract.min_qty, contract.qty_precision)
    if contract.min_trade_usdt > 0 and qty * entry < contract.min_trade_usdt:
        required = contract.min_trade_usdt / entry
        qty = quantize_qty(required, contract.qty_precision)
        if qty < contract.min_qty:
            qty = quantize_qty(contract.min_qty, contract.qty_precision)
    if qty <= 0:
        raise RuntimeError(f'Invalid quantity for {contract.symbol}: {qty}')

    sl_price = quantize_price(float(setup['invalidation_price']), contract.price_precision, contract.tick_size)
    tp_price = quantize_price(float(setup['target_price']), contract.price_precision, contract.tick_size)

    data = client.place_market_with_protection(
        symbol=contract.symbol,
        direction=direction,
        quantity=qty,
        sl_price=sl_price,
        tp_price=tp_price,
        client_order_id=client_order_id,
        price_precision=contract.price_precision,
    ) or {}

    return {
        'mode': 'vst',
        'status': data.get('status', 'UNKNOWN'),
        'order_id': data.get('orderID') or data.get('orderId'),
        'client_order_id': client_order_id,
        'symbol': contract.symbol,
        'quantity': qty,
        'margin_usdt': margin_usdt,
        'leverage': leverage,
        'idempotent_reuse': False,
        'raw': data,
    }
