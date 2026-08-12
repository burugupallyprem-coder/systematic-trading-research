"""Historical bars from Alpaca Market Data v2.

Backtests use feed=sip when available (free plan includes historical SIP except
the most recent 15 minutes); falls back to IEX automatically.
"""

import os
import time
from datetime import time as dtime

import pandas as pd
import requests

DATA_BASE = "https://data.alpaca.markets"


def _headers():
    return {
        "APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
        "APCA-API-SECRET-KEY": os.environ["ALPACA_SECRET_KEY"],
    }


def fetch_bars(symbols, start, end, timeframe="5Min", feed="sip",
               limit=10000, max_pages=5000):
    """Return DataFrame [symbol, ts, open, high, low, close, volume] (ts UTC)."""
    params = {
        "symbols": ",".join(symbols), "timeframe": timeframe,
        "start": start, "end": end, "limit": limit,
        "adjustment": "split", "feed": feed, "sort": "asc",
    }
    rows = []
    token = None
    session = requests.Session()
    for _ in range(max_pages):
        p = dict(params)
        if token:
            p["page_token"] = token
        resp = session.get(DATA_BASE + "/v2/stocks/bars", params=p,
                           headers=_headers(), timeout=90)
        if resp.status_code == 429:  # rate limited - wait and retry
            time.sleep(3)
            continue
        if resp.status_code in (400, 403) and feed == "sip":
            # SIP not available on this plan/range - restart with IEX
            return fetch_bars(symbols, start, end, timeframe, feed="iex",
                              limit=limit, max_pages=max_pages)
        resp.raise_for_status()
        data = resp.json()
        for sym, bars in (data.get("bars") or {}).items():
            for b in bars:
                rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
        token = data.get("next_page_token")
        if not token:
            break
    df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low",
                                     "close", "volume"])
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def rth_only(df):
    """Keep regular trading hours only; add ET timestamp + session date."""
    if df.empty:
        return df
    et = df["ts"].dt.tz_convert("America/New_York")
    mask = (et.dt.time >= dtime(9, 30)) & (et.dt.time < dtime(16, 0))
    out = df.loc[mask].copy()
    out["et"] = et[mask]
    out["date"] = out["et"].dt.date
    return out.reset_index(drop=True)

def fetch_bars_hist(symbols, start, end, timeframe="5Min", feed="sip", limit=10000, max_pages=5000):
    """Long-history fetch: pull each symbol SEPARATELY and, for each, try BOTH feeds and keep
    whichever reaches furthest back. SIP entitlement varies per symbol on the free plan (some
    silently fall back to IEX, which caps ~2020-07), so trying sip AND iex is what actually
    reaches the pre-COVID / COVID-crash bars. Logs the earliest date per symbol so the CI run
    shows exactly how far back each symbol goes."""
    feeds = [feed, "iex" if feed != "iex" else "sip"]
    frames = []
    for sym in symbols:
        best = None
        for fd in feeds:
            try:
                b = fetch_bars([sym], start, end, timeframe=timeframe, feed=fd,
                               limit=limit, max_pages=max_pages)
            except Exception as e:
                print(f"[fetch_bars_hist] {sym} feed={fd} failed: {e}", flush=True)
                continue
            if len(b) and (best is None or b["ts"].min() < best["ts"].min()):
                best = b
        if best is not None:
            print(f"[fetch_bars_hist] {sym}: {len(best):>7} bars, earliest {best['ts'].min().date()}", flush=True)
            frames.append(best)
        else:
            print(f"[fetch_bars_hist] {sym}: NO DATA on any feed", flush=True)
    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(["symbol", "ts"]).sort_values(["symbol", "ts"]).reset_index(drop=True)

