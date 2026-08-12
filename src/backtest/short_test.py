"""Add the MIRROR short side to the ORB and compare vs the long-only live config.
LONG (unchanged live): up-breakout, SPY breaking up, strongest-5 by RS.
SHORT (new): down-breakout, SPY breaking DOWN, weakest-5 by RS, hold to rr.
Reports year-by-year avg-daily + maxDD + TTP-FLEX pass for LONG-ONLY vs SYMMETRIC, so you
can see if shorts add crisis-alpha in down years. Needs Alpaca bars (CI). RESEARCH ONLY."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import engine, research
from src.strategies import filters, orb
from src.backtest.ttp_eval import sweep

FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)


def _daily(cfg, start="2018-01-01"):
    live = cfg["live"]; params = dict(live["params"])
    ob = int(cfg["research"].get("regime_open_bars", 3))
    cutoff = params.get("cutoff_et", "10:30")
    k = params.get("rs_topk")
    risk_pct = float(cfg["risk"]["risk_pct"]) / 100.0
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bars = data_mod.fetch_bars_hist(cfg["universe"], start, end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    spy_days = {d["date"].iloc[0]: d for s, d in groups if s == "SPY"}
    spy_up = {dt: filters.spy_long_ok(sd, ob, cutoff) for dt, sd in spy_days.items()}
    spy_dn = {dt: filters.spy_short_ok(sd, ob, cutoff) for dt, sd in spy_days.items()}
    dates = sorted({d["date"].iloc[0] for _, d in groups})
    long_r = {d: 0.0 for d in dates}; both_r = {d: 0.0 for d in dates}
    for sym, day in groups:
        dt = day["date"].iloc[0]
        rs = ctx["rs"].get(dt, {})
        # LONG side (identical to live)
        if sym in filters.top_k_symbols(rs, k) and spy_up.get(dt):
            for t in engine.simulate_day(day, orb.generate(day, {**params, "side": "long"},
                                         {"spy_long_ok": True}), cfg, "orb"):
                long_r[dt] += float(getattr(t, "r_multiple", 0.0)); both_r[dt] += float(getattr(t, "r_multiple", 0.0))
        # SHORT side (new): down-break, SPY breaking down, weakest-k
        if sym in filters.bottom_k_symbols(rs, k) and spy_dn.get(dt):
            for t in engine.simulate_day(day, orb.generate(day, {**params, "side": "short"}, None), cfg, "orb"):
                both_r[dt] += float(getattr(t, "r_multiple", 0.0))
    idx = pd.to_datetime(dates)
    return pd.Series([long_r[d] * risk_pct for d in dates], index=idx), \
           pd.Series([both_r[d] * risk_pct for d in dates], index=idx)


def _stats(r):
    r = r.dropna()
    if len(r) < 10 or r.std() == 0: return 0.0, 0.0, 0.0
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return round(r.mean() * 100, 3), round(dd * 100, 1), round(r.mean() / r.std() * np.sqrt(252), 2)


def run():
    cfg = research.load_config()
    lon, both = _daily(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[SHORT-TEST] {ts} - LONG-ONLY (live) vs SYMMETRIC (added mirror short side)",
         f"data {lon.index[0].date()}..{lon.index[-1].date()} ({len(lon)} days)", "",
         "YEAR   LONG-ONLY avgDaily/maxDD    SYMMETRIC avgDaily/maxDD"]
    for y in sorted(set(lon.index.year)):
        la, ld, _ = _stats(lon[lon.index.year == y])
        ba, bd, _ = _stats(both[both.index.year == y])
        L.append(f"{y}   {la:+.3f}% / {ld:>5}%        {ba:+.3f}% / {bd:>5}%")
    for name, r in [("LONG-ONLY (live)", lon), ("SYMMETRIC (long+short)", both)]:
        av, dd, sh = _stats(r); fp, n = sweep(r.values, step=3, **FLEX)
        L += ["", f"{name}: avgDaily {av:+.3f}%  Sharpe {sh}  maxDD {dd}%  TTP-FLEX {fp}%"]
    L.append("\nRead: shorts should HELP in down years (deeper red maxDD gets shallower / avgDaily less "
             "negative) and may HURT in strong bull years. RESEARCH ONLY - survivorship + wide-CI caveats apply.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
