"""Backtest the EXACT live filtered ORB config under Trade The Pool $50k rules (FLEX + MAX).
Re-runs the live config (regime gate + rs_topk5 + rr1.5) via the research harness so it is
apples-to-apples with what the paper bot trades - NOT the demoted raw ORB. Needs Alpaca bars
(runs in CI). TTP rules verified 2026-08-12: 6% target; STATIC max loss (FLEX 4% / MAX 3%)
from START, does NOT trail; daily pause (FLEX 2% / MAX 1%) caps the day; consistency (FLEX
50% / MAX 30%); FLEX >=3 days of >=0.5% profit, unlimited period; MAX 60-day limit."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import engine, research
from src.strategies import filters, orb


def live_daily_returns(cfg, start=None, end=None):
    """Daily % return series of the LIVE filtered ORB config over the full backtest history."""
    live = cfg["live"]
    params = dict(live["params"])
    rs = cfg["research"]
    end = end or rs.get("val_end") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    start = start or rs["train_start"]
    risk_pct = float(cfg["risk"]["risk_pct"]) / 100.0            # 0.5 -> 0.005
    bars = data_mod.fetch_bars(cfg["universe"], start, end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    rs_topk = params.get("rs_topk")

    all_dates = sorted({day["date"].iloc[0] for _, day in groups})
    r_by_date = {d: 0.0 for d in all_dates}
    for symbol, day in groups:                                   # replicate run_config + attach dates
        date = day["date"].iloc[0]
        if rs_topk:
            allowed = filters.top_k_symbols(ctx["rs"].get(date, {}), rs_topk)
            if symbol not in allowed:
                continue
        c = {"spy_long_ok": ctx["spy_long_ok"].get(date, False),
             "prev_close": ctx.get("prev_close", {}).get(symbol, {}).get(date)}
        signals = orb.generate(day, params, c)
        if signals:
            for t in engine.simulate_day(day, signals, cfg, "orb"):
                r_by_date[date] += float(getattr(t, "r_multiple", 0.0) if not isinstance(t, dict) else t.get("r_multiple", 0.0))
    return pd.Series([r_by_date[d] * risk_pct for d in all_dates], index=pd.to_datetime(all_dates))


def eval_ttp(r, start_i, target, daily_lim, max_loss, consist, min_pdays, horizon):
    eq = 0.0; pdays = 0; best_day = -9.0; end = min(len(r), start_i + horizon)
    for k in range(start_i, end):
        day = max(r[k], -daily_lim)                              # daily pause caps the loss
        if day >= 0.005: pdays += 1
        best_day = max(best_day, day)
        eq += day
        if eq <= -max_loss:
            return 0
        if eq >= target and (not min_pdays or pdays >= min_pdays) and best_day <= consist * eq:
            return 1
    return 0


def sweep(r, step=3, **kw):
    outs = [eval_ttp(r, s, **kw) for s in range(0, len(r) - 5, step)]
    return round(100 * np.mean(outs), 1) if outs else 0.0, len(outs)


def run():
    cfg = research.load_config()
    daily = live_daily_returns(cfg)
    r = daily.values
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    span = f"{daily.index[0].date()}..{daily.index[-1].date()} ({len(r)} days)"
    avg = daily.mean() * 100
    flex, n = sweep(r, target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
    mx, _ = sweep(r, target=0.06, daily_lim=0.01, max_loss=0.03, consist=0.30, min_pdays=0, horizon=60)
    msg = (f"[TTP-EVAL] {ts} - LIVE filtered ORB (regime + rs_topk5 + rr1.5) under Trade The Pool $50k\n"
           f"backtest {span} | avg daily {avg:+.3f}% | {n} rolling eval starts\n"
           f"FLEX (6% tgt, 2% daily, 4% STATIC maxloss, 50% consist, 3 profit-days, unlimited): {flex}% pass\n"
           f"MAX  (6% tgt, 1% daily, 3% STATIC maxloss, 30% consist, 60-day limit):            {mx}% pass\n"
           f"RESEARCH ONLY. This is the EXACT live config, not the demoted raw ORB.")
    print(msg)
    try:
        slackbot.post(msg)
    except Exception:
        pass


if __name__ == "__main__":
    run()
