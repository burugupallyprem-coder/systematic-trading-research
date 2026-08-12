"""Improve the SHORT side so it only fires in a genuine downtrend (not bull-market dips
that squeeze). Long side is UNTOUCHED (live config). We compare, from the SAME bars:

  LONG-ONLY   : live config (baseline)
  SYM v1      : current symmetric - short whenever SPY breaks its OR low that morning
  SYM v2      : v1 shorts, but ONLY on days SPY is in a downtrend (prior close < SPY 50d SMA)
  SYM v3      : v2 shorts at HALF risk (crash insurance without the drawdown tax)

The v2 trend gate is causal (uses only prior-day closes) and is the lever that removes the
2021 short-squeeze drawdown. Reports avgDaily / Sharpe / maxDD / TTP-FLEX per variant, plus
year-by-year maxDD so you can see the 4% TTP leash directly. RESEARCH ONLY."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import engine, research
from src.strategies import filters, orb
from src.backtest.ttp_eval import sweep

FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
MA = 50            # SPY trend lookback (trading days) for the short regime gate
SHORT_HALF = 0.5   # v3 short-side risk multiplier


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

    # --- causal SPY downtrend gate: prior close < SMA(50) of prior closes ---
    dates = sorted({d["date"].iloc[0] for _, d in groups})
    spy_close = {dt: float(spy_days[dt]["close"].iloc[-1]) for dt in spy_days}
    closes = [spy_close.get(d, np.nan) for d in dates]
    downtrend = {}
    for i, d in enumerate(dates):
        if i <= MA:
            downtrend[d] = False
            continue
        prior = [c for c in closes[i - MA:i] if not np.isnan(c)]   # strictly prior days
        pc = closes[i - 1]
        downtrend[d] = (len(prior) >= MA // 2 and not np.isnan(pc)
                        and pc < (sum(prior) / len(prior)))

    long_r = {d: 0.0 for d in dates}
    short_r = {d: 0.0 for d in dates}      # raw v1 short R*risk (all spy_dn days)
    for sym, day in groups:
        dt = day["date"].iloc[0]
        rs = ctx["rs"].get(dt, {})
        if sym in filters.top_k_symbols(rs, k) and spy_up.get(dt):
            for t in engine.simulate_day(day, orb.generate(day, {**params, "side": "long"},
                                         {"spy_long_ok": True}), cfg, "orb"):
                long_r[dt] += float(getattr(t, "r_multiple", 0.0)) * risk_pct
        if sym in filters.bottom_k_symbols(rs, k) and spy_dn.get(dt):
            for t in engine.simulate_day(day, orb.generate(day, {**params, "side": "short"}, None), cfg, "orb"):
                short_r[dt] += float(getattr(t, "r_multiple", 0.0)) * risk_pct

    idx = pd.to_datetime(dates)
    L = pd.Series([long_r[d] for d in dates], index=idx)
    S1 = pd.Series([short_r[d] for d in dates], index=idx)
    mask = pd.Series([1.0 if downtrend[d] else 0.0 for d in dates], index=idx)
    S2 = S1 * mask                 # v2: shorts only in downtrend
    return {
        "LONG-ONLY (live)":            L,
        "SYM v1 (short any red open)": L + S1,
        "SYM v2 (short in downtrend)": L + S2,
        "SYM v3 (v2, half-size short)": L + S2 * SHORT_HALF,
    }


def _stats(r):
    r = r.dropna()
    if len(r) < 10 or r.std() == 0: return 0.0, 0.0, 0.0
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return round(r.mean() * 100, 3), round(dd * 100, 1), round(r.mean() / r.std() * np.sqrt(252), 2)


def run():
    cfg = research.load_config()
    variants = _daily(cfg)
    any_r = next(iter(variants.values()))
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[SHORT-V2] {ts} - improving the short side (long side untouched)",
         f"data {any_r.index[0].date()}..{any_r.index[-1].date()} ({len(any_r)} days), SPY trend gate = {MA}d SMA", ""]
    # headline table
    L.append(f"{'VARIANT':<30} avgDaily  Sharpe  maxDD   TTP-FLEX")
    for name, r in variants.items():
        av, dd, sh = _stats(r); fp, _ = sweep(r.values, step=3, **FLEX)
        L.append(f"{name:<30} {av:+.3f}%   {sh:>4}   {dd:>5}%   {fp}%")
    # worst-year drawdown per variant (the 4% TTP leash test)
    L += ["", "Worst single-year maxDD (must stay > -4% to survive TTP FLEX):"]
    for name, r in variants.items():
        worst_y, worst_dd = None, 0.0
        for y in sorted(set(r.index.year)):
            _, dd, _ = _stats(r[r.index.year == y])
            if dd < worst_dd: worst_dd, worst_y = dd, y
        L.append(f"  {name:<30} {worst_dd:>5}%  (in {worst_y})")
    L.append("\nGoal: a variant with HIGHER Sharpe + SHALLOWER worst-year maxDD than long-only. "
             "The trend gate (v2) should remove the bull-year squeezes; half-size (v3) trades a little "
             "crash-upside for a lot less drawdown. RESEARCH ONLY - survivorship + wide-CI caveats apply.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
