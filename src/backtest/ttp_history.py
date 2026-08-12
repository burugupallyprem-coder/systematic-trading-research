"""Multi-year (up to ~10yr) history of the LIVE filtered ORB config under Trade The Pool
rules. Year-by-year, regime windows (COVID crash / 2022 bear / best-worst), and the honest
IN-SAMPLE vs OUT-OF-SAMPLE split (params were locked on 2024-2025, so pre-2024 is data the
strategy NEVER saw - the real test). Needs Alpaca bars (CI).

CAVEAT baked into the report: the universe is TODAY's liquid megacaps, so a long backtest is
SURVIVORSHIP-BIASED (we're testing known winners). Intraday ORB is less exposed than buy-hold,
but the bias inflates the pre-2024 numbers - read them as a ceiling."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import slackbot
from src.backtest import research
from src.backtest.ttp_eval import live_daily_returns, sweep

ANN = 252
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
MAXA = dict(target=0.06, daily_lim=0.01, max_loss=0.03, consist=0.30, min_pdays=0, horizon=60)


def _stats(r):
    r = r.dropna()
    if len(r) < 20 or r.std() == 0:
        return 0.0, 0.0, 0.0
    sh = r.mean() / r.std() * np.sqrt(ANN)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return round(sh, 2), round(dd * 100, 1), round(r.mean() * 100, 3)


def _flex_pass(r):
    if len(r) < 30:
        return None
    pr, _ = sweep(r.values, step=3, **FLEX)
    return pr


def run():
    cfg = research.load_config()
    daily = live_daily_returns(cfg, start="2016-01-01")           # ~10yr, whatever Alpaca gives
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    yrs = sorted(set(daily.index.year))
    L = [f"[TTP-HISTORY] {ts} - LIVE filtered ORB, {daily.index[0].date()}..{daily.index[-1].date()} "
         f"({len(daily)} days across {len(yrs)} years)", "",
         "YEAR   avgDaily   Sharpe   maxDD   TTP-FLEX pass"]
    for y in yrs:
        r = daily[daily.index.year == y]
        sh, dd, av = _stats(r); fp = _flex_pass(r)
        L.append(f"{y}   {av:+.3f}%   {sh:>5}   {dd:>6}%   {('%.0f%%' % fp) if fp is not None else '  -'}")

    def win(a, b, label):
        seg = daily[(daily.index >= a) & (daily.index <= b)]
        sh, dd, av = _stats(seg)
        L.append(f"  {label:22} {seg.index.min().date() if len(seg) else '-'}..{seg.index.max().date() if len(seg) else '-'}: "
                 f"avgDaily {av:+.3f}%  Sharpe {sh}  maxDD {dd}%  ({len(seg)}d)")

    L += ["", "=== Regime stress windows ==="]
    win("2020-02-15", "2020-04-30", "COVID crash")
    win("2020-05-01", "2021-12-31", "COVID recovery/bull")
    win("2022-01-01", "2022-10-31", "2022 bear")

    L += ["", "=== IN-SAMPLE vs OUT-OF-SAMPLE (params locked on 2024-2025) ==="]
    oos_pre = daily[daily.index < "2024-01-01"]
    insamp  = daily[(daily.index >= "2024-01-01") & (daily.index < "2026-01-01")]
    oos_post = daily[daily.index >= "2026-01-01"]
    for seg, name in [(oos_pre, "OOS pre-2024 (NEVER seen - the real test)"),
                      (insamp,  "IN-SAMPLE 2024-2025 (params tuned here)"),
                      (oos_post, "OOS 2026 (validation-forward)")]:
        sh, dd, av = _stats(seg); fp = _flex_pass(seg)
        L.append(f"  {name:44} avgDaily {av:+.3f}%  Sharpe {sh}  maxDD {dd}%  "
                 f"FLEX {('%.0f%%' % fp) if fp is not None else '-'}  ({len(seg)}d)")

    full_flex, n = sweep(daily.values, step=3, **FLEX)
    full_max, _ = sweep(daily.values, step=3, **MAXA)
    L += ["", f"=== TTP overall (full period, {n} rolling starts) ===",
          f"  FLEX {full_flex}%   MAX {full_max}%",
          "", "CAVEATS: (1) universe = TODAY's megacaps -> survivorship bias inflates pre-2024 (read as ceiling). "
          "(2) rolling starts overlap -> wide true CI. (3) RESEARCH ONLY. The pre-2024 OOS row is the honest "
          "signal: if the edge holds there, it is real; if it collapses, 70% was in-sample optimism."]
    out = "\n".join(L)
    print(out)
    try:
        slackbot.post(out)
    except Exception:
        pass


if __name__ == "__main__":
    run()
