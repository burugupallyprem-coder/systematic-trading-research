"""The real TTP question: at what PER-TRADE RISK does the LIVE long-only ORB keep its worst
year under the 4% FLEX max loss - and does the pass rate hold?

The short side can't fix drawdown (see short_v2); size can. Trades don't change with risk_pct,
so we take the live daily-return series (at the configured 0.5%) and scale it to each risk level
- exact, no re-backtest. Drawdown is measured ADDITIVELY (peak-to-trough of cumulative return),
because TTP's max loss is a STATIC dollar amount from the starting balance, not a % of equity.

Reports, per risk level: avgDaily, Sharpe (invariant to scaling), full-period additive maxDD,
WORST single-year additive maxDD (the 4% leash test), and TTP-FLEX / MAX pass rates. RESEARCH ONLY."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import slackbot
from src.backtest import research
from src.backtest.ttp_eval import live_daily_returns, sweep

RISKS = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]   # per-trade risk %, live is 0.50
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
MAXR = dict(target=0.06, daily_lim=0.01, max_loss=0.03, consist=0.30, min_pdays=0, horizon=60)


def _add_dd(r):
    """Additive peak-to-trough drawdown (fraction), matching TTP's static-dollar max loss."""
    eq = np.cumsum(r); peak = np.maximum.accumulate(eq)
    return float((eq - peak).min())


def _sharpe(r):
    r = r[~np.isnan(r)]
    if len(r) < 10 or r.std() == 0: return 0.0
    return round(r.mean() / r.std() * np.sqrt(252), 2)


def run():
    cfg = research.load_config()
    base = float(cfg["risk"]["risk_pct"])                         # 0.50
    daily = live_daily_returns(cfg, start="2018-01-01")           # long-only live, at base risk
    r0 = daily.values
    yrs = daily.index.year
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L = [f"[RISK-SWEEP] {ts} - LIVE long-only ORB, per-trade risk vs the 4% TTP leash",
         f"data {daily.index[0].date()}..{daily.index[-1].date()} ({len(r0)} days). Sharpe is the same at every "
         f"risk level (scaling cancels) = {_sharpe(r0)}.", "",
         f"{'RISK/trade':<11} avgDaily  fullDD   worstYrDD  TTP-FLEX  TTP-MAX  fits<4%?"]
    for rk in RISKS:
        s = rk / base
        r = r0 * s
        avg = np.nanmean(r) * 100
        full_dd = _add_dd(r) * 100
        worst = min(_add_dd(r[yrs == y]) for y in sorted(set(yrs))) * 100
        fp, _ = sweep(r, step=3, **FLEX); mp, _ = sweep(r, step=3, **MAXR)
        fits = "YES" if worst > -4.0 else "no"
        L.append(f"{rk:>5.2f}%     {avg:+.3f}%  {full_dd:>5.1f}%   {worst:>6.1f}%   {fp:>5}%   {mp:>5}%   {fits}")

    L += ["", "How to read this: find the largest risk whose worstYrDD is still better than -4% - that is the",
          "biggest size that would have survived a 2022-style bear under TTP FLEX. Bigger = faster to the 6%",
          "target but breaches the leash; smaller = safer but slower (and FLEX is UNTIMED, so slower is free).",
          "RESEARCH ONLY - survivorship + wide-CI caveats apply; live paper must confirm before real money."]
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
