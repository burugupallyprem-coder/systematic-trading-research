"""Daily time-series MOMENTUM on the stock universe - a FRESH, pre-registered hypothesis.

Motivation: the intraday ORB had no causal edge, but DAILY trend/momentum is causally validated on
gold (mgc_prop, Sharpe ~0.9). Does the same idea work on equities? This tests it cleanly, and - the
key lesson from the ORB look-ahead - it is judged against BUY-AND-HOLD, because on a survivor
mega-cap universe a long-only trend system can just be capturing the bull run (beta), not alpha.

PRE-REGISTRATION (declared before results):
  Strategy : per stock, long when EMA_fast > EMA_slow (daily close), else flat. Signal SHIFTED one
             day (trade on yesterday's signal = causal). Equal-weight across all long names; daily
             rebalance; turnover charged at cost_bps. Long-only (no shorts/borrow).
  Grid     : ema_fast in [20, 50] x ema_slow in [100, 200] = 4 combos (small, motivated).
  Select   : best Sharpe on TRAIN (<= 2021-12). Judge ONCE on VAL (2022-01 -> now, incl. the 2022 bear).
  Benchmark: equal-weight BUY-AND-HOLD of the same names. Momentum must beat it on Sharpe OR cut the
             drawdown materially - otherwise it is beta, not edge.
  Gate     : VAL Sharpe >= 0.8 AND edge broad across the grid (not one fluke) AND it improves on B&H
             (higher Sharpe or much shallower maxDD, especially 2022). Else: not an edge.
  Caveat   : survivorship bias (today's mega-caps) inflates BOTH momentum and B&H - read as a ceiling.
Run: python -m src.backtest.stock_momentum  (daily bars; causal by construction). RESEARCH ONLY.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import research

GRID = [(20, 100), (20, 200), (50, 100), (50, 200)]
COST_BPS = 0.0005          # ~5 bps per unit of weight turned over (commission + slippage)
TRAIN_END = "2021-12-31"
VAL_START = "2022-01-01"


def _metrics(r):
    r = r.dropna()
    if len(r) < 30 or r.std() == 0:
        return dict(sharpe=0.0, cagr=0.0, maxdd=0.0, n=len(r))
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1)
    return dict(sharpe=round(r.mean() / r.std() * np.sqrt(252), 2),
                cagr=round(cagr * 100, 1), maxdd=round(dd * 100, 1), n=len(r))


def _portfolio(close, fast, slow):
    """Causal long-only trend portfolio daily returns, net of turnover cost."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    sig = (ema_f > ema_s).astype(float).shift(1).fillna(0.0)   # SHIFT = causal
    ret = close.pct_change()
    n_long = sig.sum(axis=1).replace(0, np.nan)
    w = sig.div(n_long, axis=0).fillna(0.0)                    # equal weight across long names
    gross = (w * ret).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    return gross - turnover * COST_BPS


def _buy_hold(close):
    ret = close.pct_change()
    return ret.mean(axis=1)                                    # equal-weight, always invested


def run():
    cfg = research.load_config()
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    tradables = [s for s in cfg["universe"] if s not in ("SPY", "QQQ")]
    print(f"fetching DAILY bars for {len(tradables)} names...", flush=True)
    bars = data_mod.fetch_bars_hist(tradables, "2016-01-01", end, timeframe="1Day", feed="sip")
    if bars.empty:
        slackbot.post("[STOCK-MOMENTUM] no data"); return
    bars["date"] = pd.to_datetime(bars["ts"]).dt.tz_convert("America/New_York").dt.date
    close = bars.pivot_table(index="date", columns="symbol", values="close").sort_index()
    close.index = pd.to_datetime(close.index)

    tr = close.index <= pd.Timestamp(TRAIN_END)
    va = close.index >= pd.Timestamp(VAL_START)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[STOCK-MOMENTUM] {ts} - daily trend on {len(tradables)} stocks, CAUSAL, judged once on OOS",
         f"data {close.index[0].date()}..{close.index[-1].date()} | train<= {TRAIN_END} | val>= {VAL_START}", ""]

    # all configs: train + val Sharpe (broad-vs-fluke check, no cherry-picking)
    L.append(f"{'EMA fast/slow':<14} {'TRAIN Sharpe':>12} {'VAL Sharpe':>11} {'VAL CAGR':>9} {'VAL maxDD':>10}")
    scored = []
    for f, s in GRID:
        p = _portfolio(close, f, s)
        mtr, mva = _metrics(p[tr]), _metrics(p[va])
        scored.append(((f, s), mtr, mva, p))
        L.append(f"{f}/{s:<10} {mtr['sharpe']:>12} {mva['sharpe']:>11} {mva['cagr']:>8}% {mva['maxdd']:>9}%")

    # select by TRAIN, judge selected on VAL + per-year walk-forward
    best = max(scored, key=lambda x: x[1]["sharpe"])
    (bf, bs), _, bmva, bp = best
    L += ["", f"Selected by TRAIN: EMA {bf}/{bs}.  VAL -> Sharpe {bmva['sharpe']}, CAGR {bmva['cagr']}%, maxDD {bmva['maxdd']}%"]
    wf = []
    for y in sorted({d.year for d in close.index[va]}):
        m = _metrics(bp[(close.index.year == y) & va])
        if m["n"] > 30: wf.append(f"{y}:{m['sharpe']}")
    L.append("VAL by year (Sharpe): " + "  ".join(wf))

    # THE honest benchmark: buy-and-hold the same names
    bh = _buy_hold(close)
    bh_tr, bh_va = _metrics(bh[tr]), _metrics(bh[va])
    L += ["", "=== vs BUY-AND-HOLD (is it alpha or just beta?) ===",
          f"  buy&hold  VAL: Sharpe {bh_va['sharpe']}, CAGR {bh_va['cagr']}%, maxDD {bh_va['maxdd']}%",
          f"  momentum  VAL: Sharpe {bmva['sharpe']}, CAGR {bmva['cagr']}%, maxDD {bmva['maxdd']}%"]

    # verdict
    broad = sum(1 for _, _, mva, _ in scored if mva["sharpe"] >= 0.6) >= 3
    beats = (bmva["sharpe"] > bh_va["sharpe"] + 0.15) or (bmva["maxdd"] > bh_va["maxdd"] + 8)
    if bmva["sharpe"] >= 0.8 and broad and beats:
        v = ("PROMISING - clears the bar, broad across the grid, and beats buy&hold. NEXT: fresh "
             "forward window + deflated-Sharpe before trusting. Do NOT deploy yet.")
    elif bmva["sharpe"] >= 0.5 and beats:
        v = ("MARGINAL - some edge and improves on buy&hold, but below a confident bar. Worth a "
             "forward-only watch, not capital.")
    else:
        v = ("NOT AN EDGE - it does not beat buy&hold on a risk-adjusted basis (i.e. it is beta, not "
             "alpha) or the grid is a fluke. Honest dead end, same discipline as the ORB.")
    L += ["", f"VERDICT: {v}",
          "CAVEAT: survivorship bias inflates BOTH lines; this is a ceiling. RESEARCH ONLY - not deployed."]
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
