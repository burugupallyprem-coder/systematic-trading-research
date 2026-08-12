"""BREAKOUT entry (current live) vs RETEST entry - which funds the TTP account better?

BREAKOUT (live): a 5-min bar closes above the opening-range high -> enter next bar's open,
                 stop = range low. (Enters at the breakout candle - your drawing's tall spike.)
RETEST: after that breakout, rest a limit at the range-high level; enter ONLY if price pulls
        back and touches it (your drawing's point 2), stop = range low. If price runs without
        retesting, the trade is MISSED. If it falls straight through, it's a loss (point 1).

Same universe + same filters (regime + rs_topk) so it's apples-to-apples with the live config.
Reports trades / win% / avg-R, then daily returns at 0.25% risk -> TTP-FLEX pass, median days to
fund, and breach%. Uses the audited causal rules (stop checked before target). RESEARCH ONLY."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import research
from src.strategies import filters
from src.backtest.ttp_eval import sweep

RISK = 0.0025                       # funded size we settled on
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
H = 756


def _simulate(entry_idx, entry_px, stop, recs, flat, rr, slip):
    """R-multiple from entry, stop-before-target, flat-by close. entry_px already slipped."""
    risk = entry_px - stop
    if risk <= 0:
        return None
    target = entry_px + rr * risk
    for k in range(entry_idx, len(recs)):
        t, o, h, l, c = recs[k]
        if t >= flat:
            return (o - entry_px) / risk                 # forced flat at the open
        if l <= stop:
            return (stop - entry_px) / risk              # stop first (conservative)
        if h >= target:
            return (target - entry_px) / risk
    return (recs[-1][4] - entry_px) / risk               # data end


def _day_R(day, params, flat, slip):
    ob = int(params.get("open_bars", 3)); rr = float(params.get("rr", 1.5))
    ch, cm = [int(x) for x in params.get("cutoff_et", "10:30").split(":")]
    recs = [(r["et"].time(), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
            for _, r in day.iterrows()]
    if len(recs) < ob + 2:
        return None, None
    rng = recs[:ob]
    rh = max(b[2] for b in rng); rl = min(b[3] for b in rng)
    if rh <= rl:
        return None, None
    def before_cut(t): return (t.hour, t.minute) < (ch, cm)
    # find first breakout bar i (close > range high, before cutoff)
    i = None
    for k in range(ob, len(recs) - 1):
        if not before_cut(recs[k][0]):
            break
        if recs[k][4] > rh:
            i = k; break
    if i is None:
        return None, None
    # BREAKOUT: enter next bar open
    bo = _simulate(i + 1, recs[i + 1][1] + slip, rl, recs, flat, rr, slip)
    # RETEST: resting limit at range high, fills on first later bar that trades down to it (before cutoff)
    rt = None
    for j in range(i + 1, len(recs)):
        if not before_cut(recs[j][0]):
            break
        if recs[j][3] <= rh:                              # low touched the level -> limit fills
            rt = _simulate(j, rh + slip, rl, recs, flat, rr, slip)
            break
    return bo, rt


def _ttp(dseries):
    r = dseries.values
    fp, _ = sweep(r, step=3, **FLEX)
    outs = []
    for s in range(0, max(1, len(r) - 5), 2):
        eq = 0.0; pd_ = 0; best = -9.0; res = ("never", None)
        for k in range(s, min(len(r), s + H)):
            day = max(r[k], -FLEX["daily_lim"])
            if day >= 0.005: pd_ += 1
            best = max(best, day); eq += day
            if eq <= -FLEX["max_loss"]: res = ("breach", None); break
            if eq >= FLEX["target"] and pd_ >= FLEX["min_pdays"] and best <= FLEX["consist"] * eq:
                res = ("pass", k - s + 1); break
        outs.append(res)
    passed = [d for t, d in outs if t == "pass"]
    med = int(np.median(passed)) if passed else None
    breach = round(100 * sum(1 for t, _ in outs if t == "breach") / len(outs), 1) if outs else 0.0
    return fp, med, breach


def run():
    cfg = research.load_config()
    live = cfg["live"]; params = dict(live["params"]); rs_cfg = cfg["research"]
    k = params.get("rs_topk")
    slip = cfg["costs"]["slippage_cents"] / 100.0
    fh, fm = [int(x) for x in cfg["risk"]["flat_by_et"].split(":")]
    from datetime import time as dtime
    flat = dtime(fh, fm)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bars = data_mod.fetch_bars_hist(cfg["universe"], "2018-01-01", end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    dates = sorted({d["date"].iloc[0] for _, d in groups})
    bo_by = {d: 0.0 for d in dates}; rt_by = {d: 0.0 for d in dates}
    bo_tr = []; rt_tr = []
    for sym, day in groups:
        dt = day["date"].iloc[0]
        rs = ctx["rs"].get(dt, {})
        if sym not in filters.top_k_symbols(rs, k) or not ctx["spy_long_ok"].get(dt, False):
            continue
        bo, rt = _day_R(day, params, flat, slip)
        if bo is not None:
            bo_by[dt] += bo; bo_tr.append(bo)
        if rt is not None:
            rt_by[dt] += rt; rt_tr.append(rt)
    idx = pd.to_datetime(dates)
    bo_s = pd.Series([bo_by[d] * RISK for d in dates], index=idx)
    rt_s = pd.Series([rt_by[d] * RISK for d in dates], index=idx)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def line(name, tr, s):
        n = len(tr); win = round(100 * np.mean([1 for x in tr if x > 0]) / 1, 1) if n else 0.0
        win = round(100 * (np.array(tr) > 0).mean(), 1) if n else 0.0
        avg = round(float(np.mean(tr)), 3) if n else 0.0
        fp, med, br = _ttp(s)
        md = f"{med}d (~{med/21:.1f}mo)" if med else "never"
        return (f"{name:<10} trades {n:>5}  win {win:>4}%  avgR {avg:+.3f}   "
                f"FLEX {fp:>5}%  median {md:<13} breach {br}%")

    L = [f"[ENTRY-COMPARE] {ts} - BREAKOUT (live) vs RETEST entry, at {RISK*100:.2f}% risk",
         f"data {bo_s.index[0].date()}..{bo_s.index[-1].date()} ({len(bo_s)} days), same filters as live", "",
         line("BREAKOUT", bo_tr, bo_s),
         line("RETEST", rt_tr, rt_s), "",
         "Read: RETEST usually shows higher win% / avg-R BUT far fewer trades (it misses breakouts that "
         "never come back - often the biggest winners), so it funds SLOWER. BREAKOUT takes every signal. "
         "If RETEST funds faster AND breaches less, it wins; if it just trades less, breakout stays. RESEARCH ONLY."]
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
