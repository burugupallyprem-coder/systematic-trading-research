"""Does holding the ORB breakout for DAYS (swing) beat exiting intraday, under TTP FLEX?

Same entries + same live filters. Three exit regimes on the identical breakout signals:
  INTRADAY (live): exit at 1.5R or end of day (no overnight).
  SWING-A        : target 3R, stop = range low, hold up to 5 trading days (overnight allowed).
  SWING-B        : target 5R, hold up to 10 trading days.

Overnight gaps are modeled honestly: if a day OPENS through the stop, you exit at that open
(often WORSE than the stop) - that is the gap risk you can't control on a held position. Reports
trades / win% / avg-R / avg hold-days, then daily returns at 0.25% risk -> TTP-FLEX pass, median
days to fund, breach%. RESEARCH ONLY - the ORB's multi-day edge is UNVALIDATED; this tests it."""

from datetime import datetime, timezone, time as dtime
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import research
from src.strategies import filters
from src.backtest.ttp_eval import sweep

RISK = 0.0025
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
H = 756


def _entry(day, params, slip):
    """Find the ORB breakout entry (same as live). Returns (entry_idx, entry_px, stop, recs) or None."""
    ob = int(params.get("open_bars", 3))
    ch, cm = [int(x) for x in params.get("cutoff_et", "10:30").split(":")]
    recs = [(r["et"].time(), float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]))
            for _, r in day.iterrows()]
    if len(recs) < ob + 2:
        return None
    rh = max(b[2] for b in recs[:ob]); rl = min(b[3] for b in recs[:ob])
    if rh <= rl:
        return None
    for k in range(ob, len(recs) - 1):
        if (recs[k][0].hour, recs[k][0].minute) >= (ch, cm):
            break
        if recs[k][4] > rh:
            return (k + 1, recs[k + 1][1] + slip, rl, recs)
    return None


def _intraday_R(entry_idx, entry_px, stop, recs, flat, rr, slip):
    risk = entry_px - stop
    if risk <= 0:
        return None
    target = entry_px + rr * risk
    for k in range(entry_idx, len(recs)):
        t, o, h, l, c = recs[k]
        if t >= flat:
            return (o - entry_px) / risk
        if l <= stop:
            return (stop - entry_px) / risk
        if h >= target:
            return (target - entry_px) / risk
    return (recs[-1][4] - entry_px) / risk


def _swing_R(entry_idx, entry_px, stop, recs, dates_after, dbars, rr, max_days, slip):
    """Day 0 intraday first (5-min), then carry overnight across daily bars with gap-aware exits."""
    risk = entry_px - stop
    if risk <= 0:
        return None, 0
    target = entry_px + rr * risk
    # day 0 intraday (no forced flat - swing holds)
    for k in range(entry_idx, len(recs)):
        t, o, h, l, c = recs[k]
        if l <= stop:
            return (stop - entry_px) / risk, 0
        if h >= target:
            return (target - entry_px) / risk, 0
    # carry overnight
    held = 0
    for dt in dates_after[:max_days]:
        o, h, l, c = dbars[dt]
        held += 1
        if o <= stop:                                  # gap DOWN through stop -> worse fill
            return (o - slip - entry_px) / risk, held
        if o >= target:                                # gap UP through target
            return (o - slip - entry_px) / risk, held
        if l <= stop:
            return (stop - entry_px) / risk, held
        if h >= target:
            return (target - entry_px) / risk, held
    # time exit at last held day's close
    if held > 0:
        c = dbars[dates_after[held - 1]][3]
        return (c - entry_px) / risk, held
    return (recs[-1][4] - entry_px) / risk, 0


def _ttp(dseries):
    r = dseries.values
    fp, _ = sweep(r, step=3, **FLEX)
    outs = []
    for s in range(0, max(1, len(r) - 5), 2):
        eq = 0.0; pd_ = 0; best = -9.0; res = ("never", None)
        for k in range(s, min(len(r), s + H)):
            dd = max(r[k], -FLEX["daily_lim"])
            if dd >= 0.005: pd_ += 1
            best = max(best, dd); eq += dd
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
    params = dict(cfg["live"]["params"]); k = params.get("rs_topk")
    slip = cfg["costs"]["slippage_cents"] / 100.0
    fh, fm = [int(x) for x in cfg["risk"]["flat_by_et"].split(":")]
    flat = dtime(fh, fm)
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bars = data_mod.fetch_bars_hist(cfg["universe"], "2018-01-01", end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    # per-symbol daily OHLC + ordered dates
    dbars = {}; sdates = {}
    for sym, day in groups:
        dt = day["date"].iloc[0]
        dbars.setdefault(sym, {})[dt] = (float(day["open"].iloc[0]), float(day["high"].max()),
                                         float(day["low"].min()), float(day["close"].iloc[-1]))
    for sym in dbars:
        sdates[sym] = sorted(dbars[sym].keys())

    dates = sorted({d["date"].iloc[0] for _, d in groups})
    variants = {"INTRADAY (live)": {}, "SWING-A 3R/5d": {}, "SWING-B 5R/10d": {}}
    holds = {"SWING-A 3R/5d": [], "SWING-B 5R/10d": []}
    trs = {v: [] for v in variants}
    for v in variants:
        variants[v] = {d: 0.0 for d in dates}
    for sym, day in groups:
        dt = day["date"].iloc[0]
        rs = ctx["rs"].get(dt, {})
        if sym not in filters.top_k_symbols(rs, k) or not ctx["spy_long_ok"].get(dt, False):
            continue
        e = _entry(day, params, slip)
        if not e:
            continue
        eidx, epx, stop, recs = e
        after = [d for d in sdates[sym] if d > dt]
        r_intra = _intraday_R(eidx, epx, stop, recs, flat, float(params.get("rr", 1.5)), slip)
        r_a, h_a = _swing_R(eidx, epx, stop, recs, after, dbars[sym], 3.0, 5, slip)
        r_b, h_b = _swing_R(eidx, epx, stop, recs, after, dbars[sym], 5.0, 10, slip)
        for v, rv in [("INTRADAY (live)", r_intra), ("SWING-A 3R/5d", r_a), ("SWING-B 5R/10d", r_b)]:
            if rv is not None:
                variants[v][dt] += rv; trs[v].append(rv)
        if r_a is not None: holds["SWING-A 3R/5d"].append(h_a)
        if r_b is not None: holds["SWING-B 5R/10d"].append(h_b)

    idx = pd.to_datetime(dates)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[SWING-TEST] {ts} - ORB held for DAYS vs exited intraday, at {RISK*100:.2f}% risk",
         f"data {idx[0].date()}..{idx[-1].date()} ({len(dates)} days), same entries + filters as live", "",
         f"{'VARIANT':<17} trades  win%   avgR    avgHold  FLEX    median        breach"]
    for v in variants:
        s = pd.Series([variants[v][d] * RISK for d in dates], index=idx)
        tr = np.array(trs[v]); n = len(tr)
        win = round(100 * (tr > 0).mean(), 1) if n else 0.0
        avg = round(float(tr.mean()), 3) if n else 0.0
        hold = round(float(np.mean(holds[v])), 1) if v in holds and holds[v] else 0.0
        fp, med, br = _ttp(s)
        md = f"{med}d (~{med/21:.1f}mo)" if med else "never"
        L.append(f"{v:<17} {n:>5}  {win:>4}%  {avg:+.3f}   {hold:>4}d   {fp:>5}%  {md:<13} {br}%")
    L.append("\nRead: for SWING to be worth it, a swing row must fund FASTER (lower median) than intraday WITHOUT "
             "a higher breach%. Remember breach% here UNDERSTATES swing risk - the daily-loss cap assumes you can "
             "stop intraday, but overnight gaps bypass it. The ORB's multi-day edge is unproven. RESEARCH ONLY.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
