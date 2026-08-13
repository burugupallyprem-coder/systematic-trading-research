"""Is there ANY real causal edge left in the ORB - or is it a dead end? (anti-overfit)

After fixing the SPY-regime look-ahead, we ask one honest question with a SMALL, PRE-DECLARED grid
judged ONCE on the untouched out-of-sample window (< 2024). No searching, no cherry-picking: we
report EVERY config's OOS numbers and give a blunt verdict.

Pre-declared ablation (all long-only breakout; each row adds/removes ONE thing):
  1. base breakout        - no filters at all (the rawest ORB)
  2. + regime gate (causal) - only the fixed causal SPY-break timing gate
  3. + relative-strength    - only trade the strongest-5 vs SPY
  4. + vol floor            - only skip dead-tape (narrow opening range) days
  5. LIVE (all three)       - the current live config, now causal

Judge = OOS pre-2024 (never used for tuning). A config is only interesting if OOS expectancy is
clearly positive AND OOS Sharpe is respectable AND its worst drawdown is survivable. Otherwise the
honest call is: dead end, stop polishing. RESEARCH ONLY."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import engine, research
from src.strategies import filters, orb
from src.backtest.ttp_eval import sweep

RISK = 0.0025
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3, horizon=252)
BASE = {"open_bars": 3, "cutoff_et": "10:30", "rr": 1.5, "side": "long", "max_risk_frac": 0.02}

CONFIGS = {
    "1 base breakout":        dict(regime=False, rs=None, vol=None),
    "2 + regime (causal)":    dict(regime=True,  rs=None, vol=None),
    "3 + rel-strength top5":  dict(regime=False, rs=5,    vol=None),
    "4 + vol floor":          dict(regime=False, rs=None, vol=0.004),
    "5 LIVE (all, causal)":   dict(regime=True,  rs=5,    vol=0.004),
}


def _daily(cfg, groups, ctx, conf):
    params = {**BASE, "regime_filter": conf["regime"], "min_or_width_frac": conf["vol"]}
    k = conf["rs"]
    dates = sorted({d["date"].iloc[0] for _, d in groups})
    r_by = {d: 0.0 for d in dates}
    for sym, day in groups:
        dt = day["date"].iloc[0]
        if k and sym not in filters.top_k_symbols(ctx["rs"].get(dt, {}), k):
            continue
        c = {"prev_close": ctx.get("prev_close", {}).get(sym, {}).get(dt)}
        if conf["regime"]:
            c["spy_long_ok"] = ctx["spy_long_ok"].get(dt, False)
            c["spy_break_min"] = ctx.get("spy_break_min", {}).get(dt)
        for t in engine.simulate_day(day, orb.generate(day, params, c), cfg, "orb"):
            r_by[dt] += float(getattr(t, "r_multiple", 0.0)) * RISK
    return pd.Series([r_by[d] for d in dates], index=pd.to_datetime(dates))


def _stats(r):
    r = r.dropna()
    if len(r) < 10 or r.std() == 0:
        return dict(n=len(r), avg=0.0, sh=0.0, dd=0.0, flex=0.0)
    eq = np.cumsum(r.values); dd = float((eq - np.maximum.accumulate(eq)).min())
    fp, _ = sweep(r.values, step=3, **FLEX)
    return dict(n=int((r != 0).sum()), avg=round(r.mean() * 100, 4),
                sh=round(r.mean() / r.std() * np.sqrt(252), 2), dd=round(dd * 100, 1), flex=fp)


def run():
    cfg = research.load_config()
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bars = data_mod.fetch_bars_hist(cfg["universe"], "2018-01-01", end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[CAUSAL-EDGE] {ts} - ORB filter ablation, judged ONCE on OOS pre-2024 (causal engine)",
         "sizing 0.25%/trade. OOS = the untouched judge. All configs reported (no cherry-picking).", "",
         f"{'CONFIG':<22} {'OOS avgDaily':>12} {'Sharpe':>7} {'maxDD':>7} {'FLEX':>6} | {'full-period Sharpe':>17}"]
    verdicts = []
    for name, conf in CONFIGS.items():
        d = _daily(cfg, groups, ctx, conf)
        oos = _stats(d[d.index.year < 2024]); full = _stats(d)
        L.append(f"{name:<22} {oos['avg']:>11.4f}% {oos['sh']:>7} {oos['dd']:>6}% {oos['flex']:>5}% | {full['sh']:>17}")
        good = oos["sh"] >= 0.8 and oos["avg"] > 0 and oos["dd"] > -8.0
        verdicts.append((name, good, oos))
    winners = [v for v in verdicts if v[1]]
    L += ["", "Bar for 'interesting': OOS Sharpe >= 0.8, positive expectancy, worst DD better than -8%."]
    if winners:
        L.append("=> POSSIBLE causal edge: " + ", ".join(w[0].strip() for w in winners) +
                 " - confirm with walk-forward + a fresh forward window before trusting.")
    else:
        L.append("=> DEAD END (honest): no config clears the bar out-of-sample. The ORB's edge did not "
                 "survive the causality fix. Stop polishing this family; let live paper be the judge.")
    L.append("RESEARCH ONLY - survivorship + wide-CI caveats apply; OOS judged once, no search.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
