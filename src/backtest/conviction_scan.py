"""Does a 'golden moment' exist? Test whether any PRE-ENTRY feature separates high-expectancy
ORB trades from low ones - and whether that separation HOLDS OUT-OF-SAMPLE. Only if it does can
we honestly size UP on the best setups (and down on the rest) while keeping AVERAGE risk under the
4% TTP leash. If the gradient is in-sample only, cherry-picking is hindsight and we must NOT use it.

Pre-registered causal features (all known at/before entry, no look-ahead):
  or_width  - opening-range width as % of price (volatility of the setup)
  rs        - relative strength vs SPY that morning (how strong the name is)
  early     - the name's own opening-drive return over the opening range
  rank      - 1..5 relative-strength rank among the traded names (1 = strongest)

For each feature we bucket trades into terciles and report avg-R / win% / n, split IN-SAMPLE
(<2024) vs OUT-OF-SAMPLE (>=2024). A feature is USABLE only if top beats bottom in BOTH splits.
RESEARCH ONLY - survivorship + multiple-testing caveats; a signal here is a hypothesis, not a green light."""

from datetime import datetime, timezone
import numpy as np
import pandas as pd

from src import data as data_mod, slackbot
from src.backtest import engine, research
from src.strategies import filters, orb


def collect(cfg, start="2018-01-01"):
    live = cfg["live"]; params = dict(live["params"]); rs_cfg = cfg["research"]
    ob = int(rs_cfg.get("regime_open_bars", 3)); k = params.get("rs_topk")
    end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    bars = data_mod.fetch_bars_hist(cfg["universe"], start, end,
                               timeframe=cfg["backtest"]["timeframe"], feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    groups = research.day_groups(bars)
    ctx = research.build_context(groups, cfg)
    rows = []
    for sym, day in groups:
        dt = day["date"].iloc[0]
        rsmap = ctx["rs"].get(dt, {})
        allowed = filters.top_k_symbols(rsmap, k)
        if sym not in allowed:
            continue
        c = {"spy_long_ok": ctx["spy_long_ok"].get(dt, False),
             "prev_close": ctx.get("prev_close", {}).get(sym, {}).get(dt)}
        sigs = orb.generate(day, params, c)
        if not sigs:
            continue
        trades = engine.simulate_day(day, sigs, cfg, "orb")
        if not trades:
            continue
        rmult = sum(float(getattr(t, "r_multiple", 0.0)) for t in trades)
        ranked = sorted(allowed, key=lambda s: rsmap.get(s, -9.0), reverse=True)
        rank = ranked.index(sym) + 1 if sym in ranked else 0
        rows.append(dict(year=dt.year, r=rmult,
                         or_width=filters.or_width_frac(day, ob),
                         rs=float(rsmap.get(sym, 0.0)),
                         early=filters.early_return(day, ob),
                         rank=rank))
    return pd.DataFrame(rows)


def _bucket(df, col):
    """Tercile avg-R / win% / n for a feature within a given trade set."""
    d = df.dropna(subset=[col])
    if len(d) < 15:
        return None
    try:
        q = pd.qcut(d[col], 3, labels=["low", "mid", "high"], duplicates="drop")
    except ValueError:
        return None
    out = {}
    for lab in ["low", "mid", "high"]:
        g = d[q == lab]
        if len(g):
            out[lab] = (round(g["r"].mean(), 3), round(100 * (g["r"] > 0).mean(), 1), len(g))
    return out


def run():
    cfg = research.load_config()
    df = collect(cfg)
    IS = df[df["year"] < 2024]; OOS = df[df["year"] >= 2024]
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[CONVICTION] {ts} - is there a pre-entry 'golden moment' signal? (held out of sample)",
         f"trades: {len(df)} total | IS(<2024) {len(IS)} | OOS(>=2024) {len(OOS)} | "
         f"baseline avg-R all={df['r'].mean():.3f}", ""]
    for col in ["or_width", "rs", "early"]:
        L.append(f"--- {col} (tercile avg-R / win% / n) ---")
        verdict = "?"
        for name, part in [("IS ", IS), ("OOS", OOS)]:
            b = _bucket(part, col)
            if not b:
                L.append(f"  {name}: (too few trades)"); continue
            seg = "  ".join(f"{lab}:{b[lab][0]:+.3f}/{b[lab][1]}%/{b[lab][2]}" for lab in b)
            L.append(f"  {name}: {seg}")
        bi, bo = _bucket(IS, col), _bucket(OOS, col)
        if bi and bo and all(x in bi and x in bo for x in ("low", "mid", "high")):
            mono = lambda b: b["low"][0] < b["mid"][0] < b["high"][0]           # strict low<mid<high
            up = lambda b: b["high"][0] > b["low"][0]
            big = lambda b: (b["high"][0] - b["low"][0]) >= 0.05                # economically meaningful
            if mono(bi) and mono(bo) and big(bi) and big(bo):
                verdict = "USABLE (monotone high>mid>low in BOTH splits, sizable)"
            elif up(bi) and up(bo):
                verdict = "WEAK (top>bottom both, but not monotone - likely noise, do NOT size on it)"
            elif up(bi) != up(bo):
                verdict = "NOISE (gradient flips out-of-sample)"
            else:
                verdict = "inverse (bottom>top in both - the edge is in the WEAK setups?)"
        L.append(f"  => {verdict}\n")
    # rank 1..5 expectancy (the most actionable cherry-pick lever)
    L.append("--- relative-strength RANK 1..5 (avg-R / win% / n) ---")
    for name, part in [("IS ", IS), ("OOS", OOS)]:
        seg = []
        for rk in [1, 2, 3, 4, 5]:
            g = part[part["rank"] == rk]
            if len(g): seg.append(f"#{rk}:{g['r'].mean():+.3f}/{len(g)}")
        L.append(f"  {name}: " + "  ".join(seg))
    L.append("\nRead: a feature is only worth sizing on if its top bucket beats its bottom bucket IN BOTH "
             "IS and OOS. If so, next step is conviction sizing (more risk on top setups, less on rest) with "
             "AVERAGE risk under the 4% leash - THEN re-check drawdown, because concentration can deepen it. "
             "RESEARCH ONLY.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
