"""Does 'scaling on house money' get you FUNDED SOONER without raising the knockout rate?

Idea: start small (safe), and once you've banked a profit cushion, size up - because your
distance to the -4% floor has grown, bigger size can't breach until you give back the whole
buffer. We simulate every rolling start on the live long-only daily R-series (COVID included
once the feed fix loads it) and compare, at each starting risk:

  FLAT     - hold the starting risk the whole way
  RATCHET  - starting risk until +2% banked, then 1.5x; +4% banked, then 2.0x (capped 0.60%)

Reports median trading-days-to-pass, pass%, and breach% (the 4% leash cost) for FLAT vs RATCHET.
The ratchet thresholds are pre-declared (a sensible rule, NOT tuned). RESEARCH ONLY - survivorship
+ overlapping-starts + wide-CI caveats; live paper must confirm before real money."""

from datetime import datetime, timezone
import numpy as np

from src import slackbot
from src.backtest import research
from src.backtest.ttp_eval import live_daily_returns

STARTS = [0.0020, 0.0025]          # starting per-trade risk (fraction): 0.20%, 0.25%
TARGET, MAXLOSS, DAILYCAP = 0.06, 0.04, 0.02
MINPD, CONSIST, HORIZON = 3, 0.50, 756
CAP = 0.0060                       # never risk more than 0.60% even fully cushioned


def _risk(eq, r0, ratchet):
    if not ratchet:
        return r0
    if eq >= 0.04:
        return min(r0 * 2.0, CAP)
    if eq >= 0.02:
        return min(r0 * 1.5, CAP)
    return r0


def _run_start(dR, s, r0, ratchet):
    eq = 0.0; pdays = 0; best = -9.0
    for k in range(s, min(len(dR), s + HORIZON)):
        day = dR[k] * _risk(eq, r0, ratchet)
        day = max(day, -DAILYCAP)                 # daily pause caps the loss
        if day >= 0.005: pdays += 1
        best = max(best, day); eq += day
        if eq <= -MAXLOSS:
            return ("breach", None)
        if eq >= TARGET and pdays >= MINPD and best <= CONSIST * eq:
            return ("pass", k - s + 1)
    return ("never", None)


def _agg(dR, r0, ratchet):
    outs = [_run_start(dR, s, r0, ratchet) for s in range(0, max(1, len(dR) - 5), 2)]
    passed = [d for tag, d in outs if tag == "pass"]
    n = len(outs)
    med = int(np.median(passed)) if passed else None
    p25 = int(np.percentile(passed, 25)) if passed else None
    p75 = int(np.percentile(passed, 75)) if passed else None
    return dict(med=med, p25=p25, p75=p75,
                passpct=round(100 * len(passed) / n, 1) if n else 0.0,
                breachpct=round(100 * sum(1 for t, _ in outs if t == "breach") / n, 1) if n else 0.0)


def run():
    cfg = research.load_config()
    base = float(cfg["risk"]["risk_pct"]) / 100.0
    daily = live_daily_returns(cfg, start="2018-01-01")
    dR = (daily.values / base)                    # per-day return in R units (pre-risk)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    covid = "YES" if daily.index[0].year <= 2020 and daily.index[0].month <= 3 else f"NO (starts {daily.index[0].date()})"
    L = [f"[HOUSE-MONEY] {ts} - scaling on house money vs flat sizing ($50k TTP FLEX)",
         f"data {daily.index[0].date()}..{daily.index[-1].date()} ({len(dR)} days) | COVID crash in-sample? {covid}",
         "ratchet: start risk -> +2% banked: 1.5x -> +4% banked: 2.0x (cap 0.60%). ~21 td = 1 month.", "",
         f"{'START':<7} {'MODE':<9} median   (p25..p75)     pass%   breach%"]
    for r0 in STARTS:
        for ratchet, name in [(False, "FLAT"), (True, "RATCHET")]:
            a = _agg(dR, r0, ratchet)
            md = f"{a['med']}d (~{a['med']/21:.1f}mo)" if a['med'] else "never"
            span = f"({a['p25']}..{a['p75']})d" if a['med'] else ""
            L.append(f"{r0*100:>4.2f}%  {name:<9} {md:<14} {span:<14} {a['passpct']:>5}%  {a['breachpct']:>5}%")
        L.append("")
    L.append("Read: RATCHET should cut median days vs FLAT at the same start risk. The number that must NOT "
             "get worse is breach% - if the ratchet funds faster AND keeps breach% ~= flat, sizing on house "
             "money is a real free speedup. If breach% rises, the ramp is too aggressive. RESEARCH ONLY.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
