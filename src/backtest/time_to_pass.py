"""How long does the $50k TTP FLEX challenge take to PASS at each per-trade risk level?

Time-to-pass is a distribution (depends when you start), so we run the FLEX rule from every
rolling start on the live long-only daily series (scaled to each risk), and record how many
TRADING DAYS until the +6% target is reached (or 'breach' if -4% hits first, or 'never' if the
data window ends). We report the MEDIAN days to pass plus the 25th/75th pctile and fastest,
and convert to calendar months (~21 trading days/month). FLEX is UNTIMED, so slow is allowed.
RESEARCH ONLY - survivorship + wide-CI caveats; live paper must confirm."""

from datetime import datetime, timezone
import numpy as np

from src import slackbot
from src.backtest import research
from src.backtest.ttp_eval import live_daily_returns

RISKS = [0.50, 0.25, 0.15]
FLEX = dict(target=0.06, daily_lim=0.02, max_loss=0.04, consist=0.50, min_pdays=3)
HORIZON = 756          # up to 3 trading-years of runway per start (FLEX has no time limit)


def days_to_pass(r, start_i, target, daily_lim, max_loss, consist, min_pdays):
    eq = 0.0; pdays = 0; best = -9.0
    end = min(len(r), start_i + HORIZON)
    for k in range(start_i, end):
        day = max(r[k], -daily_lim)
        if day >= 0.005: pdays += 1
        best = max(best, day); eq += day
        if eq <= -max_loss:
            return ("breach", None)
        if eq >= target and (not min_pdays or pdays >= min_pdays) and best <= consist * eq:
            return ("pass", k - start_i + 1)
    return ("never", None)


def run():
    cfg = research.load_config()
    base = float(cfg["risk"]["risk_pct"])
    daily = live_daily_returns(cfg, start="2018-01-01")
    r0 = daily.values
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    L = [f"[TIME-TO-PASS] {ts} - $50k TTP FLEX: trading days to reach the +6% target",
         f"data {daily.index[0].date()}..{daily.index[-1].date()} ({len(r0)} days). "
         f"Only starts with full forward runway counted for timing. ~21 trading days = 1 month.", "",
         f"{'RISK/trade':<11} avgDaily   median   (p25..p75)     fastest   pass%   breach%"]
    for rk in RISKS:
        r = r0 * (rk / base)
        # only evaluate starts that have room to either pass or time out (avoid right-censor bias)
        starts = range(0, max(1, len(r) - 5), 2)
        outs = [days_to_pass(r, s, **FLEX) for s in starts]
        passed = [d for tag, d in outs if tag == "pass"]
        n = len(outs)
        pass_pct = round(100 * len(passed) / n, 1) if n else 0.0
        breach_pct = round(100 * sum(1 for tag, _ in outs if tag == "breach") / n, 1) if n else 0.0
        if passed:
            med = int(np.median(passed)); p25 = int(np.percentile(passed, 25))
            p75 = int(np.percentile(passed, 75)); fast = int(min(passed))
            avg = np.nanmean(r) * 100
            L.append(f"{rk:>5.2f}%     {avg:+.3f}%   {med:>4}d    ({p25}..{p75})d   {fast:>4}d    "
                     f"{pass_pct:>5}%   {breach_pct:>5}%")
            L.append(f"{'':>11}           ~{med/21:.1f} months   (~{p25/21:.1f}..{p75/21:.1f} mo)")
        else:
            L.append(f"{rk:>5.2f}%     never reached +6% within {HORIZON}d window")
    L += ["", "Read: 'median' = a typical run; half pass faster, half slower. Lower risk ~= proportionally more",
          "days (the +6% target is fixed, daily gain is smaller). breach% = starts that hit -4% and FAILED",
          "before reaching target - that is the number the 4% leash actually costs you. RESEARCH ONLY."]
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass


if __name__ == "__main__":
    run()
