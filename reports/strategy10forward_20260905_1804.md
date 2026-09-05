# Strategy #10 FORWARD trial - 2026-09-05 18:03 UTC

RESEARCH ONLY - frozen config, measured only on data >= lock date. No re-search, no moving goalposts (see PRE_REGISTRATION_STRATEGY10.md).
frozen: max_trades_day=2, trail_lookback=20, trend_filter=False
lock date: 2026-07-23 - scored window 2026-07-23 -> 2026-09-04

## VERDICT: FAIL (expectancy 0.019R < 0.05R; PF 1.14 < 1.15)
- forward: 464 trades, win 37.1%, 0.019R ($6.26/trade), PF 1.14, 1/1 quarters+, maxDD $4,062.59
- bootstrap 90% CI [-0.088,+0.133]R P(>0)=61% -> CI includes 0
- thresholds: interim 100, verdict 300
