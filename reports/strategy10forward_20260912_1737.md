# Strategy #10 FORWARD trial - 2026-09-12 17:36 UTC

RESEARCH ONLY - frozen config, measured only on data >= lock date. No re-search, no moving goalposts (see PRE_REGISTRATION_STRATEGY10.md).
frozen: max_trades_day=2, trail_lookback=20, trend_filter=False
lock date: 2026-07-23 - scored window 2026-07-23 -> 2026-09-11

## VERDICT: FAIL (expectancy -0.033R < 0.05R; PF 1.04 < 1.15)
- forward: 519 trades, win 35.3%, -0.033R ($1.83/trade), PF 1.04, 1/1 quarters+, maxDD $5,704.72
- bootstrap 90% CI [-0.134,+0.066]R P(>0)=29% -> CI includes 0
- thresholds: interim 100, verdict 300
