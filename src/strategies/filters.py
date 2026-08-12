"""Opt-in signal filters that attack WHY breakouts fail (not just parameters).

All pure functions, no lookahead beyond what the caller already sees. Used by
the research sweep (default OFF). They are NOT wired into live trading until a
walk-forward run shows they actually help - discipline over hope.

1. vol floor      - skip dead-tape days: opening-range width too small.
2. spy regime     - only take longs when the market (SPY) is itself breaking up.
3. relative strength - only trade the strongest names vs SPY that morning.
"""


def or_width_frac(day, open_bars):
    """Opening-range width as a fraction of price. day: DataFrame ascending."""
    rng = day.iloc[:open_bars]
    hi = float(rng["high"].max())
    lo = float(rng["low"].min())
    if lo <= 0:
        return 0.0
    return (hi - lo) / lo


def passes_vol_floor(day, open_bars, min_width_frac):
    """True if the opening range is wide enough to bother trading."""
    if not min_width_frac:
        return True
    return or_width_frac(day, open_bars) >= float(min_width_frac)


def _bar_time_ge(row, hh, mm):
    t = row["et"].time()
    return (t.hour, t.minute) >= (hh, mm)


def spy_long_ok(spy_day, open_bars, cutoff_et):
    """Market regime gate: True if SPY closes above its own opening-range high on
    some bar before cutoff (i.e. the broad market is breaking up, not fading).
    Evaluated only on bars up to cutoff - no end-of-day lookahead."""
    if spy_day is None or len(spy_day) < open_bars + 1:
        return False
    rng = spy_day.iloc[:open_bars]
    hi = float(rng["high"].max())
    ch, cm = [int(x) for x in cutoff_et.split(":")]
    for i in range(open_bars, len(spy_day)):
        row = spy_day.iloc[i]
        if _bar_time_ge(row, ch, cm):
            break
        if float(row["close"]) > hi:
            return True
    return False


def early_return(day, open_bars):
    """Return over the opening range: (last-of-range close / first open) - 1.
    A lookahead-safe momentum proxy known by the end of the opening range."""
    rng = day.iloc[:open_bars]
    first_open = float(rng.iloc[0]["open"])
    last_close = float(rng.iloc[-1]["close"])
    if first_open <= 0:
        return 0.0
    return last_close / first_open - 1.0


def top_k_symbols(rs_by_symbol, k):
    """Names with the highest relative-strength score. rs_by_symbol: {sym: score}."""
    if not rs_by_symbol or k is None:
        return set(rs_by_symbol or {})
    ranked = sorted(rs_by_symbol.items(), key=lambda kv: kv[1], reverse=True)
    return {sym for sym, _ in ranked[:int(k)]}


def spy_short_ok(spy_day, open_bars, cutoff_et):
    """Mirror of spy_long_ok: True if SPY closes BELOW its opening-range low before
    cutoff (the broad market is breaking DOWN) - the regime gate for SHORTS."""
    if spy_day is None or len(spy_day) < open_bars + 1:
        return False
    rng = spy_day.iloc[:open_bars]
    lo = float(rng["low"].min())
    ch, cm = [int(x) for x in cutoff_et.split(":")]
    for i in range(open_bars, len(spy_day)):
        row = spy_day.iloc[i]
        if _bar_time_ge(row, ch, cm):
            break
        if float(row["close"]) < lo:
            return True
    return False


def bottom_k_symbols(rs_by_symbol, k):
    """Names with the LOWEST relative-strength (weakest vs SPY) - the short candidates."""
    if not rs_by_symbol or k is None:
        return set(rs_by_symbol or {})
    ranked = sorted(rs_by_symbol.items(), key=lambda kv: kv[1])   # weakest first
    return {sym for sym, _ in ranked[:int(k)]}
