"""Opening-range breakout - long or short (params["side"]).

First `open_bars` 5-min bars define the range. LONG: a close above the range
high before cutoff -> buy next open, stop = range low. SHORT: a close below
the range low -> sell short next open, stop = range high. Target = rr x risk.
Optional `vol_confirm`: breakout bar volume > 1.5x avg range-bar volume.
One trade per symbol-day.
"""

from src.strategies import filters

NAME = "orb"


def generate(day, params, ctx=None):
    side = params.get("side", "long")
    open_bars = int(params.get("open_bars", 3))
    cutoff = params.get("cutoff_et", "11:30")
    rr = float(params.get("rr", 2.0))
    max_risk_frac = float(params.get("max_risk_frac", 0.02))
    vol_confirm = bool(params.get("vol_confirm", False))
    min_or_width_frac = params.get("min_or_width_frac")   # vol floor (opt-in)
    regime_filter = bool(params.get("regime_filter", False))  # SPY must be breaking up
    if len(day) < open_bars + 2:
        return []
    rng = day.iloc[:open_bars]
    rng_high = float(rng["high"].max())
    rng_low = float(rng["low"].min())
    rng_vol = float(rng["volume"].mean())
    if rng_high <= rng_low:
        return []
    # opt-in volatility floor: skip dead-tape days (narrow opening range)
    if min_or_width_frac and not filters.passes_vol_floor(day, open_bars, min_or_width_frac):
        return []
    # Market-regime gate (long only). CAUSAL: a long may only fire on/after the bar SPY itself
    # broke up. If ctx carries "spy_break_min" (minute-of-day of SPY's first breakout) we enforce
    # the timing so the decision NEVER uses SPY data from after the entry bar. Legacy callers that
    # pass only "spy_long_ok" keep the old per-day gate.
    time_gate = regime_filter and side == "long" and ctx is not None and "spy_break_min" in ctx
    if regime_filter and side == "long" and ctx is not None and not time_gate:
        if not ctx.get("spy_long_ok", False):
            return []
    if time_gate and ctx.get("spy_break_min") is None:
        return []   # SPY never broke up before cutoff today -> no longs (same as spy_long_ok False)
    ch, cm = [int(x) for x in cutoff.split(":")]
    for i in range(open_bars, len(day) - 1):
        t = day.iloc[i]["et"].time()
        if (t.hour, t.minute) >= (ch, cm):
            break
        row = day.iloc[i]
        close = float(row["close"])
        broke = close > rng_high if side == "long" else close < rng_low
        if broke:
            if time_gate and (t.hour * 60 + t.minute) < ctx["spy_break_min"]:
                continue   # SPY had NOT confirmed by this bar -> causal block; keep scanning
            if vol_confirm and rng_vol > 0 and float(row["volume"]) < 1.5 * rng_vol:
                return []
            stop = rng_low if side == "long" else rng_high
            risk = abs(close - stop)
            if risk <= 0 or risk / close > max_risk_frac:
                return []
            return [{"entry_bar": i + 1, "stop": stop, "rr": rr, "side": side,
                     "time_stop_bars": params.get("time_stop_bars"),
                     "reason": f"orb_{side}_break"}]
    return []
