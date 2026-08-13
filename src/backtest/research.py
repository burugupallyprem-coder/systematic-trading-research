"""Phase 1.5 research harness: disciplined parameter sweep.

Anti-overfitting rules:
- Grids are PRE-DECLARED in config.yaml (small, motivated - not thousands).
- Parameters are chosen on TRAIN data only; the winner is evaluated ONCE on
  untouched VALIDATION data. The gate applies to VALIDATION results.
- WEAK PASS label when the winner's TRAIN edge was ~zero (selection carried
  no information -> validation result is unconfirmed, could be regime/luck).
- Overfit signature flagged: good train + negative validation.
- Slippage sensitivity on the winner (does the edge survive higher costs?).

Run: python -m src.backtest.research
"""

import itertools
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml

from src import data as data_mod
from src import slackbot
from src.backtest import engine, metrics
from src.strategies import filters, momentum, orb, vwap_revert

ROOT = Path(__file__).resolve().parent.parent.parent
STRATEGIES = {"orb": orb, "vwap_revert": vwap_revert, "momentum": momentum}


def load_config():
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def expand_grid(grid):
    keys = sorted(grid.keys())
    return [dict(zip(keys, values))
            for values in itertools.product(*(grid[k] for k in keys))]


def day_groups(bars):
    groups = []
    for symbol, sym_bars in bars.groupby("symbol"):
        for _, day in sym_bars.groupby("date"):
            day = day.reset_index(drop=True)
            if len(day) >= 20:
                groups.append((symbol, day))
    return groups


def build_context(groups, cfg):
    """Per-date market context for opt-in filters: SPY regime + relative-strength
    scores. Computed once per group set (train / val) and reused across combos."""
    rs_cfg = cfg.get("research", {})
    ob = int(rs_cfg.get("regime_open_bars", 3))
    cutoff = rs_cfg.get("regime_cutoff_et", "10:30")
    spy_days, early = {}, {}
    for symbol, day in groups:
        date = day["date"].iloc[0]
        early.setdefault(date, {})[symbol] = filters.early_return(day, ob)
        if symbol == "SPY":
            spy_days[date] = day
    spy_long, spy_er, spy_break = {}, {}, {}
    for date, sday in spy_days.items():
        spy_long[date] = filters.spy_long_ok(sday, ob, cutoff)
        spy_break[date] = filters.spy_break_minute(sday, ob, cutoff)   # causal timing gate
        spy_er[date] = filters.early_return(sday, ob)
    rs = {date: {s: er - spy_er.get(date, 0.0) for s, er in syms.items()}
          for date, syms in early.items()}
    prev_close = {}
    by_sym = {}
    for symbol, day in groups:
        by_sym.setdefault(symbol, []).append(day)
    for symbol, days in by_sym.items():
        pc = None
        for d in sorted(days, key=lambda x: x["date"].iloc[0]):
            dt = d["date"].iloc[0]
            prev_close.setdefault(symbol, {})[dt] = pc
            pc = float(d["close"].iloc[-1])
    return {"spy_long_ok": spy_long, "spy_break_min": spy_break, "rs": rs, "prev_close": prev_close}


def walk_forward_folds(dates, n_folds):
    """Split a sorted unique date list into n sequential (roughly equal) folds.
    Returns list of (start_date, end_date) inclusive."""
    uniq = sorted(set(dates))
    if not uniq or n_folds < 1:
        return []
    size = max(1, len(uniq) // n_folds)
    folds = []
    for i in range(n_folds):
        lo = i * size
        hi = (i + 1) * size if i < n_folds - 1 else len(uniq)
        if lo >= len(uniq):
            break
        folds.append((uniq[lo], uniq[hi - 1]))
    return folds


def evaluate_walk_forward(groups, strat_mod, params, cfg, name, n_folds, context=None):
    """Run the chosen params across sequential folds of the validation window.
    Returns (positive_fold_count, total_folds, per_fold_expectancy_r)."""
    dates = [day["date"].iloc[0] for _, day in groups]
    folds = walk_forward_folds(dates, n_folds)
    per_fold = []
    for lo, hi in folds:
        fold_groups = [(sym, day) for sym, day in groups
                       if lo <= day["date"].iloc[0] <= hi]
        m = metrics.summarize(run_config(fold_groups, strat_mod, params, cfg, name, context))
        per_fold.append(m.get("expectancy_r", 0.0) if m.get("trades", 0) else 0.0)
    positive = sum(1 for r in per_fold if r > 0)
    return positive, len(per_fold), per_fold


def run_config(groups, strat_mod, params, cfg, name, context=None):
    trades = []
    rs_topk = params.get("rs_topk")
    for symbol, day in groups:
        ctx = None
        if context is not None:
            date = day["date"].iloc[0]
            if rs_topk:
                allowed = filters.top_k_symbols(context["rs"].get(date, {}), rs_topk)
                if symbol not in allowed:
                    continue
            ctx = {"spy_long_ok": context["spy_long_ok"].get(date, False),
                   "prev_close": context.get("prev_close", {}).get(symbol, {}).get(date)}
        signals = strat_mod.generate(day, params, ctx)
        if signals:
            trades.extend(engine.simulate_day(day, signals, cfg, name))
    return trades


def _combo_str(combo):
    return ", ".join(f"{k}={v}" for k, v in sorted(combo.items()))


def run():
    cfg = load_config()
    rs = cfg["research"]
    val_end = rs.get("val_end") or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    symbols = cfg["universe"]

    print(f"downloading {len(symbols)} symbols, {rs['train_start']} -> {val_end}", flush=True)
    bars = data_mod.fetch_bars(symbols, rs["train_start"], val_end,
                               timeframe=cfg["backtest"]["timeframe"],
                               feed=cfg["backtest"]["feed"])
    bars = data_mod.rth_only(bars)
    print(f"bars: {len(bars):,} rows", flush=True)
    train_end = pd.to_datetime(rs["train_end"]).date()
    val_start = pd.to_datetime(rs["val_start"]).date()
    train_groups = day_groups(bars[bars["date"] <= train_end])
    val_groups = day_groups(bars[bars["date"] >= val_start])
    print(f"train symbol-days: {len(train_groups):,}  val symbol-days: {len(val_groups):,}", flush=True)

    train_ctx = build_context(train_groups, cfg)
    val_ctx = build_context(val_groups, cfg)
    wf = rs.get("walkforward", {}) or {}
    wf_folds = int(wf.get("folds", 4))
    wf_min_frac = float(wf.get("min_positive_frac", 0.6))

    gate = cfg["gate"]
    train_floor = rs.get("min_train_expectancy_r", 0.02)
    report = [f"# Research report - {ts}", "",
              f"Universe {len(symbols)} - train {rs['train_start']} -> {rs['train_end']} - "
              f"validation {rs['val_start']} -> {val_end} - long+short grids - "
              f"slippage {cfg['costs']['slippage_cents']}c/side - grids pre-declared", ""]
    slack_blocks = []
    all_rows = []
    any_pass = False

    for strat_name, strat_mod in STRATEGIES.items():
        base = dict(cfg["strategies"].get(strat_name, {}))
        combos = expand_grid(rs["grids"][strat_name])
        results = []
        for idx, combo in enumerate(combos, 1):
            params = {**base, **combo}
            trades = run_config(train_groups, strat_mod, params, cfg, strat_name, train_ctx)
            m = metrics.summarize(trades)
            print(f"  [{strat_name} {idx}/{len(combos)}] {_combo_str(combo)} -> "
                  f"{m.get('trades', 0)} trades, {m.get('expectancy_r', 0)}R", flush=True)
            row = {"strategy": strat_name, **combo,
                   "train_trades": m.get("trades", 0),
                   "train_exp_r": m.get("expectancy_r", 0.0),
                   "train_pf": m.get("profit_factor", 0.0)}
            results.append((params, m, combo))
            all_rows.append(row)

        eligible = [r for r in results if r[1].get("trades", 0) >= rs["min_train_trades"]]
        if not eligible:
            report += [f"## {strat_name}: no config reached {rs['min_train_trades']} train trades", ""]
            slack_blocks.append(f"*{strat_name.upper()}* -> SKIP (no config had enough train trades)")
            continue
        eligible.sort(key=lambda r: r[1]["expectancy_r"], reverse=True)

        report += [f"## {strat_name}", "", "Top 3 on TRAIN:", ""]
        for params, m, combo in eligible[:3]:
            report.append(f"- {_combo_str(combo)}: {m['trades']} trades, "
                          f"{m['expectancy_r']}R, PF {m['profit_factor']}")
        best_params, best_train, best_combo = eligible[0]

        val_trades = run_config(val_groups, strat_mod, best_params, cfg, strat_name, val_ctx)
        vm = metrics.summarize(val_trades)
        if vm.get("trades", 0) == 0:
            report += ["", "Validation: 0 trades -> FAIL", ""]
            slack_blocks.append(f"*{strat_name.upper()}* -> FAIL (validation produced 0 trades)")
            continue
        verdict, why = metrics.gate_verdict(vm, gate)
        wf_pos, wf_total, wf_per = evaluate_walk_forward(
            val_groups, strat_mod, best_params, cfg, strat_name, wf_folds, val_ctx)
        wf_frac = wf_pos / wf_total if wf_total else 0.0
        wf_ok = wf_total > 0 and wf_frac >= wf_min_frac
        # a real PASS must clear BOTH the single-window gate AND walk-forward folds
        if verdict == "PASS" and not wf_ok:
            verdict = "FAIL"
            extra = f"walk-forward only {wf_pos}/{wf_total} folds positive (< {wf_min_frac:.0%})"
            why = extra if why == "all gate checks met" else f"{why}; {extra}"
        weak = verdict == "PASS" and best_train["expectancy_r"] < train_floor
        overfit = (best_train["expectancy_r"] >= gate["min_expectancy_r"]
                   and vm["expectancy_r"] < 0)
        label = "WEAK PASS" if weak else verdict
        if verdict == "PASS":
            any_pass = True

        sens = []
        for sc in rs.get("slippage_sensitivity_cents", []):
            cfg_s = {**cfg, "costs": {**cfg["costs"], "slippage_cents": sc}}
            sm = metrics.summarize(run_config(val_groups, strat_mod, best_params, cfg_s, strat_name, val_ctx))
            sens.append(f"{sc}c -> {sm.get('expectancy_r', 0)}R")

        report += ["", f"**Winner on validation: {label}**"
                   + (" ** OVERFIT SIGNATURE" if overfit else ""),
                   f"- winner params: {_combo_str(best_combo)}",
                   f"- train: {best_train['trades']} trades, {best_train['expectancy_r']}R, PF {best_train['profit_factor']}",
                   f"- validation: {vm['trades']} trades, win {vm['win_rate']}%, {vm['expectancy_r']}R "
                   f"(${vm['expectancy_usd']}/trade), PF {vm['profit_factor']}, "
                   f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+, maxDD ${vm['max_drawdown']:,}",
                   f"- slippage sensitivity: {' | '.join(sens)}",
                   f"- walk-forward: {wf_pos}/{wf_total} folds positive "
                   f"(per-fold R: {', '.join(f'{r:+.3f}' for r in wf_per)})",
                   f"- gate: {why}", ""]

        lines = [f"*{strat_name.upper()}* -> *{label}*" + (" (overfit signature!)" if overfit else "")]
        lines.append(f"    winner: {_combo_str(best_combo)}")
        lines.append(f"    validation: {vm['trades']} trades | win {vm['win_rate']}% | "
                     f"{vm['expectancy_r']:+}R (${vm['expectancy_usd']}/trade) | PF {vm['profit_factor']} | "
                     f"{vm['quarters_positive']}/{vm['quarters_total']} quarters+ | maxDD ${vm['max_drawdown']:,}")
        lines.append(f"    train: {best_train['trades']} trades | {best_train['expectancy_r']:+}R | "
                     f"PF {best_train['profit_factor']}"
                     + ("  <- near-zero selection edge: validation result is UNCONFIRMED" if weak else ""))
        if sens:
            lines.append(f"    costs: {' | '.join(sens)}")
        lines.append(f"    walk-forward: {wf_pos}/{wf_total} folds positive "
                     f"(need >= {wf_min_frac:.0%})")
        slack_blocks.append("\n".join(lines))

    verdict_line = ("Verdict: a config cleared the gate on validation - candidate for Phase 2 paper deployment (owner's call)."
                    if any_pass else
                    "Verdict: nothing clears the gate on validation. We keep iterating or accept the honest NO EDGE.")
    report += ["", verdict_line, ""]

    # Kill-switch input: research decides whether live trading is ARMED. The
    # trader only obeys this when config arming.mode == "auto" (see trader.py).
    arming = {
        "armed": bool(any_pass),
        "verdict": "PASS" if any_pass else "NO-EDGE",
        "reason": ("a config cleared the single-window gate AND walk-forward"
                   if any_pass else
                   "no config cleared gate + walk-forward - do not arm"),
        "updated_utc": ts,
        "note": "Consumed by trader entry session only when config arming.mode == 'auto'.",
    }
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "arming.json").write_text(json.dumps(arming, indent=2), encoding="utf-8")

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    (out_dir / f"research_{stamp}.md").write_text("\n".join(report), encoding="utf-8")
    pd.DataFrame(all_rows).to_csv(out_dir / f"research_grid_{stamp}.csv", index=False)
    print(f"report written: reports/research_{stamp}.md", flush=True)

    header = (f"*[RESEARCH]* {ts}\n"
              f"Sweep: {len(symbols)} symbols | train {rs['train_start']} -> {rs['train_end']} | "
              f"validation {rs['val_start']} -> {val_end} | long+short grids | "
              f"{sum(len(expand_grid(rs['grids'][s])) for s in STRATEGIES)} configs")
    footer = (f"{verdict_line}\n"
              "_How to read: R = avg profit per $1 risked (gate >= +0.05R) | "
              "PF = gross wins / gross losses (gate >= 1.15) | "
              "quarters+ = calendar quarters profitable (gate >= 60%). "
              "Train picks the winner; only validation counts._\n"
              f"Full detail: reports/research_{stamp}.md in the repo")
    slackbot.post("\n\n".join([header] + slack_blocks + [footer]))


if __name__ == "__main__":
    run()
