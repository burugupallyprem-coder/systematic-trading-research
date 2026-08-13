# Systematic Trading Research Platform (US Equities)

A from-scratch quantitative research platform for designing, backtesting, and paper-deploying
intraday systematic strategies — engineered around one principle: **methodological rigor over
P&L.** It is built to *disprove its own ideas* before trusting them, and it does: across six
strategy families, it rejected five with real out-of-sample metrics and deployed only a
measured benchmark to paper.

> **Research / paper-trading only.** No live capital. The broker client is hard-locked to paper endpoints.

**Tools & libraries:** Python (pandas, NumPy) · Alpaca API (equities data + paper execution) ·
GitHub Actions (CI + serverless scheduling) · Slack API (monitoring) · YAML config · unit tests.

---

![Out-of-sample scorecard: most strategies correctly rejected against a pre-registered +0.05R gate](docs/results_scorecard.png)

*Every strategy is judged once on untouched out-of-sample data against a pre-registered gate. The platform's value is the discipline to reject its own ideas — five of six here.*

## Results at a glance (real, out-of-sample, after modeled costs)

Every strategy below ran through the **same pre-registered gate** (net expectancy in R, profit
factor, walk-forward folds, cost sensitivity), judged **once** on an untouched validation window:

| Strategy | Out-of-sample result | Verdict |
|---|---|---|
| Filtered Opening-Range Breakout (regime + relative-strength) | ~150 val trades, **+0.082R**, PF **1.289**, walk-forward **3/4** folds | Deployed to paper as a measured benchmark (flagged unconfirmed) |
| VWAP mean-reversion | 1,948 val trades, **−0.094R**, PF 0.853, 0/3 quarters+ | Rejected |
| Momentum continuation | 604 val trades, **+0.006R**, PF 1.136 (below gate) | Rejected |
| 3-step break-and-retest scalp (1-min) | 2,070 val trades, **+0.098R**, PF 1.172; bootstrap 90% CI **[+0.042, +0.153]** | Weak pass — flagged fragile |
| Momentum on **real index futures** (MES/MNQ/MYM, Databento) | 314 trades, **−0.069R**, PF 0.938, 1/4 folds | Rejected — edge did not survive real costs |
| Diversified time-series momentum (21 CME markets) | OOS Sharpe **−0.27**, −4.4%/yr, maxDD −49.6% | Rejected on this window |

**The measurable impact:** the pipeline caught a strategy that looked strong on an equity proxy
(**+0.263R** in-sample) and proved it was a **money-loser on real futures data with real costs**
— i.e., it prevented deploying a losing strategy *before* a data-feed subscription or live risk.
That "don't-fool-yourself" catch is the entire point.

> **⚠️ 2026-08-13 correction:** the *Filtered ORB* row above, and the 2026-08-12 prop-firm log at the
> bottom of this file, were affected by a **look-ahead bug in the market-regime filter** (since fixed).
> The causal out-of-sample edge is materially weaker — **OOS Sharpe fell 1.57 → 0.46, FLEX 80% → 46%**,
> and 2022 turns negative. Full write-up, before/after numbers, and charts are in the
> **Correction & Research Log (2026-08-13)** section below. Finding this before funding an account is
> the platform doing its job.

## What I built, how, and what happened
- **Engineered a no-lookahead, event-driven backtest engine** (pandas/NumPy): signals never see
  the bar they trade on; stops checked before targets; gaps fill on the unfavorable side;
  fixed-fractional risk sizing. Validated with offline unit tests.
- **Built a pre-registered research harness** with train/validation separation, multi-fold
  walk-forward, bootstrap significance testing, per-regime breakdowns, and cost-sensitivity
  sweeps — the guardrails that separate real edges from over-fit noise.
- **Integrated real market data** from Alpaca (equities) and Databento (CME futures, roll-adjusted
  continuous series), with a pre-flight cost estimator that caps spend before any download.
- **Automated the full pipeline serverlessly** on GitHub Actions: pre-market briefing, entry
  session, end-of-day reconciliation, and weekly research sweeps — with Slack monitoring, a
  self-rendering dashboard, server-side bracket orders, hard risk caps, and a paper-lock plus an
  arming kill-switch wired to research verdicts.

## Architecture
```
GitHub Actions (scheduler) -> Signal layer -> No-lookahead backtest engine -> Risk engine
        |                                                                          |
   Slack alerts + dashboard  <-  Alpaca paper (code-locked) / Databento data  <----+
```

## Repository layout
- `src/strategies/` — signal generators (ORB, VWAP-reversion, momentum, filters)
- `src/backtest/` — engine, metrics, research harness, walk-forward, futures + TSMOM
- `src/live/` — paper execution, risk engine, pre-market, dashboard
- `tests/` — offline unit tests (engine, filters, gate, cost model)
- `reports/` — timestamped research reports (the full audit trail)

---
*A self-directed study in quantitative research methodology and trading-systems engineering:
reproducibility, cost realism, and intellectual honesty about what does and does not work.*

---

## Correction & Research Log — 2026-08-13: a look-ahead bug found, fixed, and its real impact

**Headline:** On 2026-08-12 I documented the ORB as a viable prop-firm edge (~80% FLEX, out-of-sample
Sharpe 1.57). On 2026-08-13, prompted by a look-ahead audit, I found a **timing look-ahead in the
market-regime filter** that was inflating those numbers. After fixing it, the out-of-sample edge
roughly halves-to-thirds and turns **negative in the 2022 bear**. This is the platform working as
designed — catching the illusion *before* a dollar of challenge fee was spent.

### The bug — what it was
The ORB only goes long when the market (SPY) is "breaking up." That regime flag was computed in
`build_context()` as a **single per-day boolean**: *did SPY close above its opening range anytime
before the 10:30 cutoff?* But a stock long can enter at, say, 10:05 — and the flag could be `True`
only because SPY broke up at 10:20. So a 10:05 entry was being approved with SPY data from 10:20 —
**future information**. The price/execution engine was already clean; the leak was purely in the
*filter's timing*.

```
10:05  stock breaks out → decision made
         ↑ but the per-day regime flag already "knew" SPY would confirm at 10:20 (the future)
```

### How it was found
An interview-style check from a mentor: *"when the strategy decides on bar 10, does its INPUT contain
any bar-11+ information?"* Auditing the whole signal path — `orb.generate` → `build_context` →
`filters.spy_long_ok` — showed the regime boolean scanned SPY all the way to 10:30, decoupled from
each trade's actual entry time. The execution engine passed the audit; the regime *input* did not.

### The fix — make the gate causal
`filters.spy_break_minute()` now records the **exact minute SPY first breaks up**, and `orb.generate`
allows a long **only on/after that minute**. A stock long can no longer fire before SPY has actually
confirmed. Proven with a unit case: stock breaks 09:45, SPY confirms 10:00 → the old gate entered at
09:50 (look-ahead), the fixed gate waits and enters 10:05 (causal). Note the **live bot was already
causal** (it checks SPY "so far" each poll), so only the *backtest* was optimistic — this fix makes
the backtest match what live actually does.

### Before vs after (causal), 2020-07 → 2026-08

![ORB avg daily return by year, before vs after the look-ahead fix](docs/lookahead_year_by_year.png)

![Out-of-sample Sharpe, FLEX pass, and breach rate before vs after the fix](docs/lookahead_oos_metrics.png)

| Metric | Before (look-ahead) | After (causal) |
|---|---|---|
| OOS pre-2024 avg daily | +0.067% | **+0.018%** |
| OOS pre-2024 Sharpe | 1.57 | **0.46** |
| OOS pre-2024 FLEX pass | 80% | **46%** |
| OOS pre-2024 max drawdown | −9.3% | **−19.6%** |
| 2022 bear (avg daily / maxDD) | +0.014% / −7.8% | **−0.056% / −15.2%** |
| Overall FLEX (full period) | 80.4% | **55.6%** |
| Time-to-fund @0.25% (median / breach) | 138d / 1.3% | **246d / 18.5%** |

### Honest conclusion (this supersedes the 2026-08-12 log below)
- **Do not fund a TTP account on the ORB as it stands.** The causal out-of-sample edge is marginal
  (Sharpe ~0.46), it **loses in bear markets** (2022: −15% drawdown), and survivorship bias inflates
  even the +0.018% figure. Roughly **two-thirds of the apparent edge was the look-ahead.**
- **The process worked.** The leak was caught before any fee was paid; the live bot was never
  affected. **Live paper is now the ground truth.**
- **Open question (being tested next):** does *any* causal ORB configuration — with/without the regime
  gate, long-only vs base — have a real out-of-sample edge, judged **once** on a small pre-declared
  grid (no overfitting)? If not, this family is honestly a dead end and we stop polishing it.

*The 2026-08-12 log below is kept unedited for the record. Its ORB numbers are now known to be
inflated by the look-ahead described above — read it as "before the correction."*

---

## Research Log — 2026-08-12 (SUPERSEDED — ORB numbers inflated by the look-ahead fixed on 2026-08-13; kept for the record)

A full day of pre-registration-style testing to answer one question: **can the live ORB fund a
prop account, and at what size / speed / risk?** Every result below is RESEARCH ONLY, run in CI on
real data, and deliberately haircut for known biases. All new modules live in `src/backtest/`.

### Headline conclusions
- **Two real funded paths exist — and neither is gold ORB or a swing/short version:**
  1. **Stock ORB → Trade The Pool (TTP) FLEX**, traded **manually** (TTP bans bots), long-only,
     **0.25% risk with a house-money ratchet**. ~6 months to fund, ~1% breach on the sample.
  2. **Gold *daily-trend* → Apex** (automatable) — the existing `mgc_prop` strategy, **not** the ORB.
- **The ORB edge is stock-specific.** It did not transfer to gold intraday (avg-R +0.019, Apex bust
  63–78%). Ported filtered ORBs on crypto/FX already existed; the intraday breakout is a US-equity
  phenomenon.

### What we tested and found (stock ORB, TTP $50k FLEX)
- **TTP facts (verified):** US-accepted, real equities, **static** 4% max loss (does not trail),
  6% target, FLEX is **untimed** (MAX has a 60-day limit and is unpassable at safe size). **Automated
  trading is prohibited** — deployment must be manual off the bot's signals.
- **Multi-year edge (2020-07→2026):** out-of-sample pre-2024 held (+0.067%/day, Sharpe 1.57, FLEX
  ~80% over 864 unseen days). **2022 bear is the weak spot: −7.8% drawdown > the 4% leash.**
- **Short side (`short_test`, `short_v2`):** symmetric raises avg-R and Sharpe but **deepens
  drawdown** (bull-year short squeezes); trend-gating + half-size did not fix it; TTP pass rate
  unchanged. **Dropped for TTP** (kept only as an idea for an unconstrained own-account).
- **Position size is the real lever (`risk_sweep`):** worst-year drawdown scales ~linearly with
  per-trade risk. 0.50% → −7.8% (breaches); **0.25% → −3.9% (just fits)**; 0.20% → −3.1% (safe).
- **Time-to-fund (`time_to_pass`):** pass rate is ~80% at *every* size (FLEX is untimed); size only
  changes **speed** and **breach**. 0.25% ≈ 6.6 months / 1.3% breach; 0.15% ≈ 13 months / 0% breach.
- **House-money ratchet (`house_money`):** start small, size up only after a cushion → funds ~1 month
  faster with the *same* breach (a free speedup). Starting bigger (0.30–0.50%) blows breach up to 5–10%.
- **Golden-moment / conviction sizing (`conviction_scan`):** **no** stable pre-entry signal (rank &
  features were noise out-of-sample). The edge is broad, not cherry-pickable → **flat sizing**.
- **Breakout vs retest entry (`orb_entry_compare`):** **breakout wins on everything.** The retest is
  adversely selected (it catches weak pullbacks and misses the runners), so keep entering at breakout.
- **Intraday vs swing (`swing_test`):** swing funds faster (~1 month) and the ORB *does* have some
  multi-day momentum, **but win rate collapses 54% → 27%, breach jumps to 30–38% plus overnight gap
  risk.** A high win rate beats high R under a 4% leash → **intraday stays**.

### Income reality ($50k FLEX, 70% split, 0.25% risk)
| Year type | Take-home / year | Take-home / month |
|---|---|---|
| Best (2021-like) | ~$5,000 | ~$415 |
| Typical | ~$3,000 | ~$250 |
| Worst (2022-like) | ~$600 | ~$50 |

The eval phase (~6 months) is **unpaid**; commissions (~$4–5/day) meaningfully erode thin years.
One $50k account at safe size is **side income, not a salary** — scaling toward TTP's $450k cap is
how the numbers grow.

### Cross-instrument (gold, Apex $50k — `mgc_prop/orb_apex` in the OANDA repo)
Gold M15 ORB, long+short, **COVID included** (2019→2026): avg-R **+0.019**, Apex **pass 16–32% /
bust 63–78%** at every size. **SKIP** — Apex's trailing drawdown is fatal to a near-zero edge. Gold's
real edge is the **daily-trend** system, not the ORB.

### New modules added today
`src/backtest/`: `short_test`, `short_v2`, `ttp_eval`, `ttp_history`, `risk_sweep`, `time_to_pass`,
`conviction_scan`, `house_money`, `orb_entry_compare`, `swing_test`, `data_probe`; plus
`data.fetch_bars_hist` and `strategies/filters.spy_short_ok` / `bottom_k_symbols`. Gold: the OANDA
repo's `mgc_prop/orb_apex.py` (reuses the verified Apex engine).

### Honest caveats on all of the above
Bull-heavy sample (free Alpaca floors at 2020-07, so **stock backtests exclude the COVID crash**);
**survivorship bias** (today's megacaps flatter the past); overlapping rolling starts → **wide
confidence intervals**; the backtest is a **ceiling** vs manual/live fills; and **live paper has not
yet confirmed** the edge post-carry-bug-fix. Verify exact TTP rules (esp. the consistency %) on the
account page before funding. Nothing here is deployed or financial advice.
