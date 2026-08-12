"""How far back does Alpaca's free feed actually return 5-min bars? We only got 2020-07 for
the full universe - this pins WHY (pre-COVID / COVID-crash availability). RESEARCH ONLY."""
from datetime import datetime, timezone
from src import data as data_mod, slackbot

def run():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    L = [f"[DATA-PROBE] {ts} - Alpaca 5-min history availability (free feed)"]
    for label, start, end in [("pre-COVID 2018", "2018-01-01", "2018-03-01"),
                              ("pre-COVID 2019", "2019-01-01", "2019-03-01"),
                              ("COVID crash", "2020-02-15", "2020-04-15"),
                              ("2020-07", "2020-07-01", "2020-07-31")]:
        try:
            b = data_mod.fetch_bars(["SPY"], start, end, timeframe="5Min", feed=data_mod_feed())
            if len(b):
                L.append(f"  {label:16} {start}..{end}: {len(b):>6} SPY bars  (earliest {b['ts'].min()})")
            else:
                L.append(f"  {label:16} {start}..{end}: 0 bars - NOT available on this feed")
        except Exception as e:
            L.append(f"  {label:16} {start}..{end}: ERROR {e}")
    L.append("If pre-2020 shows 0 bars, the free feed does not carry it -> the actual COVID-crash "
             "intraday test needs a paid Alpaca data plan or another source.")
    out = "\n".join(L); print(out)
    try: slackbot.post(out)
    except Exception: pass

def data_mod_feed():
    import yaml
    from src.backtest.research import load_config
    return load_config()["backtest"]["feed"]

if __name__ == "__main__":
    run()
