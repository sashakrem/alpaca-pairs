# ETF Momentum Rotation Bot (Alpaca paper trading)

A monthly ETF rotation strategy, backtested against buy-and-hold, that trades an
[Alpaca](https://alpaca.markets) **paper** account automatically.

## The strategy

Each month end, score 10 ETFs that cover the major asset classes:

| | | | | |
|---|---|---|---|---|
| SPY US large caps | QQQ Nasdaq 100 | IWM US small caps | EFA developed ex-US | EEM emerging markets |
| TLT long Treasuries | IEF mid Treasuries | GLD gold | DBC commodities | VNQ real estate |

**Score** = average of the 1-, 3-, 6- and 12-month returns.
**Hold** the top 3 at 1/3 each. A pick that didn't beat T-bills (BIL) gives its
slot to BIL instead. In a broad sell-off everything fails that test and the
portfolio sits in T-bills. That rule is what cut the June–December 2008 loss from −34% to −3%.

The signal uses the last completed month's close. Trades happen the next trading morning.

## Backtest: $100,000, June 2008 – Nov 2026

`python backtest.py`. It executes at the next day's open, tracks real share counts and charges 5 bps per dollar traded.

| Strategy | Final value | Per year | Sharpe | Worst drop | Worst year | Losing years |
|---|---|---|---|---|---|---|
| **momentum** | $484,815 | 8.9% | 0.67 | **−22%** | **−10%** | 6 of 19 |
| buy & hold SPY | **$723,254** | **11.3%** | 0.64 | −51% | −34% | 3 of 19 |
| 60% SPY / 40% IEF | $431,072 | 8.3% | 0.77 | −31% | −17% | 3 of 19 |

**What this does and doesn't show**

- All three made money over 18 years. **None beat simply holding SPY** on total return.
  The momentum strategy trades return for much smaller losses in crashes
  (2008: −3% vs −34%; 2022: −3% vs −18%). It lagged badly in calm bull markets (2016, 2019).
- **Costs matter.** It trades about 6× its value per year. At 20 bps instead of 5 the
  yearly return falls from 8.9% to 7.8% (`--cost-bps 20`).
- **Some selection bias.** Before building this I compared 5 momentum variants
  (lookback, number held, weighting) and kept the blended-lookback/top-3 version, which
  was the best of them. Expect real results somewhat below the backtest.
- **Short periods are luck.** Over any few months the result is mostly noise. The strategy
  lost money in 6 of 19 calendar years. A paper account showing a profit after one month
  proves nothing either way.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # paste your Alpaca PAPER keys
python -m pytest              # offline tests, no keys needed
python backtest.py            # needs internet (Yahoo Finance), no keys
python trade.py               # dry run: shows account, target and planned orders
python trade.py --execute     # places the orders in the paper account
```

## Running it automatically

Run `python trade.py --execute` once each weekday at about 10:00 New York time. It rebalances
on the first run after each month end. On every other day it prints the status and exits, so
running it daily is safe. Example cron line (machine clock in New York time):

```
0 10 * * 1-5  cd /path/to/repo && .venv/bin/python trade.py --execute >> logs/trade.log 2>&1
```

## Safety

- **Paper only.** The Alpaca client is created with `paper=True` and there is no option to change it.
- **Dry run by default.** Orders go out only with `--execute`, and only while the market is open.
- **Once per month.** Orders are tagged `bot-<YYYY-MM>-…`. If any exist for the current signal,
  the bot won't trade again (`--force` overrides). No local state file is needed.
- **No margin.** Sells fill first. Buys are capped at the cash actually available, with 1% kept back.
- **Leaves your other positions alone.** Symbols outside the strategy are ignored.

## Code

| File | What it does |
|---|---|
| `bot/strategy.py` | Momentum score, top-3 selection, T-bill filter; buy-and-hold and 60/40 baselines |
| `bot/backtest.py` | Daily simulation: next-open execution, share holdings, costs; performance stats |
| `bot/broker.py` | Order planning (pure function) and execution on Alpaca |
| `bot/data.py` | Yahoo Finance history for backtests; Alpaca market data for live trading |
| `backtest.py`, `trade.py` | Command-line entry points |
| `tests/test_bot.py` | No look-ahead, cash fallback, backtest arithmetic, order planning, order building against a fake Alpaca client |
