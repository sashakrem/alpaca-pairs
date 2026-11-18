# Statistical Arbitrage with Ornstein-Uhlenbeck Spreads

A pairs-trading research system: find cointegrated stock pairs, model their spread as an
Ornstein-Uhlenbeck process, and (in later stages) backtest and paper-trade it on Alpaca.

**Status:** Stage 1 (pair screening + OU modeling) ✅ · Stage 2 (Kalman hedge ratio) · Stage 3 (backtest) · Stage 4 (live paper trading)

## The math

**Cointegration.** Two prices can each be random walks while a linear combination
`log Y − β log X − α` is stationary. That combination, the spread, is what we trade.
It's tested with Engle-Granger: OLS for β, then an ADF unit-root test on the residuals.

**Multiple testing.** 10 stocks give 45 pairs, and 20 stocks give 190. At α = 0.05 you expect
about 5% of unrelated pairs to look cointegrated by chance. The screener:
1. runs Engle-Granger in both directions and Bonferroni-corrects the better one (×2), and
2. controls the false discovery rate across all pairs with Benjamini-Hochberg.

**Spread dynamics.** The spread is modeled as an OU process:

$$dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t$$

Sampled at interval Δt, it is exactly an AR(1) with coefficient `b = e^{-θΔt}`, so OLS on
`X_{t+1} = a + bX_t + ε` gives θ, μ, σ. From those:
- **half-life** `ln 2 / θ`: how fast mispricings close (we want roughly 2–60 trading days)
- **stationary std** `σ / √(2θ)`: the natural unit for z-score entry and exit signals

**Out-of-sample validation.** β, α and the OU parameters are estimated on a formation period
and then *frozen*. The holdout spread is re-tested for stationarity. Pairs that break are rejected.

**Numerical note.** `simulate_ou` implements both the exact transition density and
Euler-Maruyama. Euler's AR coefficient is `1 − θΔt` instead of `e^{−θΔt}`, so it biases θ
and is unstable for `θΔt > 2`. `tests/test_core.py` demonstrates the bias.

## Validation on synthetic data

`python run_screen.py --demo` builds a market of 12 random walks plus 3 planted cointegrated
pairs with known β and half-life. Result: 12 pairs look significant without correction, 5 survive
FDR control, and both pairs flagged as candidates are planted ones with β and half-life
recovered. One planted pair fails the holdout ADF test (p = 0.07): with about 190 holdout days,
ADF has limited power. That's the false-negative cost of a strict filter.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste your Alpaca *paper* keys into .env
python -m pytest            # math checks, no keys needed
python run_screen.py --demo # synthetic run, no keys needed
python run_screen.py --universe staples
```

Universes: `banks`, `energy`, `staples`, `semis`, `utilities`, or `--symbols A B C ...`.
Results are saved to `results/screen_<name>.csv`.

## Known limitations
- **Survivorship bias:** universes are today's large caps.
- **Engle-Granger** assumes a single cointegrating vector. Johansen is the multivariate generalization (planned).
- **A static β** is estimated once. Stage 2 replaces it with a Kalman filter.
