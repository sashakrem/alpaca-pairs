# Statistical Arbitrage with Ornstein-Uhlenbeck Spreads

A pairs-trading research system: find cointegrated stock pairs, model their spread as an
Ornstein-Uhlenbeck process, and (in later stages) backtest and paper-trade it on Alpaca.

**Status:** Stage 1 (pair screening + OU modeling) ✅ · Stage 2 (Kalman hedge ratio) ✅ · Stage 3 (backtest) · Stage 4 (live paper trading)

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

## Stage 2: time-varying hedge ratio (Kalman filter)

A fixed β assumes the relationship between two companies never changes. The Kalman filter
lets β drift slowly and re-estimates it every day. Getting this right took three tries, and
the two failures are the interesting part.

**Trap 1: the textbook model finds pairs in pure noise.** The usual setup lets an intercept
and β follow random walks and trades the filter's forecast error. Fitted by maximum likelihood,
the intercept absorbs the spread, so the forecast error looks stationary for *any* two stocks.
On two independent random walks it gives ADF p ≈ 0 (`test_textbook_random_walk_intercept_model_finds_fake_pairs`).

**Trap 2: writing the model in price levels.** `log Y_t = β_t log X_t + m_t` looks fine,
but log prices sit around 4, so a small move in β_t shifts Y by about 4·Δβ and acts as a free
intercept. In synthetic tests this made one leg of a real pair look "cointegrated" with
every unrelated stock (14 false candidates in one 153-pair market). The β that fits levels also
isn't the β that hedges returns, so it lagged real drift.

**The model used.** Written in daily log returns, with the OU spread as an explicit state:

$$\Delta y_t = \beta_t\,\Delta x_t + (m_t - m_{t-1}) + v_t,\qquad \beta_t = \beta_{t-1} + w_t,\qquad m_t = \varphi\, m_{t-1} + \eta_t$$

State `s_t = [β_t, m_t, m_{t-1}]`, transition `F = [[1,0,0],[0,φ,0],[0,1,0]]`, observation row
`H_t = [Δx_t, 1, −1]`. φ = e^{−θ} links it to the OU SDE, so the half-life is −ln 2 / ln φ.
The four parameters (q, φ, σ²_η, r) are fitted by maximizing the prediction-error log-likelihood
from 9 starting points, because the likelihood has competing local optima. The filter loop is
compiled with numba (about 30× faster), and a test checks it against a plain matrix implementation.

**What gets tested is the portfolio, not the filter's state.** Selection uses the value of the
self-financing portfolio you'd actually hold (long Y, short β_{t−1}·X, rebalanced daily):

$$V_t = \sum_{s \le t} \left(\Delta y_s - \beta_{s-1}\,\Delta x_s\right)$$

V can't profit from β moving, only from real co-movement. A pair is a candidate if V has an OU
half-life of 1–60 days in formation, V passes ADF on the holdout (frozen parameters,
Benjamini-Hochberg across all pairs), and V's holdout half-life is also in range.

The holdout ADF includes a linear trend (`regression="ct"`). I added this **after** a control
pair failed: GLD and IAU (two gold ETFs) differ in fees by 0.15%/yr, so their hedged portfolio
drifts slowly downhill. Without the trend term the test rejects them (p = 0.87); with it they
pass (p = 0.002). The change was made on a control pair, before rerunning the sector screens,
and the calibration below was rerun with it.

### Calibration: 12 synthetic markets, 2,700 unrelated pairs, 54 planted pairs

| Hedge-ratio drift (in returns) | Static (stage 1) found | Kalman found | False positives: static / Kalman |
|---|---|---|---|
| none | **18/18** | 11/18 | 0 / 2 |
| 30% over the sample | **18/18** | 11/18 | 0 / 0 |
| 100% over the sample | 7/18 | **11/18** | 0 / 0 |

The Kalman screen's 2 false positives out of 35 discoveries (≈6%) match the 5% FDR target.
**Conclusion:** the Kalman filter is worth it only when hedge ratios drift a lot. Otherwise a
fixed β is more powerful, because it has fewer parameters to estimate.

### Results on real data (formation 2019–2023, holdout 2024–Nov 2026)

| Universe | Static candidates | Kalman candidates |
|---|---|---|
| staples, banks, energy, semis, utilities (225 pairs) | 0 | 0 |
| control: GLD/IAU, SPY/IVV, GOOGL/GOOG, EWA/EWC | 0 | 1 (GLD/IAU) |

No large-cap pair in these sectors held up out of sample with either method. The pairs that
passed formation (e.g. CL/KHC, BAC/PNC) broke in 2024–26. Some are far from their mean
(CL/KHC at z ≈ 7), and a naive strategy would keep adding to that losing position. The likely
reasons: these are heavily traded stocks where simple pairs have been arbitraged away, and
2020–2023 included regime changes (COVID, rate hikes).

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
python run_screen.py --universe staples --method kalman   # stage 2
python run_screen.py --demo --method kalman --beta-drift 1.0
```

Universes: `banks`, `energy`, `staples`, `semis`, `utilities`, or `--symbols A B C ...`.
Results are saved to `results/screen_<name>.csv` (or `screen_<name>_kalman.csv`).

## Known limitations
- **Survivorship bias:** universes are today's large caps.
- **Engle-Granger** assumes a single cointegrating vector. Johansen is the multivariate generalization (planned).
- **Kalman sensitivity:** with a stable β it finds fewer real pairs than the static screen (11/18 vs 18/18).
- **ADF over-rejects** when a random walk carries large stationary noise (Schwert, 1989). If one
  stock has strong mean reversion of its own, it can occasionally pair with an unrelated stock.
  The out-of-sample half-life check catches most of these.
- **Only one hedge ratio per pair.** Baskets (Johansen) are not yet supported.
