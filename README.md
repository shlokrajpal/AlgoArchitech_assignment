# Nifty 50 Quantitative Strategy & Analytics Pipeline

## Project Overview
This repository contains a complete, end-to-end quantitative trading pipeline for the Nifty 50 index.

## Data Ingestion & Business Cleaning
Because long-horizon, 1-minute resolution data for the Nifty 50 index is rarely available via a single free source, the ingestion module programmatically unifies historical Kaggle datasets with recent data dynamically fetched via the Yahoo Finance and Moneycontrol APIs. 
*Note: Since the Nifty 50 is an index, raw volume is zero. The strategy models price action strictly on OHLC data, assuming execution would occur via highly correlated derivatives (e.g., Nifty futures/options).*

### Protecting Against Garbage-In, Garbage-Out (GIGO)
Financial data is prone to bad ticks, which can falsely trigger Stop Losses (SL) or Take Profits (TP) in a backtest. The cleaning pipeline applies strict business logic:
1.  **Structural Integrity:** Drops biologically impossible bars (e.g., `High < Low` or `Open` outside the `High/Low` range).
2.  **Session Filtering:** Identifies and drops "Short Sessions" (like Muhurat trading) that severely distort intraday seasonality calculations.
3.  **Statistical Anomaly Detection:** Implemented a **Rolling Hampel Filter** (30-minute window, 3-sigma dynamic threshold) with a hard absolute floor (0.25% 1-min return). This successfully isolates true bad ticks without accidentally stripping out legitimate high-volatility market moves, providing a clean canvas for the HMM.

## Strategy Architecture & Hypothesis
**The Hypothesis:** Market microstructure behaves differently across volatility regimes. Low volatility environments are typically range-bound (choppy), whereas high volatility environments tend to be directional (trendy). 

**The Methodology:**
1.  **Regime Detection:** I trained a Gaussian HMM on 30-minute realized volatility. Testing states 2 through 6, the Bayesian Information Criterion (BIC) indicated that a 4-state model was optimal. These states proved to be sticky (lasting days) and were categorized as *Low, Normal, High,* and *Extreme* volatility.
2.  **Signal Generation:** * *Calm Regimes:* Mean-reversion logic (fading the distance from the 15-minute SMA).
    * *Volatile Regimes:* Momentum logic (following the distance from the 15-minute SMA).

*Optimization Attempt:* After initial testing, I theorized that trading purely in the "extreme" regimes (filtering out the middle noise) would provide a cleaner edge.

## Risk-First Backtesting
The backtester was built from scratch to avoid the hidden assumptions of out-of-the-box libraries, prioritizing realistic constraints:

* **Heavy Transaction Friction:** Modeled at **5 bps per side (0.05%)**, resulting in a 10 bps round-trip cost per trade. This accounts for slippage, broker commissions, and exchange transaction charges. 
* **Intrabar Path Resolution:** A major flaw in basic backtesters is assuming a favorable execution when both TP and SL are hit in the same 1-minute bar. This engine simulates the micro-structure path based on the bar's polarity:
    * *Bullish Bar (Close >= Open):* Assumes path is `Open -> Low -> High -> Close`. (SL hit first on Longs).
    * *Bearish Bar (Close < Open):* Assumes path is `Open -> High -> Low -> Close`. (TP hit first on Longs).
* **No Overnight Risk:** All positions are strictly flattened at the end of the session to eliminate overnight gap risks.

## Performance & Results
Currently, the strategy operates at a net loss. This is an expected reality when forcing a highly restrictive, pure-price strategy to absorb heavy, realistic institutional transaction costs. Most of the gross edge is consumed by friction.

However, the architecture behaves exactly as hypothesized:
* **Base Model (2 Meta-Regimes):** Yielded a total return of -14.44%, and a Max Drawdown of -14.44%. 
* **Extreme Regimes Filter (3 Regimes):** By filtering out the noise and only trading the extremes, the trade count was reduced from 16,393 to 11,025. Consequently, the total loss was mitigated to -9.75%, and the Max Drawdown was contained to -9.75%.
