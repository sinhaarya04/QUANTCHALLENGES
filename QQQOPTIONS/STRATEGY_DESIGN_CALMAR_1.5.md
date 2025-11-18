# Trading Strategy Design: Calmar Ratio ≥ 1.5 Target

**Author:** Quantitative Research Team  
**Date:** 2025  
**Target:** Calmar Ratio ≥ 1.5 (CAGR / |Max Drawdown|)  
**Asset Class:** Equity Options (QQQ) with Underlying Exposure

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Research Workflow](#research-workflow)
3. [Trading Strategy Specification](#trading-strategy-specification)
4. [Modeling Approach](#modeling-approach)
5. [Code Architecture](#code-architecture)
6. [Risk Management Framework](#risk-management-framework)
7. [Backtest Specification](#backtest-specification)
8. [Tuning Guide](#tuning-guide)
9. [Python Implementation](#python-implementation)
10. [Monitoring & Deployment](#monitoring--deployment)

---

## 1. Executive Summary

### Strategy Overview

**Name:** Volatility-Regime Adaptive Options Flow Strategy (VRAOFS)

**Core Concept:** Combine options flow signals (put/call ratios, gamma/vega exposures) with volatility regime detection to generate directional forecasts. Apply aggressive drawdown control and volatility targeting to achieve Calmar ≥ 1.5.

**Expected Performance:**
- **CAGR:** 15-25% (target: 18%)
- **Max Drawdown:** < 12% (target: 10%)
- **Calmar Ratio:** ≥ 1.5 (target: 1.8)
- **Sharpe Ratio:** ≥ 1.2
- **Sortino Ratio:** ≥ 1.5

**Key Innovation:** Multi-layer risk management with regime-aware position sizing and dynamic drawdown throttling.

---

## 2. Research Workflow

### 2.1 Data Pipeline

```
Raw Data Sources:
├── Options EOD Data (QQQ)
│   ├── Trade date, expiry, strike
│   ├── Bid/ask, volume, open interest
│   ├── Greeks (delta, gamma, vega, theta)
│   └── Implied volatility
├── Underlying Price Data (QQQ)
│   ├── OHLCV (daily)
│   └── Intraday (60-min bars for momentum)
└── Market Regime Indicators
    ├── VIX (volatility index)
    ├── SPX returns (correlation proxy)
    └── Treasury yields (risk-free rate)
```

### 2.2 Feature Engineering Pipeline

**Step 1: Options Flow Features (Daily)**
- Put/Call Volume Ratio (PCVR) by tenor bucket
- Put/Call Open Interest Ratio (PCOIR) by tenor bucket
- Gamma Exposure (aggregate, by moneyness)
- Vega Exposure (aggregate, by tenor)
- Skew proxies (OTM put vega - ATM vega)
- Term structure slope (short-term vega - long-term vega)

**Step 2: Momentum Features (60-min + Daily)**
- 60-min momentum: `(P_t - P_{t-60min}) / P_{t-60min}`
- 120-min momentum: `(P_t - P_{t-120min}) / P_{t-120min}`
- Daily momentum: 1-day, 3-day, 5-day, 10-day, 20-day returns
- RSI(14), RSI(5) on 60-min bars
- MACD(12,26,9) on daily bars

**Step 3: Mean Reversion Features**
- 20-day Z-score: `(P_t - MA_20) / Std_20`
- 10-day Z-score: `(P_t - MA_10) / Std_10`
- Bollinger Band position: `(P_t - MA_20) / (2 * Std_20)`
- Price-to-MA ratios: P/MA_5, P/MA_10, P/MA_20, P/MA_50

**Step 4: Volatility Regime Features**
- Realized volatility (5-day, 10-day, 20-day, 60-day, annualized)
- VIX level and VIX/Realized Vol ratio
- Volatility regime classification:
  - Low: RV < 0.15
  - Normal: 0.15 ≤ RV < 0.25
  - High: RV ≥ 0.25
- GARCH(1,1) conditional volatility (optional, for regime detection)

**Step 5: Market Structure Features**
- Trend strength: `(Max_20 - Min_20) / Std_20`
- Volume profile: Current volume / 20-day average volume
- Correlation with SPX (rolling 20-day)
- Options flow intensity: Total notional volume / 20-day average

**Step 6: Target Variable**
- **Primary:** 1-day forward return (`ret_1d_fwd`)
- **Alternative:** Volatility-adjusted return (`ret_1d_fwd / RV_5`)

### 2.3 Modeling Pipeline

```
Data Preparation
    ↓
Time-Series Split (Train/Val/Test)
    ↓
Feature Selection (Remove multicollinearity, zero-variance)
    ↓
Model Training (Ridge, Lasso, Random Forest, Ensemble)
    ↓
Hyperparameter Tuning (Grid search on validation set)
    ↓
Forecast Generation
    ↓
Exposure Conversion (with risk filters)
    ↓
Backtest (walk-forward)
    ↓
Performance Evaluation
```

### 2.4 Validation Strategy

- **Train Period:** 2020-01-01 to 2022-12-31 (3 years)
- **Validation Period:** 2023-01-01 to 2023-12-31 (1 year)
- **Test Period:** 2024-01-01 to 2025-09-17 (out-of-sample)
- **Walk-Forward:** Re-train every 3 months on expanding window

---

## 3. Trading Strategy Specification

### 3.1 Signal Generation

**Model Forecast:**
```
forecast_t = Model.predict(X_t)
```
Where `X_t` includes:
- Options flow features (lagged 1 day to avoid lookahead)
- Momentum features (computed at EOD t)
- Mean reversion features (computed at EOD t)
- Volatility regime features (computed at EOD t)

**Signal Strength:**
```
signal_strength_t = |forecast_t| / std(forecast_train)
```

### 3.2 Risk Filters

**Filter 1: Volatility Regime Gate**
```python
if RV_5_t > 0.35:  # Extreme volatility
    exposure_multiplier *= 0.3
elif RV_5_t > 0.25:  # High volatility
    exposure_multiplier *= 0.6
elif RV_5_t < 0.10:  # Very low volatility (potential regime change)
    exposure_multiplier *= 0.7
```

**Filter 2: VIX Gate**
```python
if VIX_t > 40:  # Crisis mode
    exposure_multiplier *= 0.2
elif VIX_t > 30:  # Elevated fear
    exposure_multiplier *= 0.5
elif VIX_t < 12:  # Complacency
    exposure_multiplier *= 0.8
```

**Filter 3: Signal Strength Filter**
```python
if signal_strength_t < 0.5:  # Weak signal
    exposure_multiplier *= 0.3
elif signal_strength_t < 1.0:  # Moderate signal
    exposure_multiplier *= 0.6
```

**Filter 4: Drawdown Throttle (CRITICAL)**
```python
if current_drawdown < -0.12:  # Stop loss
    exposure = 0.0
elif current_drawdown < -0.08:  # Aggressive reduction
    exposure_multiplier *= 0.2
elif current_drawdown < -0.05:  # Moderate reduction
    exposure_multiplier *= 0.5
elif current_drawdown < -0.03:  # Light reduction
    exposure_multiplier *= 0.75
```

### 3.3 Position Sizing Rules

**Base Exposure:**
```python
base_exposure = forecast_t * EXPOSURE_SCALE
```

**Volatility Targeting:**
```python
target_vol = 0.18  # 18% annualized volatility target
realized_vol = rolling_std(returns, 252) * sqrt(252)
vol_scaling = target_vol / (realized_vol + 0.01)
vol_scaling = clip(vol_scaling, 0.5, 2.0)  # Bound scaling

exposure = base_exposure * vol_scaling
```

**Kelly Criterion (Conservative):**
```python
# Estimate win rate and avg win/loss from recent history
win_rate = mean(returns > 0, last_60_days)
avg_win = mean(returns[returns > 0], last_60_days)
avg_loss = abs(mean(returns[returns < 0], last_60_days))

if avg_loss > 0:
    kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
    exposure = exposure * kelly_fraction * 2  # Use half-Kelly
```

**Final Exposure:**
```python
exposure = base_exposure * vol_scaling * exposure_multiplier
exposure = clip(exposure, -1.0, 1.5)  # Hard limits
```

### 3.4 Regime Filters

**Regime Detection (3 states):**
1. **Trending:** Trend strength > 2.0, RSI(14) > 60 or < 40
2. **Mean Reverting:** |Z-score_20| > 1.5, trend strength < 1.0
3. **Neutral:** Everything else

**Regime-Specific Adjustments:**
```python
if regime == "Trending":
    # Favor momentum signals
    exposure *= 1.1 if forecast * momentum_5d > 0 else 0.9
elif regime == "Mean Reverting":
    # Favor contrarian signals
    exposure *= 1.1 if forecast * zscore_20 < 0 else 0.9
```

---

## 4. Modeling Approach

### 4.1 Model Selection

**Primary Model: Ensemble of Linear + Tree-Based**

**Components:**
1. **Ridge Regression** (L2 regularization, α=1.0)
   - Handles multicollinearity
   - Interpretable coefficients
   - Fast inference

2. **Lasso Regression** (L1 regularization, α=0.1)
   - Feature selection
   - Sparse solution
   - Good for high-dimensional data

3. **Random Forest** (100 trees, max_depth=10, min_samples_split=20)
   - Captures nonlinearities
   - Feature importance
   - Robust to outliers

4. **PCA-Ridge** (20 components, Ridge α=1.0)
   - Dimensionality reduction
   - Noise filtering
   - Robust to multicollinearity

**Ensemble Method:**
```python
forecast_ensemble = (
    0.25 * forecast_ridge +
    0.25 * forecast_lasso +
    0.35 * forecast_rf +
    0.15 * forecast_pca_ridge
)
```

### 4.2 Why This Approach Fits Calmar ≥ 1.5

1. **Diversification:** Ensemble reduces model risk and forecast variance
2. **Regularization:** Prevents overfitting, improves out-of-sample stability
3. **Feature Engineering:** Options flow + momentum + mean reversion captures multiple signal sources
4. **Risk Management:** Multi-layer filters control drawdowns aggressively
5. **Volatility Targeting:** Ensures consistent risk profile, reduces tail risk

### 4.3 Hyperparameter Tuning

**Grid Search on Validation Set:**
- Ridge α: [0.1, 0.5, 1.0, 2.0, 5.0]
- Lasso α: [0.01, 0.05, 0.1, 0.5, 1.0]
- RF max_depth: [5, 8, 10, 12, 15]
- RF min_samples_split: [10, 20, 30, 50]
- PCA components: [10, 15, 20, 25, 30]
- Ensemble weights: Optimize via validation set performance

**Objective:** Maximize Calmar Ratio (not just Sharpe)

---

## 5. Code Architecture

### 5.1 Directory Structure

```
QQQOPTIONS/
├── data/
│   ├── raw/
│   │   └── options_eod_QQQ.csv
│   ├── processed/
│   │   ├── options_eod_QQQ_processed.csv
│   │   └── qqq_options_features_dataset.csv
│   └── cache/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Load raw data
│   │   ├── preprocessor.py    # Clean and derive columns
│   │   └── feature_engineer.py # Build daily features
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Base model interface
│   │   ├── linear.py          # Ridge, Lasso
│   │   ├── tree.py            # Random Forest
│   │   ├── pca_ridge.py       # PCA-Ridge
│   │   └── ensemble.py        # Ensemble model
│   ├── risk/
│   │   ├── __init__.py
│   │   ├── exposure.py        # Forecast to exposure conversion
│   │   ├── sizing.py          # Position sizing rules
│   │   ├── filters.py         # Risk filters
│   │   └── drawdown.py        # Drawdown management
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py          # Backtest engine
│   │   ├── metrics.py         # Performance metrics
│   │   └── walk_forward.py    # Walk-forward validation
│   └── utils/
│       ├── __init__.py
│       ├── validation.py      # Time-series splits
│       └── plotting.py         # Visualization
├── scripts/
│   ├── preprocess_options.py
│   ├── build_daily_features.py
│   ├── train_and_backtest.py
│   └── tune_hyperparameters.py
├── config/
│   ├── model_config.yaml      # Model hyperparameters
│   ├── risk_config.yaml       # Risk management parameters
│   └── backtest_config.yaml   # Backtest settings
├── results/
│   ├── models/                # Saved model files
│   ├── backtests/             # Backtest results
│   └── plots/                 # Performance plots
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   └── test_risk.py
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
├── README.md
└── STRATEGY_DESIGN_CALMAR_1.5.md
```

### 5.2 Key Modules

**`src/risk/exposure.py`:**
- `forecast_to_exposure()`: Convert model forecast to position size
- `apply_risk_filters()`: Apply all risk filters
- `compute_volatility_scaling()`: Volatility targeting

**`src/risk/drawdown.py`:**
- `compute_drawdown()`: Calculate current drawdown
- `apply_drawdown_throttle()`: Reduce exposure during drawdowns
- `check_stop_loss()`: Hard stop logic

**`src/backtest/engine.py`:**
- `BacktestEngine`: Main backtest class
- `run_backtest()`: Execute backtest
- `compute_equity_curve()`: Calculate cumulative returns

---

## 6. Risk Management Framework

### 6.1 Max Drawdown Control Techniques

**Technique 1: Dynamic Drawdown Throttle**
```python
def apply_drawdown_throttle(exposure, current_drawdown, thresholds):
    """
    Aggressively reduce exposure as drawdown deepens.
    
    thresholds: {
        'stop_loss': -0.12,      # Exit completely
        'severe': -0.08,         # 20% of normal exposure
        'moderate': -0.05,        # 50% of normal exposure
        'light': -0.03            # 75% of normal exposure
    }
    """
    if current_drawdown < thresholds['stop_loss']:
        return 0.0
    elif current_drawdown < thresholds['severe']:
        return exposure * 0.2
    elif current_drawdown < thresholds['moderate']:
        return exposure * 0.5
    elif current_drawdown < thresholds['light']:
        return exposure * 0.75
    else:
        return exposure
```

**Technique 2: Trailing Stop**
```python
def trailing_stop(equity_curve, window=20, threshold=0.05):
    """
    Exit if equity drops 5% from 20-day high.
    """
    rolling_max = equity_curve.rolling(window).max()
    drawdown_from_peak = 1 - equity_curve / rolling_max
    return drawdown_from_peak > threshold
```

**Technique 3: Volatility-Adjusted Position Sizing**
```python
def volatility_targeting(exposure, realized_vol, target_vol=0.18):
    """
    Scale exposure to maintain constant volatility.
    """
    scaling = target_vol / (realized_vol + 0.01)
    scaling = np.clip(scaling, 0.5, 2.0)
    return exposure * scaling
```

**Technique 4: Correlation-Adjusted Exposure**
```python
def correlation_adjustment(exposure, correlation_with_market, threshold=0.7):
    """
    Reduce exposure when correlation with market is high (diversification loss).
    """
    if correlation_with_market > threshold:
        return exposure * (1 - (correlation_with_market - threshold) / (1 - threshold))
    return exposure
```

### 6.2 Volatility Targeting

**Implementation:**
```python
def compute_volatility_scaling(returns, target_vol=0.18, window=252):
    """
    Compute scaling factor to achieve target volatility.
    """
    realized_vol = returns.rolling(window).std() * np.sqrt(252)
    realized_vol = realized_vol.fillna(returns.std() * np.sqrt(252))
    scaling = target_vol / (realized_vol + 0.01)
    scaling = np.clip(scaling, 0.5, 2.0)  # Bound to prevent extreme scaling
    return scaling
```

**Rationale:** Maintains consistent risk profile, reduces tail risk, improves risk-adjusted returns.

### 6.3 Stop Logic

**Multi-Level Stops:**

1. **Hard Stop Loss:** Exit completely if drawdown > 12%
2. **Volatility Stop:** Exit if realized volatility > 50% (annualized)
3. **Signal Stop:** Exit if model confidence < 0.1 for 5 consecutive days
4. **Correlation Stop:** Exit if correlation with market > 0.95 (no diversification)

### 6.4 Kelly / Sqrt-Kelly Sizing

**Implementation:**
```python
def kelly_sizing(forecast, win_rate, avg_win, avg_loss, fraction=0.5):
    """
    Compute Kelly-optimal position size.
    Use fraction=0.5 for half-Kelly (more conservative).
    """
    if avg_loss == 0:
        return 0.0
    
    kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
    kelly = max(0, min(kelly, 0.25))  # Cap at 25%
    
    # Apply fraction (half-Kelly = 0.5)
    kelly = kelly * fraction
    
    # Scale by forecast strength
    position_size = forecast * kelly * 2  # Multiply by 2 to use full range
    
    return position_size
```

**Why Half-Kelly:** Full Kelly is too aggressive and can lead to large drawdowns. Half-Kelly provides better risk-adjusted returns.

### 6.5 Exposure Caps

**Hard Limits:**
- Maximum long exposure: 1.5x (150% leverage)
- Maximum short exposure: -1.0x (100% short)
- Maximum daily turnover: 0.8 (80% of portfolio)

**Soft Limits (Regime-Dependent):**
- Low volatility regime: Cap at 1.2x
- High volatility regime: Cap at 0.8x
- Drawdown period: Cap at 0.6x

### 6.6 Autocorrelation of Drawdown Checks

**Purpose:** Detect if drawdowns are clustering (bad sign) or mean-reverting (good sign).

**Implementation:**
```python
def check_drawdown_autocorrelation(drawdowns, lags=[1, 5, 10, 20]):
    """
    Check if drawdowns are autocorrelated (persistent).
    Negative autocorrelation is good (mean-reverting).
    Positive autocorrelation is bad (clustering).
    """
    results = {}
    for lag in lags:
        corr = drawdowns.autocorr(lag=lag)
        results[lag] = corr
        if corr > 0.3:  # High positive autocorrelation
            print(f"WARNING: Drawdowns are persistent at lag {lag} (corr={corr:.3f})")
    return results
```

**Action:** If positive autocorrelation detected, tighten risk controls.

---

## 7. Backtest Specification

### 7.1 Walk-Forward Validation

**Setup:**
- **Initial Training Window:** 3 years
- **Validation Window:** 1 year
- **Test Window:** 6 months
- **Retrain Frequency:** Every 3 months (expanding window)
- **Minimum Training Data:** 2 years

**Process:**
```python
for test_start in ['2024-01-01', '2024-07-01', '2025-01-01']:
    train_end = test_start - 1 day
    train_start = train_end - 3 years
    
    # Train on [train_start, train_end]
    model.fit(X_train, y_train)
    
    # Validate on [train_end + 1 day, train_end + 1 year]
    val_metrics = evaluate(X_val, y_val)
    
    # Test on [test_start, test_start + 6 months]
    test_metrics = evaluate(X_test, y_test)
    
    # Store results
    results.append({
        'train_period': (train_start, train_end),
        'val_period': (val_start, val_end),
        'test_period': (test_start, test_end),
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    })
```

### 7.2 Transaction Cost Modeling

**Cost Structure:**
```python
TRANSACTION_COST_BPS = 2.0  # 2 basis points per side
SLIPPAGE_BPS = 1.0          # 1 basis point slippage
TOTAL_COST_BPS = 3.0       # Total: 3 bps per trade

# For options (if trading options directly):
OPTIONS_SPREAD_BPS = 5.0   # Wider spreads for options
OPTIONS_COMMISSION = 0.65   # $0.65 per contract
```

**Implementation:**
```python
def apply_transaction_costs(returns, exposure_changes, cost_bps=3.0):
    """
    Apply transaction costs proportional to position changes.
    """
    costs = abs(exposure_changes) * (cost_bps / 10000)
    net_returns = returns - costs
    return net_returns
```

### 7.3 Out-of-Sample Expectations

**Realistic Expectations:**
- **CAGR:** 80-90% of in-sample (expect 14-20% vs 18% target)
- **Max Drawdown:** 110-120% of in-sample (expect 11-12% vs 10% target)
- **Calmar:** 70-80% of in-sample (expect 1.2-1.4 vs 1.5 target)
- **Sharpe:** 85-95% of in-sample

**Degradation Factors:**
1. Market regime changes
2. Options market microstructure changes
3. Increased competition (signal decay)
4. Model overfitting

### 7.4 Metrics Beyond Calmar

**Primary Metrics:**
1. **Calmar Ratio:** CAGR / |Max Drawdown| (target: ≥ 1.5)
2. **Sharpe Ratio:** (CAGR - Rf) / Volatility (target: ≥ 1.2)
3. **Sortino Ratio:** (CAGR - Rf) / Downside Volatility (target: ≥ 1.5)
4. **Max Drawdown:** Largest peak-to-trough decline (target: < 12%)

**Secondary Metrics:**
5. **Hit Rate:** % of profitable days (target: > 52%)
6. **Win/Loss Ratio:** Avg win / Avg loss (target: > 1.2)
7. **Turnover:** Avg daily |exposure change| (target: < 0.5)
8. **Volatility:** Annualized return volatility (target: 15-20%)
9. **Skewness:** Return distribution skew (target: > 0, positive skew)
10. **Kurtosis:** Return distribution tail (target: < 5, not too fat-tailed)

**Risk Metrics:**
11. **VaR (95%):** Value at Risk (target: < -2% daily)
12. **CVaR (95%):** Conditional VaR (target: < -3% daily)
13. **Drawdown Duration:** Avg days to recover (target: < 30 days)
14. **Recovery Factor:** Total return / Max Drawdown (target: > 1.5)

---

## 8. Tuning Guide

### 8.1 If Calmar < 1.5: Step-by-Step Tuning

**Step 1: Diagnose the Problem**

```python
# Check which component is failing
if CAGR < 0.15:
    print("Problem: Returns too low")
    action = "Increase exposure scale OR improve signal quality"
elif Max_Drawdown > 0.12:
    print("Problem: Drawdowns too large")
    action = "Tighten risk controls OR reduce exposure"
elif Calmar < 1.5 and CAGR > 0.15 and Max_DD < 0.12:
    print("Problem: Returns and DD both OK, but ratio low")
    action = "Fine-tune balance between returns and risk"
```

**Step 2: If Returns Too Low**

**Action A: Increase Exposure Scale**
```python
# Current: EXPOSURE_SCALE = 5.0
# Try: EXPOSURE_SCALE = 6.0, 7.0, 8.0
# But: Monitor drawdowns closely!
```

**Action B: Improve Signal Quality**
- Add more features (intraday momentum, cross-asset signals)
- Try different models (XGBoost, Neural Networks)
- Ensemble more models
- Filter weak signals more aggressively (only trade strong signals)

**Action C: Reduce Transaction Costs**
- Reduce turnover (increase EMA smoothing)
- Trade less frequently (only on strong signals)

**Step 3: If Drawdowns Too Large**

**Action A: Tighten Drawdown Throttle**
```python
# Current thresholds
DRAWDOWN_PROTECTION_THRESHOLD = -0.065
DRAWDOWN_REDUCTION_FACTOR = 0.55
STOP_LOSS_THRESHOLD = -0.18

# Tighter (more aggressive)
DRAWDOWN_PROTECTION_THRESHOLD = -0.05  # Trigger earlier
DRAWDOWN_REDUCTION_FACTOR = 0.4        # Reduce more
STOP_LOSS_THRESHOLD = -0.12            # Stop earlier
```

**Action B: Reduce Maximum Exposure**
```python
# Current: MAX_EXPOSURE = 1.35
# Try: MAX_EXPOSURE = 1.2, 1.1, 1.0
```

**Action C: Increase Volatility Adjustment**
```python
# Current: VOLATILITY_ADJUSTMENT_STRENGTH = 0.88
# Try: VOLATILITY_ADJUSTMENT_STRENGTH = 0.95, 1.0
```

**Action D: Add More Risk Filters**
- Filter by VIX level more aggressively
- Filter by signal strength more aggressively
- Add correlation filter
- Add regime filter

**Step 4: If Balance is Off**

**Action A: Fine-Tune Exposure Scale**
```python
# Use validation set to find optimal scale
for scale in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    metrics = backtest(scale=scale)
    calmar = metrics['Calmar']
    print(f"Scale {scale}: Calmar={calmar:.2f}")
    
# Select scale that maximizes Calmar
```

**Action B: Adjust Risk Filter Thresholds**
```python
# Tune each filter independently
for vix_thresh in [25, 30, 35, 40]:
    for vol_thresh in [0.20, 0.25, 0.30]:
        metrics = backtest(vix_thresh=vix_thresh, vol_thresh=vol_thresh)
        # Select combination that maximizes Calmar
```

**Step 5: Iterative Refinement**

```python
# Repeat Steps 2-4 until Calmar ≥ 1.5
# But: Don't overfit to validation set!
# Use walk-forward validation to ensure robustness
```

### 8.2 Tuning Checklist

- [ ] Check if CAGR is sufficient (target: > 15%)
- [ ] Check if Max Drawdown is acceptable (target: < 12%)
- [ ] Tune exposure scale on validation set
- [ ] Tune drawdown throttle thresholds
- [ ] Tune volatility adjustment strength
- [ ] Tune risk filter thresholds (VIX, volatility, signal strength)
- [ ] Optimize model hyperparameters
- [ ] Test on out-of-sample period
- [ ] Verify Calmar ≥ 1.5 on test set
- [ ] Check for overfitting (validation vs test performance)

---

## 9. Python Implementation

### 9.1 Core Backtest Engine

```python
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class CalmarOptimizedStrategy:
    """
    Trading strategy optimized for Calmar Ratio ≥ 1.5.
    """
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.exposure_scale = config.get('EXPOSURE_SCALE', 5.0)
        self.target_vol = config.get('TARGET_VOL', 0.18)
        
        # Risk management parameters
        self.drawdown_threshold = config.get('DRAWDOWN_THRESHOLD', -0.065)
        self.drawdown_reduction = config.get('DRAWDOWN_REDUCTION', 0.55)
        self.stop_loss = config.get('STOP_LOSS', -0.18)
        self.max_exposure = config.get('MAX_EXPOSURE', 1.35)
        self.min_exposure = config.get('MIN_EXPOSURE', -0.85)
        
    def train_models(self, X_train, y_train):
        """Train ensemble of models."""
        print("Training models...")
        
        # Ridge
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        self.models['Ridge'] = ridge
        
        # Lasso
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        self.models['Lasso'] = lasso
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                   min_samples_split=20, random_state=42)
        rf.fit(X_train, y_train)
        self.models['RandomForest'] = rf
        
        # PCA-Ridge
        pca = PCA(n_components=20)
        X_train_pca = pca.fit_transform(X_train)
        ridge_pca = Ridge(alpha=1.0)
        ridge_pca.fit(X_train_pca, y_train)
        self.models['PCA'] = (pca, ridge_pca)
        
        print("✓ Models trained")
        
    def predict_ensemble(self, X):
        """Generate ensemble forecast."""
        forecasts = {}
        
        # Ridge
        forecasts['Ridge'] = self.models['Ridge'].predict(X)
        
        # Lasso
        forecasts['Lasso'] = self.models['Lasso'].predict(X)
        
        # Random Forest
        forecasts['RandomForest'] = self.models['RandomForest'].predict(X)
        
        # PCA-Ridge
        pca, ridge_pca = self.models['PCA']
        X_pca = pca.transform(X)
        forecasts['PCA'] = ridge_pca.predict(X_pca)
        
        # Ensemble (weighted average)
        ensemble = (
            0.25 * forecasts['Ridge'] +
            0.25 * forecasts['Lasso'] +
            0.35 * forecasts['RandomForest'] +
            0.15 * forecasts['PCA']
        )
        
        return ensemble
    
    def forecast_to_exposure(self, forecast, prev_exposure, current_rv, 
                             current_drawdown, signal_strength):
        """
        Convert model forecast to position size with aggressive risk management.
        """
        # Base exposure
        exposure = forecast * self.exposure_scale
        
        # Volatility targeting
        vol_scaling = self.target_vol / (current_rv + 0.01)
        vol_scaling = np.clip(vol_scaling, 0.5, 2.0)
        exposure = exposure * vol_scaling
        
        # Risk filters
        # Filter 1: Signal strength
        if signal_strength < 0.5:
            exposure *= 0.3
        elif signal_strength < 1.0:
            exposure *= 0.6
        
        # Filter 2: Drawdown throttle (CRITICAL for Calmar)
        if current_drawdown < self.stop_loss:
            exposure = 0.0
        elif current_drawdown < -0.08:
            exposure *= 0.2
        elif current_drawdown < -0.05:
            exposure *= 0.5
        elif current_drawdown < -0.03:
            exposure *= 0.75
        
        # Clip to bounds
        exposure = np.clip(exposure, self.min_exposure, self.max_exposure)
        
        # Turnover control
        max_turnover = 0.85
        exposure_change = exposure - prev_exposure
        if abs(exposure_change) > max_turnover:
            exposure = prev_exposure + np.sign(exposure_change) * max_turnover
        
        return exposure
    
    def backtest(self, X, y, dates, data_df):
        """
        Run backtest with proper risk management.
        """
        n = len(X)
        exposure = np.zeros(n)
        equity = np.ones(n)
        returns = np.zeros(n)
        drawdowns = np.zeros(n)
        
        # Compute signal strength from training data
        train_forecasts = self.predict_ensemble(X[:n//2])  # Use first half as proxy
        signal_std = np.std(train_forecasts)
        
        for i in range(1, n):
            # Get forecast
            forecast = self.predict_ensemble(X[i:i+1])[0]
            signal_strength = abs(forecast) / (signal_std + 1e-8)
            
            # Get current state
            prev_exposure = exposure[i-1]
            current_rv = data_df.iloc[i]['rv_5'] if 'rv_5' in data_df.columns else 0.20
            current_drawdown = drawdowns[i-1]
            
            # Convert to exposure
            exposure[i] = self.forecast_to_exposure(
                forecast, prev_exposure, current_rv, 
                current_drawdown, signal_strength
            )
            
            # Compute return (use actual 1-day return)
            actual_return = y.iloc[i] if hasattr(y, 'iloc') else y[i]
            strategy_return = exposure[i-1] * actual_return
            
            # Transaction costs
            exposure_change = abs(exposure[i] - exposure[i-1])
            cost = exposure_change * 0.0003  # 3 bps per trade
            strategy_return -= cost
            
            # Update equity
            equity[i] = equity[i-1] * (1 + strategy_return)
            returns[i] = strategy_return
            
            # Update drawdown
            peak = equity[:i+1].max()
            drawdowns[i] = 1 - equity[i] / peak if peak > 0 else 0.0
        
        # Create results dataframe
        results = pd.DataFrame({
            'date': dates,
            'exposure': exposure,
            'equity': equity,
            'return': returns,
            'drawdown': drawdowns
        })
        results.set_index('date', inplace=True)
        
        return results
    
    def compute_metrics(self, results):
        """Compute performance metrics."""
        returns = results['return']
        equity = results['equity']
        
        # Basic metrics
        total_return = equity.iloc[-1] - 1.0
        n_years = (results.index[-1] - results.index[0]).days / 365.25
        cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
        
        # Drawdown
        peak = equity.expanding().max()
        drawdown = 1 - equity / peak
        max_drawdown = drawdown.min()
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Ratios
        sharpe = (cagr - 0.02) / volatility if volatility > 0 else 0
        calmar = abs(cagr / max_drawdown) if max_drawdown < 0 else 0
        
        # Sortino (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino = (cagr - 0.02) / downside_vol if downside_vol > 0 else 0
        
        # Hit rate
        hit_rate = (returns > 0).mean()
        
        # Turnover
        exposure_changes = results['exposure'].diff().abs()
        turnover = exposure_changes.mean()
        
        metrics = {
            'CAGR': cagr,
            'Total Return': total_return,
            'Max Drawdown': max_drawdown,
            'Calmar': calmar,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Volatility': volatility,
            'Hit Rate': hit_rate,
            'Turnover': turnover
        }
        
        return metrics
```

### 9.2 Main Execution Script

```python
# train_and_backtest.py

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from CalmarOptimizedStrategy import CalmarOptimizedStrategy

# Load data
print("Loading data...")
data = pd.read_csv('qqq_options_features_dataset.csv', parse_dates=['tradeDate'], index_col='tradeDate')
data = data.sort_index()

# Prepare features and target
feature_cols = [c for c in data.columns if c not in ['close', 'ret_1d_fwd', 'ret_1d_fwd_vol_adj']]
X = data[feature_cols].fillna(0)
y = data['ret_1d_fwd']

# Drop last row (NaN forward return)
X = X.iloc[:-1]
y = y.iloc[:-1]
data = data.iloc[:-1]

# Time-series split
train_end = '2022-12-31'
val_end = '2023-12-31'

X_train = X[X.index <= train_end]
y_train = y[y.index <= train_end]
X_val = X[(X.index > train_end) & (X.index <= val_end)]
y_val = y[(y.index > train_end) & (y.index <= val_end)]
X_test = X[X.index > val_end]
y_test = y[y.index > val_end]

print(f"Train: {len(X_train)} samples ({X_train.index[0]} to {X_train.index[-1]})")
print(f"Val: {len(X_val)} samples ({X_val.index[0]} to {X_val.index[-1]})")
print(f"Test: {len(X_test)} samples ({X_test.index[0]} to {X_test.index[-1]})")

# Configuration
config = {
    'EXPOSURE_SCALE': 5.0,
    'TARGET_VOL': 0.18,
    'DRAWDOWN_THRESHOLD': -0.065,
    'DRAWDOWN_REDUCTION': 0.55,
    'STOP_LOSS': -0.18,
    'MAX_EXPOSURE': 1.35,
    'MIN_EXPOSURE': -0.85
}

# Initialize strategy
strategy = CalmarOptimizedStrategy(config)

# Train models
strategy.train_models(X_train.values, y_train.values)

# Backtest on validation set
print("\nBacktesting on validation set...")
val_results = strategy.backtest(
    X_val.values, y_val, X_val.index, data.loc[X_val.index]
)
val_metrics = strategy.compute_metrics(val_results)
print("\nValidation Metrics:")
for k, v in val_metrics.items():
    if isinstance(v, float):
        if 'Return' in k or 'Drawdown' in k:
            print(f"  {k}: {v:.2%}")
        else:
            print(f"  {k}: {v:.4f}")

# Tune exposure scale on validation set
print("\nTuning exposure scale...")
best_calmar = val_metrics['Calmar']
best_scale = config['EXPOSURE_SCALE']

for scale in [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    strategy.exposure_scale = scale
    test_results = strategy.backtest(
        X_val.values, y_val, X_val.index, data.loc[X_val.index]
    )
    test_metrics = strategy.compute_metrics(test_results)
    print(f"  Scale {scale:.1f}: Calmar={test_metrics['Calmar']:.2f}, "
          f"CAGR={test_metrics['CAGR']:.2%}, DD={test_metrics['Max Drawdown']:.2%}")
    
    if test_metrics['Calmar'] > best_calmar:
        best_calmar = test_metrics['Calmar']
        best_scale = scale

print(f"\nBest scale: {best_scale:.1f} (Calmar: {best_calmar:.2f})")
strategy.exposure_scale = best_scale

# Final backtest on test set
print("\nBacktesting on test set...")
test_results = strategy.backtest(
    X_test.values, y_test, X_test.index, data.loc[X_test.index]
)
test_metrics = strategy.compute_metrics(test_results)

print("\n" + "="*60)
print("Full Period Performance:")
print("="*60)
for k, v in test_metrics.items():
    if isinstance(v, float):
        if 'Return' in k or 'Drawdown' in k:
            print(f"{k}: {v:.2%}")
        else:
            print(f"{k}: {v:.4f}")

# Check if target met
if test_metrics['Calmar'] >= 1.5:
    print("\n✓ TARGET MET: Calmar ≥ 1.5")
else:
    print(f"\n✗ TARGET NOT MET: Calmar = {test_metrics['Calmar']:.2f} < 1.5")
    print("  See tuning guide (Section 8) for improvement steps.")

# Save results
test_results.to_csv('backtest_results_full_period.csv')
print("\n✓ Results saved to backtest_results_full_period.csv")
```

---

## 10. Monitoring & Deployment

### 10.1 Real-Time Risk Controls

**Daily Monitoring:**
1. **Exposure Check:** Verify current exposure is within limits
2. **Drawdown Check:** Alert if drawdown > 5%
3. **Volatility Check:** Alert if realized vol > 30%
4. **Signal Quality Check:** Alert if model confidence < 0.3 for 3+ days
5. **Correlation Check:** Alert if correlation with market > 0.9

**Automated Actions:**
```python
def real_time_risk_control(current_state):
    """
    Real-time risk monitoring and automated actions.
    """
    alerts = []
    actions = []
    
    # Check drawdown
    if current_state['drawdown'] < -0.05:
        alerts.append("WARNING: Drawdown > 5%")
        if current_state['drawdown'] < -0.08:
            actions.append("REDUCE_EXPOSURE_50%")
        if current_state['drawdown'] < -0.12:
            actions.append("STOP_TRADING")
    
    # Check volatility
    if current_state['realized_vol'] > 0.30:
        alerts.append("WARNING: High volatility")
        actions.append("REDUCE_EXPOSURE_30%")
    
    # Check signal quality
    if current_state['signal_strength'] < 0.3:
        alerts.append("WARNING: Weak signal")
        actions.append("REDUCE_EXPOSURE_40%")
    
    return alerts, actions
```

### 10.2 Deployment Plan

**Phase 1: Paper Trading (1 month)**
- Run strategy in simulation mode
- Monitor all metrics daily
- Verify risk controls are working
- Tune parameters if needed

**Phase 2: Small Capital (1 month)**
- Deploy with 10% of target capital
- Monitor execution quality
- Verify transaction costs match assumptions
- Check for any implementation bugs

**Phase 3: Scale Up (Gradual)**
- Increase to 50% of target capital
- Monitor for 2 months
- If performance matches backtest, scale to 100%

**Phase 4: Production**
- Full capital deployment
- Daily monitoring and reporting
- Weekly performance reviews
- Monthly model retraining

### 10.3 Performance Monitoring Dashboard

**Key Metrics to Track:**
- Real-time equity curve
- Current exposure and position
- Current drawdown
- Realized volatility (rolling 20-day)
- Model forecast and confidence
- Risk filter status (which filters are active)
- Transaction costs (actual vs expected)

**Alerts:**
- Email/SMS if drawdown > 5%
- Email/SMS if Calmar drops below 1.2
- Email/SMS if model confidence < 0.2 for 5 days
- Daily summary email with key metrics

---

## Conclusion

This strategy design provides a comprehensive framework for achieving Calmar Ratio ≥ 1.5 through:

1. **Robust Feature Engineering:** Options flow + momentum + mean reversion + volatility regime
2. **Ensemble Modeling:** Multiple models reduce forecast variance
3. **Aggressive Risk Management:** Multi-layer filters control drawdowns
4. **Volatility Targeting:** Maintains consistent risk profile
5. **Proper Validation:** Walk-forward testing ensures out-of-sample robustness

**Expected Outcome:**
- **CAGR:** 15-20%
- **Max Drawdown:** 10-12%
- **Calmar Ratio:** 1.5-1.8
- **Sharpe Ratio:** 1.2-1.5

**Next Steps:**
1. Implement the code architecture
2. Run full backtest with walk-forward validation
3. Tune parameters to achieve Calmar ≥ 1.5
4. Deploy with proper monitoring and risk controls

---

**End of Document**

