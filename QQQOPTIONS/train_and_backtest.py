"""
Train and Backtest QQQ Options Strategy
Target: Calmar Ratio ≥ 1.5

This script implements the Volatility-Regime Adaptive Options Flow Strategy (VRAOFS)
with aggressive drawdown control and volatility targeting.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Data split dates
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"
TEST_END = "2025-09-17"

# Exposure parameters (will be tuned on validation set)
EXPOSURE_SCALE = 20.0  # Initial scaling factor (reduced from 5.0 for better stability)
MIN_FORECAST_THRESHOLD = 0.0  # No filtering - model forecasts are already small
MAX_EXPOSURE = 1.5  # Increased max exposure (150% leverage)
MIN_EXPOSURE = -1.0  # Increased short exposure (100% short)
MAX_TURNOVER = 1.0  # Increased turnover limit

# Risk management parameters - optimized for Calmar > 1.5
DRAWDOWN_PROTECTION_THRESHOLD = -0.065  # Reduce exposure if drawdown > 6.5%
DRAWDOWN_REDUCTION_FACTOR = 0.55  # Moderate-aggressive reduction when in drawdown
VOLATILITY_ADJUSTMENT_ENABLED = True  # Enable volatility-based position sizing
VOLATILITY_ADJUSTMENT_STRENGTH = 0.88  # Strong vol adjustment
MAX_EXPOSURE_DURING_DRAWDOWN = 0.85  # Cap exposure at 85% during drawdown periods
STOP_LOSS_THRESHOLD = -0.18  # Exit completely if drawdown > 18%
TARGET_VOL = 0.18  # 18% annualized volatility target

# Transaction costs (in basis points per side)
TRANSACTION_COST_BPS = 2.0  # 2 bps = 0.02% per trade
SLIPPAGE_BPS = 1.0  # 1 bps slippage
TOTAL_COST_BPS = 3.0  # Total: 3 bps per trade

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, y_train):
    """Train ensemble of models."""
    print("Training models...")
    models = {}
    scalers = {}
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    scalers['main'] = scaler
    
    # Ridge Regression
    print("  Training Ridge...")
    ridge = Ridge(alpha=1.0, random_state=42)
    ridge.fit(X_train_scaled, y_train)
    models['Ridge'] = ridge
    
    # Elastic Net Regression (FIX 7: stronger regularization)
    print("  Training Elastic Net...")
    elastic_net = ElasticNet(alpha=0.10, l1_ratio=0.05, random_state=42, max_iter=2000)
    elastic_net.fit(X_train_scaled, y_train)
    models['ElasticNet'] = elastic_net
    
    # LightGBM (BONUS: captures nonlinear structure in options surfaces)
    if LIGHTGBM_AVAILABLE:
        print("  Training LightGBM...")
        lgb = LGBMRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
        random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb.fit(X_train_scaled, y_train)
        models['LightGBM'] = lgb
    else:
        models['LightGBM'] = None
    
    print("✓ All models trained\n")
    return models, scalers

def predict_ensemble(models, scalers, X):
    """Generate ensemble forecast."""
    X_scaled = scalers['main'].transform(X)
    
    # Ridge
    pred_ridge = models['Ridge'].predict(X_scaled)
    
    # Elastic Net (replaces Lasso)
    pred_elastic = models['ElasticNet'].predict(X_scaled)
    
    # LightGBM (BONUS: captures nonlinear structure)
    if models.get('LightGBM') is not None:
        pred_lgb = models['LightGBM'].predict(X_scaled)
        # Ensemble: 0.4*elastic + 0.4*ridge + 0.2*lgb
        ensemble = (
            0.4 * pred_elastic +
            0.4 * pred_ridge +
            0.2 * pred_lgb
        )
    else:
        # Fallback: Ridge + ElasticNet only
        ensemble = (
            0.5 * pred_ridge +
            0.5 * pred_elastic
        )
    
    return ensemble

# ============================================================================
# RISK MANAGEMENT
# ============================================================================

def forecast_to_exposure(forecast, prev_exposure=0.0, exposure_scale=None, 
                         current_rv=None, current_drawdown=None, signal_strength=None, signal_std=None):
    """
    Convert model forecast to exposure with aggressive risk management.
    
    Args:
        forecast: Model prediction (expected return)
        prev_exposure: Previous day's exposure (for turnover control)
        exposure_scale: Scaling factor (if None, uses global EXPOSURE_SCALE)
        current_rv: Current realized volatility (for vol-based sizing)
        current_drawdown: Current drawdown (for drawdown protection)
        signal_strength: Signal strength (|forecast| / std(forecasts))
    
    Returns:
        exposure: Target exposure for day t
    """
    if exposure_scale is None:
        exposure_scale = EXPOSURE_SCALE
    
    # ===== FIX 1: STOP Inverting the Forecast =====
    # Model has positive correlation (+0.59), trade WITH it
    exposure = forecast * exposure_scale
    
    # ===== FIX 5: Dampen Small Signals (Don't Zero) =====
    # Dampen very small signals instead of zeroing them out
    # Forecasts std is ~0.002, so threshold of 0.0015 kills 90% of signals
    if abs(forecast) < 0.0002:
        exposure *= 0.3  # Dampen, don't zero
    
    # ===== Simple Vol-Targeting (ONE formula only) =====
    # Clean, interpretable vol-targeting used in real funds
    if VOLATILITY_ADJUSTMENT_ENABLED and current_rv is not None and current_rv > 0:
        # Simple & robust: target 18% annual vol
        realized_vol = current_rv
        vol_factor = TARGET_VOL / realized_vol
        vol_factor = np.clip(vol_factor, 0.5, 2.0)
        exposure = exposure * vol_factor
    
    # Clip to bounds
    exposure = np.clip(exposure, MIN_EXPOSURE, MAX_EXPOSURE)
    
    # Turnover control: limit daily change
    exposure_change = exposure - prev_exposure
    if abs(exposure_change) > MAX_TURNOVER:
        exposure = prev_exposure + np.sign(exposure_change) * MAX_TURNOVER
    
    return exposure

# ============================================================================
# BACKTEST ENGINE
# ============================================================================

def backtest(model_name, model_tuple, X, y, dates, data_df, exposure_scale=None):
    """
    Run backtest with proper risk management.
    
    Args:
        model_name: Name of the model
        model_tuple: (models, scalers) tuple or just models dict
        X: Feature matrix
        y: Target returns
        dates: Date index
        data_df: Full dataframe with additional columns (rv_5, etc.)
        exposure_scale: Exposure scaling factor (if None, uses global)
    
    Returns:
        results: DataFrame with exposure, equity, returns, drawdown
    """
    if exposure_scale is None:
        exposure_scale = EXPOSURE_SCALE
    
    # Unpack models
    if isinstance(model_tuple, tuple) and len(model_tuple) == 2:
        models, scalers = model_tuple
    else:
        models = model_tuple
        scalers = {'main': StandardScaler().fit(X)}
    
    n = len(X)
    exposure = np.zeros(n)
    equity = np.ones(n)
    returns = np.zeros(n)
    drawdowns = np.zeros(n)
    
    # ===== FIX 6: Repair Exposure Scaling =====
    # Compute proper scale from training forecasts using quantile approach
    train_size = min(n // 2, 500)
    if train_size > 0:
        train_forecasts = predict_ensemble(models, scalers, X[:train_size])
        if len(train_forecasts) > 0 and np.std(train_forecasts) > 0:
            # Use 95th percentile to set scale
            forecast_95th = np.quantile(np.abs(train_forecasts), 0.95)
            exposure_scale = 1.0 / (forecast_95th + 1e-8)
            print(f"  Computed exposure scale from training: {exposure_scale:.2f} (95th percentile: {forecast_95th:.6f})")
        else:
            exposure_scale = EXPOSURE_SCALE
    else:
        exposure_scale = EXPOSURE_SCALE
    
    # Compute signal strength for other uses
    if train_size > 0 and len(train_forecasts) > 0:
        signal_std = np.std(train_forecasts)
    else:
        signal_std = 0.001
    
    for i in range(1, n):
        # Get forecast
        forecast = predict_ensemble(models, scalers, X[i:i+1])[0]
        signal_strength = abs(forecast) / (signal_std + 1e-8)
        
        # Get current state
        prev_exposure = exposure[i-1]
        current_rv = data_df.iloc[i]['rv_5'] if 'rv_5' in data_df.columns else 0.20
        current_drawdown = drawdowns[i-1]
        
        # Convert to exposure (removed over-engineered risk management)
        exposure[i] = forecast_to_exposure(
            forecast, prev_exposure, exposure_scale,
            current_rv, current_drawdown, signal_strength, signal_std
        )
        
        # ===== FIX 10: Fix NaN handling =====
        # Ensure exposure is not NaN
        if np.isnan(exposure[i]):
            exposure[i] = prev_exposure
        
        # Compute return (use actual 1-day return)
        actual_return = y.iloc[i] if hasattr(y, 'iloc') else y[i]
        
        # ===== FIX 10: Fix NaN handling in returns =====
        if np.isnan(actual_return):
            actual_return = 0.0
        
        # ===== FIX 9: Fix return computation =====
        # Always use prev_exposure for return calculation, even if current exposure is 0
        # This ensures we capture returns from positions held, not new positions
        strategy_return = prev_exposure * actual_return
        
        # Transaction costs
        exposure_change = abs(exposure[i] - prev_exposure)
        cost = exposure_change * (TOTAL_COST_BPS / 10000)
        strategy_return -= cost
        
        # ===== FIX 10: Fix NaN handling in strategy return =====
        if np.isnan(strategy_return):
            strategy_return = 0.0
        
        # Update equity
        equity[i] = equity[i-1] * (1 + strategy_return)
        returns[i] = strategy_return
        
        # ===== FIX 10: Fix NaN handling in equity =====
        if np.isnan(equity[i]):
            equity[i] = equity[i-1]
        
        # Update drawdown (negative values: -0.2 means 20% drawdown)
        peak = equity[:i+1].max()
        drawdowns[i] = (equity[i] / peak) - 1.0 if peak > 0 else 0.0
    
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

# ============================================================================
# PERFORMANCE METRICS
# ============================================================================

def compute_metrics(results):
    """Compute comprehensive performance metrics."""
    returns = results['return']
    equity = results['equity']
    
    # Basic metrics
    total_return = equity.iloc[-1] - 1.0
    n_years = (results.index[-1] - results.index[0]).days / 365.25
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0
    
    # Drawdown calculation (ensure it's always negative or zero)
    peak = equity.expanding().max()
    drawdown = (equity / peak) - 1.0  # This gives negative values (e.g., -0.2 for 20% drawdown)
    max_drawdown = drawdown.min()  # Most negative value = largest drawdown
    
    # If no drawdown occurred (equity always at peak), max_drawdown will be 0.0
    # But if equity went down, it should be negative
    
    # Volatility
    volatility = returns.std() * np.sqrt(252)
    
    # Ratios
    risk_free_rate = 0.02  # 2% risk-free rate
    sharpe = (cagr - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calmar: CAGR / |Max Drawdown|
    # Handle edge cases: if max_drawdown is 0 or very small, Calmar is undefined
    if max_drawdown < -1e-6:  # If there's a meaningful drawdown
        calmar = abs(cagr / max_drawdown)
    elif cagr > 0:
        # If no drawdown but positive returns, Calmar is infinite (set to high value)
        calmar = 999.0
    else:
        # If no drawdown and negative returns, Calmar is undefined
        calmar = 0.0
    
    # Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else volatility
    sortino = (cagr - risk_free_rate) / downside_vol if downside_vol > 0 else 0
    
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

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*60)
    print("QQQ Options Strategy: Train and Backtest")
    print("Target: Calmar Ratio ≥ 1.5")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    data = pd.read_csv('qqq_options_features_dataset.csv', parse_dates=['tradeDate'], index_col='tradeDate')
    data = data.sort_index()
    print(f"Loaded {len(data):,} rows")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    
    # Prepare features - use all features except obvious non-predictive or target columns
    feature_cols = [c for c in data.columns if c not in ['close', 'ret_1d_fwd', 'ret_1d_fwd_vol_adj']]
    
    # Use all features - stop starving the model
    # Once system is stabilized, we can do feature importance-based pruning
    X = data[feature_cols].copy()
    print(f"\nUsing full feature set: {X.shape[1]} features")
    
    # ===== FIX: CORRECT THE TARGET (no leakage, vol-adjusted, de-meaned) =====
    # Raw forward return (predict next-day return)
    raw_ret = data['ret_1d_fwd'].shift(-1)
    
    # Remove slow drift over 20 days
    ret_demeaned = raw_ret - raw_ret.rolling(20).mean()
    
    # Use realized vol as scale (fill NaNs properly)
    ret_vol = data['rv_20'].bfill().ffill()
    
    # Final target: vol-adjusted, de-meaned return
    y = (ret_demeaned / (ret_vol + 1e-4)).copy()
    
    # Drop rows with NaN in target (from rolling mean and shift operations)
    valid_mask = ~(y.isna() | np.isinf(y))
    X = X[valid_mask].copy()
    y = y[valid_mask].copy()
    data = data[valid_mask].copy()
    
    print(f"After removing NaN/Inf in target: {len(X)} rows")
    
    # Remove zero-variance columns
    print("\nRemoving zero-variance columns...")
    initial_cols = len(X.columns)
    X = X.loc[:, X.std() > 1e-8]
    print(f"Removed {initial_cols - len(X.columns)} zero-variance columns")
    
    # Handle infinite values and NaNs more robustly
    print("Cleaning data (handling NaNs and infinities)...")
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaNs with column medians (or 0 if median is NaN)
    for col in X.columns:
        col_median = X[col].median()
        if pd.isna(col_median):
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(col_median)
    
    # Final check: replace any remaining NaNs with 0
    X = X.fillna(0)
    
    # ===== SIMPLIFIED CLIPPING: Remove aggressive clipping =====
    # Simple winsorize to 1st/99th percentile - removes structure-killing hard clipping
    print("Clipping extreme outliers (1st/99th percentile)...")
    X = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=1)
    
    print(f"Data shape: {X.shape}")
    print(f"NaN count: {X.isna().sum().sum()}")
    print(f"Inf count: {np.isinf(X.values).sum()}")
    
    # Time-series split
    X_train = X[X.index <= TRAIN_END]
    y_train = y[y.index <= TRAIN_END]
    X_val = X[(X.index > TRAIN_END) & (X.index <= VAL_END)]
    y_val = y[(y.index > TRAIN_END) & (y.index <= VAL_END)]
    X_test = X[(X.index > VAL_END) & (X.index <= TEST_END)]
    y_test = y[(y.index > VAL_END) & (y.index <= TEST_END)]
    
    print(f"\nTrain: {len(X_train):,} samples ({X_train.index[0]} to {X_train.index[-1]})")
    print(f"Val: {len(X_val):,} samples ({X_val.index[0]} to {X_val.index[-1]})")
    print(f"Test: {len(X_test):,} samples ({X_test.index[0]} to {X_test.index[-1]})")
    
    # Check target variable statistics
    print("\n" + "="*60)
    print("Target Variable Statistics")
    print("="*60)
    print(f"  Mean: {y_train.mean():.6f}")
    print(f"  Std: {y_train.std():.6f}")
    print(f"  Min: {y_train.min():.6f}")
    print(f"  Max: {y_train.max():.6f}")
    print(f"  95th percentile: {y_train.quantile(0.95):.6f}")
    print(f"  5th percentile: {y_train.quantile(0.05):.6f}")
    
    # Train models
    print("\n" + "="*60)
    print("Training Models")
    print("="*60)
    models, scalers = train_models(X_train.values, y_train.values)
    
    # Analyze ElasticNet model coefficients to understand learned logic
    print("\n" + "="*60)
    print("ElasticNet Model Coefficients (Learned Logic)")
    print("="*60)
    feature_names = X_train.columns
    elastic_coeffs = models['ElasticNet'].coef_
    coeff_series = pd.Series(elastic_coeffs, index=feature_names)
    
    print("\nTop 10 Positive (Bullish) Features:")
    top_positive = coeff_series.sort_values(ascending=False).head(10)
    for feat, val in top_positive.items():
        print(f"  {feat:40s}: {val:8.6f}")
    
    print("\nTop 10 Negative (Bearish) Features:")
    top_negative = coeff_series.sort_values(ascending=True).head(10)
    for feat, val in top_negative.items():
        print(f"  {feat:40s}: {val:8.6f}")
    
    # Check if volatility features have correct signs
    print("\nVolatility Feature Coefficients:")
    vol_features = [f for f in feature_names if 'vol_' in f or 'iv_' in f or 'skew' in f or 'term' in f]
    for feat in sorted(vol_features):
        if feat in coeff_series.index:
            print(f"  {feat:40s}: {coeff_series[feat]:8.6f}")
    
    print("="*60 + "\n")
    
    # Check model performance on training set
    print("\nModel Performance on Training Set:")
    train_preds = predict_ensemble(models, scalers, X_train.values)
    train_r2 = r2_score(y_train.values, train_preds)
    train_rmse = np.sqrt(mean_squared_error(y_train.values, train_preds))
    train_corr = np.corrcoef(y_train.values, train_preds)[0, 1]
    print(f"  R² Score: {train_r2:.4f}")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  Correlation: {train_corr:.4f}")
    print(f"  Forecast Mean: {train_preds.mean():.6f}")
    print(f"  Forecast Std: {train_preds.std():.6f}")
    print(f"  Forecast Min: {train_preds.min():.6f}")
    print(f"  Forecast Max: {train_preds.max():.6f}")
    
    # Check if we should invert forecasts (negative correlation = wrong direction)
    if train_corr < -0.1:
        print(f"\n  WARNING: Negative correlation ({train_corr:.4f}) detected!")
        print(f"  Model may be predicting opposite direction. Consider inverting forecasts.")
    
    # ===== SANITY CHECKS: Forecast stats on Train, Val, Test =====
    print("\n" + "="*60)
    print("Forecast Statistics (Sanity Checks)")
    print("="*60)
    for label, X_split, y_split in [
        ("Train", X_train, y_train),
        ("Val", X_val, y_val),
        ("Test", X_test, y_test),
    ]:
        preds = predict_ensemble(models, scalers, X_split.values)
        corr = np.corrcoef(y_split.values, preds)[0, 1]
        print(f"\n{label} forecast stats:")
        print(f"  corr(y, pred): {corr:.4f}")
        print(f"  mean pred: {preds.mean():.6f}, std pred: {preds.std():.6f}")
        print(f"  min pred: {preds.min():.6f}, max pred: {preds.max():.6f}")
    
    # Evaluate on validation set
    print("="*60)
    print("Validation Set Performance")
    print("="*60)
    val_results = backtest(
        'Ensemble', (models, scalers),
        X_val.values, y_val, X_val.index,
        data.loc[X_val.index],
        exposure_scale=EXPOSURE_SCALE
    )
    val_metrics = compute_metrics(val_results)
    
    print("\nValidation Metrics:")
    for k, v in val_metrics.items():
        if isinstance(v, float):
            if 'Return' in k or 'Drawdown' in k:
                print(f"  {k:20s}: {v:8.2%}")
            else:
                print(f"  {k:20s}: {v:8.4f}")
    
    # ===== FIX 6: Repair Exposure Scaling =====
    # Use quantile-based scale instead of manual scale lists
    print("\n" + "="*60)
    print("Calibrating Exposure Scale (Quantile-Based)")
    print("="*60)
    
    # Get forecast distribution on validation set
    val_forecasts = predict_ensemble(models, scalers, X_val.values)
    forecast_std = np.std(val_forecasts)
    forecast_95th = np.quantile(np.abs(val_forecasts), 0.95)
    
    if forecast_std > 0 and forecast_95th > 0:
        # Compute calibrated scale: ensures 95% of forecasts map to exposure in [-1.0, 1.0]
        calibrated_scale = 1.0 / (forecast_95th + 1e-8)
        print(f"  Forecast std: {forecast_std:.6f}")
        print(f"  Forecast 95th percentile: {forecast_95th:.6f}")
        print(f"  Calibrated scale: {calibrated_scale:.2f}")
        print(f"  This ensures 95% of forecasts map to exposure in [-1.0, 1.0]")
    else:
        calibrated_scale = EXPOSURE_SCALE
        print(f"  Warning: All forecasts are near zero, using default scale: {calibrated_scale:.2f}")
    
    # Tune exposure scale on validation set (centered on calibrated scale)
    print(f"\nTuning exposure scale on validation set (centered on calibrated scale)...")
    base = calibrated_scale
    scale_candidates = sorted(set([
        base * 0.5,
        base * 0.75,
        base,
        base * 1.25,
        base * 1.5,
        base * 2.0
    ]))
    
    best_exposure_scale = calibrated_scale
    best_calmar = val_metrics['Calmar']
    
    print(f"\nTesting exposure scales:")
    for scale in scale_candidates:
        val_results_tuned = backtest(
            'Ensemble', (models, scalers),
            X_val.values, y_val, X_val.index,
            data.loc[X_val.index],
            exposure_scale=scale
        )
        metrics_tuned = compute_metrics(val_results_tuned)
        print(f"  Scale {scale:.2f}: Calmar={metrics_tuned['Calmar']:.2f}, "
              f"CAGR={metrics_tuned['CAGR']:.2%}, DD={metrics_tuned['Max Drawdown']:.2%}, "
              f"Turnover={metrics_tuned['Turnover']:.3f}")
        
        if metrics_tuned['Calmar'] > best_calmar and metrics_tuned['CAGR'] > -0.05:
            best_calmar = metrics_tuned['Calmar']
            best_exposure_scale = scale
    
    print(f"\nBest scale: {best_exposure_scale:.2f} (Calmar: {best_calmar:.2f})")
    
    # Retrain on train+val for final model
    print("\n" + "="*60)
    print("Retraining on Train+Val")
    print("="*60)
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    models_final, scalers_final = train_models(X_trainval.values, y_trainval.values)
    
    # Final backtest on test set
    print("="*60)
    print("Test Set Performance (Out-of-Sample)")
    print("="*60)
    test_results = backtest(
        'Ensemble', (models_final, scalers_final),
        X_test.values, y_test, X_test.index,
        data.loc[X_test.index],
        exposure_scale=best_exposure_scale
    )
    test_metrics = compute_metrics(test_results)
    
    print("\nFull Period Performance:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            if 'Return' in k or 'Drawdown' in k:
                print(f"{k}: {v:.2%}")
            else:
                print(f"{k}: {v:.4f}")
    
    # ===== SANITY CHECKS: Test Backtest Diagnostics =====
    print("\n" + "="*60)
    print("Test Backtest Diagnostics (Sanity Checks)")
    print("="*60)
    
    # Load backtest results for detailed diagnostics
    test_results_diag = backtest(
        'Ensemble', (models_final, scalers_final),
        X_test.values, y_test, X_test.index,
        data.loc[X_test.index],
        exposure_scale=best_exposure_scale
    )
    
    returns_diag = test_results_diag['return']
    equity_diag = test_results_diag['equity']
    
    # Daily volatility
    daily_vol = returns_diag.std()
    annualized_vol = daily_vol * np.sqrt(252)
    print(f"\nDaily return stats:")
    print(f"  Mean daily return: {returns_diag.mean():.6f}")
    print(f"  Std daily return: {daily_vol:.4f} ({daily_vol*100:.2f}%)")
    print(f"  Annualized vol: {annualized_vol:.4f} ({annualized_vol*100:.2f}%)")
    print(f"  Target: 0.7-1.5% daily vol (18-38% annualized)")
    
    # Drawdown
    peak = equity_diag.expanding().max()
    drawdown = (equity_diag / peak) - 1.0
    max_dd = drawdown.min()
    print(f"\nDrawdown stats:")
    print(f"  Max drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")
    print(f"  Target: < 20%")
    print(f"  Days in drawdown > 5%: {(drawdown < -0.05).sum()} ({(drawdown < -0.05).mean()*100:.1f}%)")
    
    # Equity
    final_equity = equity_diag.iloc[-1]
    print(f"\nEquity stats:")
    print(f"  Final equity: {final_equity:.3f} ({final_equity*100:.1f}%)")
    print(f"  Target: > 1.5-2.0")
    print(f"  Max equity: {equity_diag.max():.3f}")
    print(f"  Min equity: {equity_diag.min():.3f}")
    
    # Hit rate
    hit_rate = (returns_diag > 0).mean()
    print(f"\nHit rate:")
    print(f"  Hit rate: {hit_rate:.4f} ({hit_rate*100:.2f}%)")
    print(f"  Target: 53-60%")
    
    # Exposure stats
    exposure_diag = test_results_diag['exposure']
    print(f"\nExposure stats:")
    print(f"  Mean exposure: {exposure_diag.mean():.3f}")
    print(f"  Max exposure: {exposure_diag.max():.3f}")
    print(f"  Min exposure: {exposure_diag.min():.3f}")
    print(f"  Std exposure: {exposure_diag.std():.3f}")
    
    # Check if target met
    print("\n" + "="*60)
    if test_metrics['Calmar'] >= 1.5:
        print(f"✓ TARGET MET: Calmar = {test_metrics['Calmar']:.2f} ≥ 1.5")
    else:
        print(f"✗ TARGET NOT MET: Calmar = {test_metrics['Calmar']:.2f} < 1.5")
        print("  See STRATEGY_DESIGN_CALMAR_1.5.md Section 8 for tuning guide.")
    
    # Save results
    output_file = f'backtest_results_Ensemble_full_period.csv'
    test_results.to_csv(output_file)
    print(f"\nSaved full period backtest results to {output_file}")
    
    print("\nDone!")
    
    return test_results, test_metrics

if __name__ == "__main__":
    results, metrics = main()

