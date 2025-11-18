"""
Build Daily Features and Final Dataset
======================================

This script:
1. Aggregates processed options data into daily features
2. Adds underlying QQQ returns and realized volatility
3. Adds momentum and market structure features
4. Creates the final dataset for modeling
"""

import pandas as pd
import numpy as np

def daily_agg(day_df: pd.DataFrame) -> pd.Series:
    """
    Aggregate daily options data into features.
    Includes all the features from preprocess_options.py plus aggregations.
    """
    out = {}
    
    # ===== Global flow / positioning =====
    call_vol = day_df["call_notional_vol"].sum()
    put_vol = day_df["put_notional_vol"].sum()
    out["pc_vol_ratio"] = put_vol / call_vol if call_vol > 0 else np.nan
    
    call_oi = day_df["call_notional_oi"].sum()
    put_oi = day_df["put_notional_oi"].sum()
    out["pc_oi_ratio"] = put_oi / call_oi if call_oi > 0 else np.nan
    
    # out["gamma_exposure_total"] = day_df["gamma_exposure"].sum() # BUGGY: Causes e+160 overflow
    # out["vega_exposure_total"] = day_df["vega_exposure"].sum() # BUGGY: Causes e+160 overflow
    
    # ===== FIX 2: Dealer Flow Features =====
    # Total delta exposure (already computed in preprocess)
    total_delta = day_df["total_delta_exposure"].sum()
    out["total_delta_exposure"] = total_delta
    
    # Delta flow (1d and 5d changes) - will be computed after aggregation
    # Gamma exposure normalized (to avoid overflow)
    gamma_total = day_df["gamma_exposure"].sum()
    spot_price = day_df["stockPrice"].iloc[0] if len(day_df) > 0 else 1.0
    out["gamma_exposure_norm"] = gamma_total / (spot_price**2 * 1e12) if spot_price > 0 else 0.0
    
    # Gamma flow (change in gamma exposure) - will be computed after aggregation
    # Vanna approximation (delta sensitivity to volatility)
    # Approximate vanna as: delta_change / vol_change proxy
    # For now, use a simple proxy based on vega and delta
    vega_total = day_df["vega_exposure"].sum()
    out["vanna_approx"] = (total_delta * vega_total) / (spot_price * 1e12) if spot_price > 0 else 0.0
    
    # ===== Short-term vs longer-term sentiment =====
    short = day_df[day_df["tenor_bucket"] == "1_7"]
    mid = day_df[day_df["tenor_bucket"] == "8_30"]
    long_term = day_df[day_df["tenor_bucket"] == "31_90"]
    very_long = day_df[day_df["tenor_bucket"] == "90_plus"]
    
    for label, sub in [("1_7", short), ("8_30", mid), ("31_90", long_term), ("90_plus", very_long)]:
        cvol = sub["call_notional_vol"].sum()
        pvol = sub["put_notional_vol"].sum()
        out[f"pc_vol_ratio_{label}"] = pvol / cvol if cvol > 0 else np.nan
        
        coi = sub["call_notional_oi"].sum()
        poi = sub["put_notional_oi"].sum()
        out[f"pc_oi_ratio_{label}"] = poi / coi if coi > 0 else np.nan
    
    # ===== Skew proxy using vega / volumes across moneyness =====
    mid_8_30 = mid.copy()
    otm_puts = mid_8_30[mid_8_30["moneyness_bucket"] == "otm_put"]
    atm = mid_8_30[mid_8_30["moneyness_bucket"] == "atm"]
    otm_calls = mid_8_30[mid_8_30["moneyness_bucket"] == "otm_call"]
    
    # vega-based skew (rough proxy for IV skew demand)
    vega_otm_put = otm_puts["vega_exposure"].sum()
    vega_atm = atm["vega_exposure"].sum()
    vega_otm_call = otm_calls["vega_exposure"].sum()
    
    out["vega_otm_put_8_30"] = vega_otm_put
    out["vega_atm_8_30"] = vega_atm
    out["vega_otm_call_8_30"] = vega_otm_call
    
    # Normalized vega features (FIX 2)
    vega_total_8_30 = vega_otm_put + vega_atm + vega_otm_call
    out["vega_otm_put_8_30_norm"] = vega_otm_put / (abs(vega_total_8_30) + 1e8) if abs(vega_total_8_30) > 0 else 0.0
    out["vega_atm_8_30_norm"] = vega_atm / (abs(vega_total_8_30) + 1e8) if abs(vega_total_8_30) > 0 else 0.0
    
    out["skew_put_atm_vega_8_30"] = vega_otm_put - vega_atm
    out["skew_put_call_vega_8_30"] = vega_otm_put - vega_otm_call
    
    # Skew slope (FIX 2)
    out["skew_slope_8_30"] = (vega_otm_put - vega_otm_call) / (abs(vega_atm) + 1e8) if abs(vega_atm) > 0 else 0.0
    
    # ===== Term structure proxy =====
    for bucket in ["1_7", "8_30", "31_90", "90_plus"]:
        sub = day_df[day_df["tenor_bucket"] == bucket]
        out[f"vega_total_{bucket}"] = sub["vega_exposure"].sum()
    
    out["vega_term_slope_short_long"] = out["vega_total_8_30"] - out["vega_total_1_7"]
    
    # ===== NEW: Extract PCR features by tenor and moneyness =====
    # These are already per-day, so just take the first value
    pcr_tenor_cols = [c for c in day_df.columns if c.startswith('pcr_vol_tenor_')]
    pcr_money_cols = [c for c in day_df.columns if c.startswith('pcr_vol_money_')]
    
    for col in pcr_tenor_cols + pcr_money_cols:
        if len(day_df) > 0:
            out[col] = day_df[col].iloc[0]
        else:
            out[col] = np.nan
    
    return pd.Series(out)


def main():
    print("="*60)
    print("Building Daily Features and Final Dataset")
    print("="*60)
    
    # Load processed options data
    print("\nLoading processed options data...")
    df = pd.read_csv(
        "options_eod_QQQ_processed.csv",
        parse_dates=["tradeDate", "expirDate"]
    )
    
    print(f"Loaded {len(df):,} rows")
    print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
    print(f"Unique dates: {df['tradeDate'].nunique()}")
    
    # Aggregate to daily features
    print("\nAggregating to daily features...")
    daily_features = df.groupby("tradeDate", group_keys=False).apply(daily_agg).sort_index()
    
    print(f"Created {len(daily_features):,} daily feature rows")
    print(f"Number of features: {len(daily_features.columns)}")
    
    # Build underlying returns & realized volatility
    print("\n" + "="*60)
    print("Building Underlying Returns & Realized Volatility")
    print("="*60)
    
    # Extract QQQ stock price
    qqq = (
        df.groupby("tradeDate")["stockPrice"]
        .first()
        .to_frame(name="close")
        .sort_index()
    )
    
    print(f"Extracted {len(qqq):,} daily QQQ prices")
    
    # Forward returns
    qqq["ret_1d_fwd"] = qqq["close"].pct_change().shift(-1)
    
    # Realized volatility
    print("\nComputing realized volatility features...")
    rets = qqq["close"].pct_change()
    for w in [5, 10, 20]:
        qqq[f"rv_{w}"] = rets.rolling(w).std() * np.sqrt(252)
        print(f"  - rv_{w}: {w}-day rolling realized vol (annualized)")
    
    # Volatility-adjusted return
    qqq["ret_1d_fwd_vol_adj"] = qqq["ret_1d_fwd"] / (qqq["rv_5"] + 0.01)
    
    # Add momentum and market structure features
    print("\nAdding momentum and market structure features...")
    
    # Past returns
    for lag in [1, 2, 3, 5, 10, 20]:
        qqq[f"ret_lag_{lag}"] = qqq["close"].pct_change(lag)
    
    # Moving averages
    for window in [5, 10, 20, 50]:
        qqq[f"ma_{window}"] = qqq["close"].rolling(window).mean()
        qqq[f"price_ma_{window}_ratio"] = qqq["close"] / qqq[f"ma_{window}"] - 1.0
    
    # RSI
    for window in [5, 14]:
        delta = qqq["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        qqq[f"rsi_{window}"] = 100 - (100 / (1 + rs))
    
    # Volatility regime
    qqq["vol_regime_low"] = (qqq["rv_5"] < 0.15).astype(float)
    qqq["vol_regime_normal"] = ((qqq["rv_5"] >= 0.15) & (qqq["rv_5"] <= 0.25)).astype(float)
    qqq["vol_regime_high"] = (qqq["rv_5"] > 0.25).astype(float)
    
    # Trend strength
    for window in [10, 20]:
        high_low = qqq["close"].rolling(window).max() - qqq["close"].rolling(window).min()
        price_range = qqq["close"].rolling(window).std()
        qqq[f"trend_strength_{window}"] = high_low / (price_range + 1e-8)
    
    # Z-scores
    for window in [10, 20]:
        ma = qqq["close"].rolling(window).mean()
        std = qqq["close"].rolling(window).std()
        qqq[f"zscore_{window}"] = (qqq["close"] - ma) / (std + 1e-8)
    
    # ===== NEW: Enhanced regime features =====
    # Vol regime (binary)
    qqq["vol_regime"] = (qqq["rv_5"] > qqq["rv_5"].median()).astype(int)
    
    # Trend regime (200D SMA)
    qqq["ma_200"] = qqq["close"].rolling(200).mean()
    qqq["trend_regime"] = (qqq["close"] > qqq["ma_200"]).astype(int)
    
    # Vol ratio (20D / 60D)
    qqq["rv_60"] = rets.rolling(60).std() * np.sqrt(252)
    qqq["vol_ratio_20_60"] = qqq["rv_20"] / (qqq["rv_60"] + 1e-8)
    
    # Join everything together
    print("\nJoining QQQ data with options features...")
    data = qqq.join(daily_features, how="inner")
    print(f"After join: {len(data):,} rows")
    
    # ===== FIX 2: Add delta flow and gamma flow features =====
    # Compute flow features (changes over time)
    if "total_delta_exposure" in data.columns:
        data["delta_flow_1d"] = data["total_delta_exposure"].diff(1)
        data["delta_flow_5d"] = data["total_delta_exposure"].diff(5)
    
    if "gamma_exposure_norm" in data.columns:
        data["gamma_flow"] = data["gamma_exposure_norm"].diff(1)
    
    # ===== NEW: Add lagged features (1-5 days) =====
    print("\nAdding lagged features (1-5 days)...")
    
    # Lag target return
    for lag in [1, 2, 3, 5]:
        data[f"ret_1d_fwd_lag{lag}"] = data["ret_1d_fwd"].shift(lag)
    
    # Lag key predictive features
    key_features_to_lag = [
        "pc_vol_ratio", "pc_oi_ratio", "rv_5", "rv_10", "rv_20",
        "skew_put_call_vega_8_30", "vega_term_slope_short_long",
        "pcr_vol_tenor_8_30", "pcr_vol_money_atm"
    ]
    
    for feature in key_features_to_lag:
        if feature in data.columns:
            for lag in [1, 2, 3, 5]:
                data[f"{feature}_lag{lag}"] = data[feature].shift(lag)
    
    print(f"  - Added lagged target returns (1, 2, 3, 5 days)")
    print(f"  - Added lagged key features (1, 2, 3, 5 days)")
    
    # Drop rows with missing data
    print("\nDropping rows with missing data...")
    initial_rows = len(data)
    data = data.dropna()
    print(f"Dropped {initial_rows - len(data):,} rows with missing data")
    print(f"Final dataset: {len(data):,} rows")
    
    # Save final dataset
    output_file = "qqq_options_features_dataset.csv"
    print(f"\nSaving final dataset to {output_file}...")
    data.to_csv(output_file)
    print(f"Saved {len(data):,} rows to {output_file}")
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Date range: {data.index.min()} to {data.index.max()}")
    print(f"Total features: {len(data.columns)}")
    print(f"\nKey features included:")
    print(f"  - Options flow: pc_vol_ratio, pc_oi_ratio, gamma/vega exposures")
    print(f"  - Volatility skew proxies (vega-based)")
    print(f"  - Term structure proxies (vega-based)")
    print(f"  - PCR by tenor/moneyness: pcr_vol_tenor_..., pcr_vol_money_...")
    print(f"  - Underlying returns: ret_1d_fwd, ret_1d_fwd_vol_adj")
    print(f"  - Realized volatility: rv_5, rv_10, rv_20")
    print(f"  - Momentum: ret_lag_*, ma_*, rsi_*")
    print(f"  - Market structure: trend_strength_*, zscore_*")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

