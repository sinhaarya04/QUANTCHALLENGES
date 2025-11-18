import pandas as pd
import numpy as np

CONTRACT_SIZE = 100  # standard US equity options

# Load the CSV file
print("Loading options data...")
df = pd.read_csv(
    "options_eod_QQQ.csv",
    parse_dates=["tradeDate", "expirDate"]
)

print(f"Loaded {len(df):,} rows")
print(f"Date range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")

# Basic quality filters
print("\nApplying quality filters...")
initial_count = len(df)
df = df[df["stockPrice"] > 0]
df = df[df["dte"].between(1, 365)]  # ignore expired/silly tenors
print(f"Filtered from {initial_count:,} to {len(df):,} rows ({initial_count - len(df):,} removed)")

# --- Derived columns ---

print("\nComputing derived columns...")

# Moneyness
df["moneyness"] = df["strike"] / df["stockPrice"]
df["log_moneyness"] = np.log(df["moneyness"])

# Define tenor bucket function
def tenor_bucket(d):
    if d <= 7:
        return "1_7"
    elif d <= 30:
        return "8_30"
    elif d <= 90:
        return "31_90"
    else:
        return "90_plus"

# Define moneyness bucket function
def moneyness_bucket(m):
    if m < 0.9:
        return "deep_put"
    elif m < 0.98:
        return "otm_put"
    elif m <= 1.02:
        return "atm"
    elif m <= 1.1:
        return "otm_call"
    else:
        return "deep_call"

# Apply bucket functions
df["tenor_bucket"] = df["dte"].apply(tenor_bucket)
df["moneyness_bucket"] = df["moneyness"].apply(moneyness_bucket)

# Mid prices
df["call_mid"] = (df["callBidPrice"] + df["callAskPrice"]) / 2
df["put_mid"] = (df["putBidPrice"] + df["putAskPrice"]) / 2

# Notional volumes (dollar-ish)
df["call_notional_vol"] = df["callVolume"] * df["stockPrice"] * CONTRACT_SIZE
df["put_notional_vol"] = df["putVolume"] * df["stockPrice"] * CONTRACT_SIZE

# Notional OI
df["call_notional_oi"] = df["callOpenInterest"] * df["stockPrice"] * CONTRACT_SIZE
df["put_notional_oi"] = df["putOpenInterest"] * df["stockPrice"] * CONTRACT_SIZE

# Total notional volume and OI
df["total_notional_vol"] = df["call_notional_vol"] + df["put_notional_vol"]
df["total_notional_oi"] = df["call_notional_oi"] + df["put_notional_oi"]

# Notional volume / OI ratios
df["call_vol_oi_ratio"] = df["call_notional_vol"] / (df["call_notional_oi"] + 1e-8)
df["put_vol_oi_ratio"] = df["put_notional_vol"] / (df["put_notional_oi"] + 1e-8)
df["total_vol_oi_ratio"] = df["total_notional_vol"] / (df["total_notional_oi"] + 1e-8)

# Greeks-based exposures (dealer-style proxies)
# gamma is same sign for calls/puts
oi_total = df["callOpenInterest"] + df["putOpenInterest"]

df["gamma_exposure"] = df["gamma"] * oi_total * (df["stockPrice"]**2) * CONTRACT_SIZE
df["vega_exposure"] = df["vega"] * oi_total * CONTRACT_SIZE

# Delta exposures
# If delta is call Δ, then put Δ ≈ call Δ - 1 (Black–Scholes parity)
df["call_delta"] = df["delta"]
df["put_delta"] = df["delta"] - 1.0

# Delta exposure (notional)
df["call_delta_exposure"] = df["call_delta"] * df["callOpenInterest"] * df["stockPrice"] * CONTRACT_SIZE
df["put_delta_exposure"] = df["put_delta"] * df["putOpenInterest"] * df["stockPrice"] * CONTRACT_SIZE
df["total_delta_exposure"] = df["call_delta_exposure"] + df["put_delta_exposure"]

print("\nDerived columns created:")
print(f"  - moneyness, log_moneyness, moneyness_bucket")
print(f"  - tenor_bucket")
print(f"  - call_mid, put_mid")
print(f"  - call_notional_vol, put_notional_vol, total_notional_vol")
print(f"  - call_notional_oi, put_notional_oi, total_notional_oi")
print(f"  - call_vol_oi_ratio, put_vol_oi_ratio, total_vol_oi_ratio")
print(f"  - gamma_exposure, vega_exposure")
print(f"  - call_delta, put_delta")
print(f"  - call_delta_exposure, put_delta_exposure, total_delta_exposure")

# --- START: NEW FEATURES BLOCK (PCR Only) ---

print("\nCalculating new features (PCR)...")

# 1. Create new Put-Call Ratios (Volume)
# We already have notional volumes per contract, let's group them by date and buckets
pcr_by_tenor = df.groupby(['tradeDate', 'tenor_bucket']).agg(
    call_vol=('call_notional_vol', 'sum'),
    put_vol=('put_notional_vol', 'sum')
).unstack(fill_value=1e-8) # Use 1e-8 to avoid zero division, pivot

pcr_by_moneyness = df.groupby(['tradeDate', 'moneyness_bucket']).agg(
    call_vol=('call_notional_vol', 'sum'),
    put_vol=('put_notional_vol', 'sum')
).unstack(fill_value=1e-8) # Pivot

# Calculate ratios and store in new daily DataFrame
daily_pcr_features = pd.DataFrame(index=pcr_by_tenor.index)

# PCR by Tenor
for tenor in df['tenor_bucket'].unique():
    daily_pcr_features[f'pcr_vol_tenor_{tenor}'] = \
        pcr_by_tenor[('put_vol', tenor)] / pcr_by_tenor[('call_vol', tenor)]

# PCR by Moneyness
for money in df['moneyness_bucket'].unique():
    daily_pcr_features[f'pcr_vol_money_{money}'] = \
        pcr_by_moneyness[('put_vol', money)] / pcr_by_moneyness[('call_vol', money)]

# Merge PCR features back into the main DataFrame
df = df.merge(daily_pcr_features, on='tradeDate', how='left')
print(f"  - pcr_vol_tenor_... (Daily)")
print(f"  - pcr_vol_money_... (Daily)")

# --- END: NEW FEATURES BLOCK ---

# Save the processed data
output_file = "options_eod_QQQ_processed.csv"
print(f"\nSaving processed data to {output_file}...")
df.to_csv(output_file, index=False)
print(f"Saved {len(df):,} rows to {output_file}")

# Display summary statistics
print("\n=== Summary Statistics ===")
print(f"\nTenor bucket distribution:")
print(df["tenor_bucket"].value_counts().sort_index())
print(f"\nMoneyness bucket distribution:")
print(df["moneyness_bucket"].value_counts().sort_index())

print(f"\nDate range: {df['tradeDate'].min()} to {df['tradeDate'].max()}")
print(f"Unique dates: {df['tradeDate'].nunique()}")
print(f"Unique strikes per date (avg): {df.groupby('tradeDate')['strike'].nunique().mean():.1f}")

print(f"\nNotional volume stats (daily avg):")
daily_vol = df.groupby("tradeDate")["total_notional_vol"].sum()
print(f"  Mean: ${daily_vol.mean():,.0f}")
print(f"  Median: ${daily_vol.median():,.0f}")
print(f"  Max: ${daily_vol.max():,.0f}")

print(f"\nNotional OI stats (daily avg):")
daily_oi = df.groupby("tradeDate")["total_notional_oi"].sum()
print(f"  Mean: ${daily_oi.mean():,.0f}")
print(f"  Median: ${daily_oi.median():,.0f}")
print(f"  Max: ${daily_oi.max():,.0f}")

print(f"\nGamma exposure stats (daily sum):")
daily_gamma = df.groupby("tradeDate")["gamma_exposure"].sum()
print(f"  Mean: {daily_gamma.mean():,.0f}")
print(f"  Median: {daily_gamma.median():,.0f}")

print(f"\nVega exposure stats (daily sum):")
daily_vega = df.groupby("tradeDate")["vega_exposure"].sum()
print(f"  Mean: {daily_vega.mean():,.0f}")
print(f"  Median: {daily_vega.median():.0f}")

print("\nDone!")
