import pandas as pd
import numpy as np
import os
import glob

# Set name cryptocurrency
cryptocurrency = 'BTC'

# Clean original data
def clean_and_compute_returns(df, price_col='close', time_col='date', outlier_thresh=10):
    # Load the data
    df = pd.read_csv(df, skiprows=1)

    # Parse timestamps
    # Explicitly convert timestamp column
    df[time_col] = pd.to_datetime(df[time_col], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    # Remove zero or missing prices
    df = df[df[price_col] > 0].dropna(subset=[price_col])

    # Remove duplicated timestamps
    df = df[~df.index.duplicated(keep='first')]

    # Resample to 1-minute intervals and forward fill prices
    df = df.resample('1min').last()
    df[price_col] = df[price_col].ffill()

    # Compute log returns
    df['log_return'] = np.log(df[price_col] / df[price_col].shift(1))

    return df

all_dfs = []

print("Step 1")

# Find all CSV files in the current directory
for file_path in glob.glob("*.csv"):
    print(f"Processing: {file_path}")
    raw_df = file_path
    try:
        cleaned_df = clean_and_compute_returns(raw_df)
        all_dfs.append(cleaned_df)
    except Exception as e:
        print(f"Skipping {file_path} due to error: {e}")

# Combine all cleaned data
combined_df = pd.concat(all_dfs).sort_index()

# Preview result
print(combined_df.head())
print(f"Total rows after combining: {len(combined_df)}")

combined_df = combined_df[~combined_df.index.duplicated(keep='first')]

print("Step 2")

# Drop the first day entirely
first_day = combined_df.index.min().date()
combined_df = combined_df[combined_df.index.date > first_day]

# Define final start and end timestamps
start_timestamp = combined_df.index.min().normalize()  # First midnight
end_timestamp = pd.Timestamp('2024-12-31 23:59:00')

# Trim anything after this point
combined_df = combined_df.loc[start_timestamp:end_timestamp]

# Optional: Save result
combined_df.to_csv(f"Bitstamp_{cryptocurrency}USD_1minute.csv")

print("Step 3")

print(combined_df.index.min())
print(combined_df.index.max())
print(combined_df.head())

min_price = combined_df['close'].min()
print(f"📉 Minimum price: {min_price}")
max_price = combined_df['close'].max()
print(f"📉 Maximum price: {max_price}")

min_log_return = combined_df['log_return'].min()
print(f"📉 Minimum log_return: {min_log_return}")
max_log_return = combined_df['log_return'].max()
print(f"📉 Maximum log_return: {max_log_return}")

expected_range = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='1min')
missing = expected_range.difference(combined_df.index)
print(f"Missing timestamps: {len(missing)}")

print("Missing timestamps:")
for ts in missing:
    print(ts)

# Adding missing values
# Ensure datetime index and sorted
combined_df.index = pd.to_datetime(combined_df.index)
combined_df = combined_df.sort_index()

# Define the full range of timestamps you expect — 1-minute frequency here
full_index = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='1min')

# Reindex to include all missing timestamps
combined_df = combined_df.reindex(full_index)

missing_times = combined_df[combined_df['close'].isna()].index

# Forward-fill price and any other desired columns
combined_df['close'] = combined_df['close'].fillna(method='ffill')

expected_range = pd.date_range(start=combined_df.index.min(), end=combined_df.index.max(), freq='1min')
missing = expected_range.difference(combined_df.index)
print(f"Missing timestamps: {len(missing)}")

print("Missing timestamps:")
for ts in missing:
    print(ts)

dup_counts = combined_df.index.value_counts()
duplicate_timestamps = dup_counts[dup_counts > 1]
print("Duplicate timestamps and their counts:")
print(duplicate_timestamps)

print("Step 4")
# From here different timeframes
# 5 minute
# Resample to 5-minute intervals
df_5min = combined_df.resample('5min').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_5min['log_return'] = np.log(df_5min['close'] / df_5min['close'].shift(1))

print(df_5min.head())
# Save or use further
df_5min.to_csv(f"Bitstamp_{cryptocurrency}USD_5minute.csv")

# 10 minute
# Resample to 10-minute intervals
df_10min = combined_df.resample('10min').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_10min['log_return'] = np.log(df_10min['close'] / df_10min['close'].shift(1))

print(df_10min.head())
# Save or use further
df_10min.to_csv(f"Bitstamp_{cryptocurrency}USD_10minute.csv")

# 15 minute
# Resample to 15-minute intervals
df_15min = combined_df.resample('15min').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_15min['log_return'] = np.log(df_15min['close'] / df_15min['close'].shift(1))

print(df_15min.head())
# Save or use further
df_15min.to_csv(f"Bitstamp_{cryptocurrency}USD_15minute.csv")

# 30 minute
# Resample to 30-minute intervals
df_30min = combined_df.resample('30min').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_30min['log_return'] = np.log(df_30min['close'] / df_30min['close'].shift(1))

print(df_30min.head())
# Save or use further
df_30min.to_csv(f"Bitstamp_{cryptocurrency}USD_30minute.csv")

# 1 hour
# Resample to 1 hour intervals
df_1h = combined_df.resample('1h').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_1h['log_return'] = np.log(df_1h['close'] / df_1h['close'].shift(1))

print(df_1h.head())
# Save or use further
df_1h.to_csv(f"Bitstamp_{cryptocurrency}USD_1hour.csv")

# 4 hour
# Resample to 4 hour intervals
df_4h = combined_df.resample('4h').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_4h['log_return'] = np.log(df_4h['close'] / df_4h['close'].shift(1))

expected_range = pd.date_range(start=df_4h.index.min(), end=df_4h.index.max(), freq='4h')
missing = expected_range.difference(df_4h.index)
print(f"Missing timestamps: {len(missing)}")

print("Missing timestamps:")
for ts in missing:
    print(ts)

print(df_4h.head())
# Save or use further
df_4h.to_csv(f"Bitstamp_{cryptocurrency}USD_4hours.csv")

# 12 hour
# Resample to 12 hour intervals
df_12h = combined_df.resample('12h').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_12h['log_return'] = np.log(df_12h['close'] / df_12h['close'].shift(1))

print(df_12h.head())
# Save or use further
df_12h.to_csv(f"Bitstamp_{cryptocurrency}USD_12hours.csv")

# 24 hour
# Resample to 24 hour intervals
df_24h = combined_df.resample('24h').agg({
    'symbol': 'last',
    'open': 'first',
    'low': 'min',
    'high': 'max',
    'close': 'last',
    'Volume USD': 'sum',
    f'Volume {cryptocurrency}': 'sum'
})

# Add log return column
df_24h['log_return'] = np.log(df_24h['close'] / df_24h['close'].shift(1))

print(df_24h.head())
# Save or use further
df_24h.to_csv(f"Bitstamp_{cryptocurrency}USD_24hours.csv")

# Delete all CSVs in the root of Colab environment
for file in glob.glob("*.csv"):
    os.remove(file)
    print(f"Deleted: {file}")

lowest_return_row = combined_df['log_return'].idxmin()
print(f"Timestamp with lowest return: {lowest_return_row}")
print("Full row details:")
print(combined_df.loc[lowest_return_row])