import glob
import os
import itertools
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import concurrent.futures
import multiprocessing
from tqdm import tqdm

# Prepare values
def prepare_moving_averages(df, param_grid):
    unique_p = sorted(set(p for p, _, _, _ in param_grid))
    unique_q = sorted(set(q for _, q, _, _ in param_grid))

    # Compute moving averages first, store in dicts
    ma_short_dict = {}
    for p in unique_p:
        ma_short_dict[p] = df['price'].rolling(p).mean()

    ma_long_dict = {}
    for q in unique_q:
        ma_long_dict[q] = df['price'].rolling(q).mean()
    
    # Prepare lists to hold all series for concatenation
    ma_short_dfs = []
    ma_long_dfs = []
    crossover_up_dfs = []
    crossover_down_dfs = []

    # Add moving averages to lists with proper column names
    for p in unique_p:
        s = ma_short_dict[p].rename(f'ma_short_{p}')
        ma_short_dfs.append(s)
    for q in unique_q:
        s = ma_long_dict[q].rename(f'ma_long_{q}')
        ma_long_dfs.append(s)
    for p in unique_p:
        for q in unique_q:
            ma_short = ma_short_dict[p]
            ma_long = ma_long_dict[q]
            crossover_up = (ma_short >= ma_long) & (ma_short.shift(1) < ma_long.shift(1))
            crossover_down = (ma_short <= ma_long) & (ma_short.shift(1) > ma_long.shift(1))
            crossover_up_dfs.append(crossover_up.rename(f'crossover_up_{p}_{q}'))
            crossover_down_dfs.append(crossover_down.rename(f'crossover_down_{p}_{q}'))

    # Concatenate all at once to df
    df = pd.concat(
        [df] + ma_short_dfs + ma_long_dfs + crossover_up_dfs + crossover_down_dfs,
        axis=1
    )

    return df

def prepare_rsi(df, param_grid):
    unique_h = sorted(set(h for h, _, _ in param_grid))
    # Calculate price differences once
    delta = df['price'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    rsi_series_list = []
    
    for h in unique_h:
        avg_gain = gain.rolling(h).mean()
        avg_loss = loss.rolling(h).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = f'rsi_{h}'
        rsi_series_list.append(rsi)
    
    df = pd.concat([df] + rsi_series_list, axis=1)
    return df

def prepare_support_resistance(df, param_grid):
    unique_j = sorted(set(j for j, _, _ in param_grid))
    
    max_list = []
    min_list = []

    for j in unique_j:
        max_series = df['price'].rolling(window=j).max().shift(1)
        max_series.name = f'max_prev_{j}'

        min_series = df['price'].rolling(window=j).min().shift(1)
        min_series.name = f'min_prev_{j}'

        max_list.append(max_series)
        min_list.append(min_series)

    df_max = pd.concat(max_list, axis=1)
    df_min = pd.concat(min_list, axis=1)

    df = pd.concat([df, df_max, df_min], axis=1)
    return df

def prepare_filter(df, param_grid):
    unique_j = sorted(set(j for j, _, _, _ in param_grid))
    features = []

    for j in unique_j:
        rolling_max = df['price'].rolling(window=j).max().shift(1).rename(f'fmax_prev_{j}')
        rolling_min = df['price'].rolling(window=j).min().shift(1).rename(f'fmin_prev_{j}')
        features.extend([rolling_max, rolling_min])

    # Combine all at once to avoid fragmentation
    df = pd.concat([df] + features, axis=1)
    return df

def prepare_channel(df, param_grid):
    unique_j = sorted(set(j for j, _, _, _ in param_grid))
    max_list = []
    min_list = []

    for j in unique_j:
        max_series = df['price'].rolling(window=j).max()
        max_series.name = f'rolling_max_{j}'

        min_series = df['price'].rolling(window=j).min()
        min_series.name = f'rolling_min_{j}'

        max_list.append(max_series)
        min_list.append(min_series)

    df_max = pd.concat(max_list, axis=1)
    df_min = pd.concat(min_list, axis=1)

    df = pd.concat([df, df_max, df_min], axis=1)
    return df

# Strategy implementations
def strategy_moving_average(df, p, q, x, d):
    df = df.copy()

    df['ma_short'] = df[f'ma_short_{p}']
    df['ma_long'] = df[f'ma_long_{q}']
    # Detect crossover
    df['crossover_up'] = df[f'crossover_up_{p}_{q}']
    df['crossover_down'] = df[f'crossover_down_{p}_{q}']
    # MA condition with band
    df['condition_long'] = df['ma_short'] > (1 + x) * df['ma_long']
    df['condition_short'] = df['ma_short'] < (1 - x) * df['ma_long']
    # Position
    df['position'] = 0
    
    # Get crossover indices
    crossover_up_indices = df.index[df['crossover_up']].tolist()
    crossover_down_indices = df.index[df['crossover_down']].tolist()
    
    for idx in crossover_up_indices:
        t = df.index.get_loc(idx)
        if t + d < len(df):
            if df['condition_long'].iloc[t + 1:t + 1 + d].all():
                df.loc[df.index[t + d], 'position'] = 1

    for idx in crossover_down_indices:
        t = df.index.get_loc(idx)
        if t + d < len(df):
            if df['condition_short'].iloc[t + 1:t + 1 + d].all():
                df.loc[df.index[t + d], 'position'] = -1
    
    return df

def strategy_rsi_cross(df, h, v, d):
    df = df.copy()

    df['rsi'] = df[f'rsi_{h}']

    # Lower and upper bound
    lower = 50 - v
    upper = 50 + v

    df['condition_buy'] = df['rsi'] > lower
    df['condition_sell'] = df['rsi'] < upper
    df['crossed_up'] = (df['rsi'].shift(1) <= lower) & (df['rsi'] > lower)
    df['crossed_down'] = (df['rsi'].shift(1) >= upper) & (df['rsi'] < upper)
    
    df['position'] = 0

    crossed_up_indices = np.where(df['crossed_up'].values)[0]
    for t in crossed_up_indices:
        if t + d < len(df) and df['condition_buy'].iloc[t + 1:t + 1 + d].all():
            df.iloc[t + d, df.columns.get_loc('position')] = 1

    crossed_down_indices = np.where(df['crossed_down'].values)[0]
    for t in crossed_down_indices:
        if t + d < len(df) and df['condition_sell'].iloc[t + 1:t + 1 + d].all():
            df.iloc[t + d, df.columns.get_loc('position')] = -1

    return df

def strategy_support_resistance(df, j, x, d):
    df = df.copy()

    # Rolling high/low
    df['max_prev'] = df[f'max_prev_{j}']
    df['min_prev'] = df[f'min_prev_{j}']

    # Band thresholds
    df['breakout_up'] = df['price'] > (1 + x) * df['max_prev']
    df['breakout_down'] = df['price'] < (1 - x) * df['min_prev']

    # Initialize position
    df['position'] = 0

    # Get crossover indices
    breakout_up_indices = df.index[df['breakout_up']].tolist()
    breakout_down_indices = df.index[df['breakout_down']].tolist()
    
    for idx in breakout_up_indices:
        t = df.index.get_loc(idx)
        if t + d < len(df):
            if df['breakout_up'].iloc[t + 1:t + 1 + d].all():
                df.loc[df.index[t + d], 'position'] = 1

    for idx in breakout_down_indices:
        t = df.index.get_loc(idx)
        if t + d < len(df):
            if df['breakout_down'].iloc[t + 1:t + 1 + d].all():
                df.loc[df.index[t + d], 'position'] = -1

    return df

def strategy_filter_rule(df, j, x, y, d):
    df = df.copy()

    # Rolling high/low
    df['max_prev'] = df[f'fmax_prev_{j}']
    df['min_prev'] = df[f'fmin_prev_{j}']

    df['position'] = 0

    # For buy:
    cond_buy = (df['price'] > (1 + x) * df['min_prev']) & (df['price'].shift(1) <= df['min_prev'].shift(1))
    # For sell:
    cond_sell = (df['price'] < (1 - y) * df['max_prev']) & (df['price'].shift(1) >= df['max_prev'].shift(1))
    
    # Vectorized rolling window: check if all d values are True via .sum()
    buy_signal = cond_buy.rolling(window=d).sum() == d
    sell_signal = cond_sell.rolling(window=d).sum() == d

    # Mark the position at the day AFTER the d consecutive days
    df.loc[buy_signal.shift(1).fillna(False), 'position'] = 1
    df.loc[sell_signal.shift(1).fillna(False), 'position'] = -1

    return df

def strategy_channel_breakout(df, j, c, x, d):
    df = df.copy()

    # Calculate c
    df['rolling_max'] = df[f'rolling_max_{j}']
    df['rolling_min'] = df[f'rolling_min_{j}']

    # Check if channel exists at time t
    df['in_channel'] = df['rolling_max'] <= (1 - c) * df['rolling_min']
    
    # Check that channel exists for d consecutive days
    channel_d_days = df['in_channel'].rolling(window=d).apply(lambda x: x.all(), raw=True) == 1

    # BUY condition for each day: price > (1 + x) * previous day rolling_max
    cond_buy = df['price'] > (1 + x) * df['rolling_max'].shift(1)
    buy_valid = cond_buy.rolling(window=d).apply(lambda x: x.all(), raw=True) == 1

    # SELL condition for each day: price < (1 - x) * previous day rolling_min
    cond_sell = df['price'] < (1 - x) * df['rolling_min'].shift(1)
    sell_valid = cond_sell.rolling(window=d).apply(lambda x: x.all(), raw=True) == 1

    # Final valid buy and sell signals must have channel for d days as well
    buy_signal = (channel_d_days & buy_valid)
    sell_signal = (channel_d_days & sell_valid)

    df['position'] = 0
    df.loc[buy_signal.shift(1).fillna(False), 'position'] = 1
    df.loc[sell_signal.shift(1).fillna(False), 'position'] = -1

    return df

# Start strategy with column name
def strategy_ma(df, p, q, x, d):
    if p >= q:
        return None
    name = f"MA_p{p}_q{q}_x{x:.4f}_d{d}"
    df = strategy_moving_average(df, p, q, x, d)
    return name, df['position']

def strategy_rsi(df, h, v, d):
    name = f"RSI_h{h}_v{v}_d{d}"
    df = strategy_rsi_cross(df, h, v, d)
    return name, df['position']

def strategy_filter(df, j, x, y, d):
    name = f"FILTER_j{j}_x{x:.4f}_y{y:.4f}_d{d}"
    df = strategy_filter_rule(df, j, x, y, d)
    return name, df['position']

def strategy_sr(df, j, x, d):
    name = f"SR_j{j}_x{x:.4f}_d{d}"
    df = strategy_support_resistance(df, j, x, d)
    return name, df['position']

def strategy_channel(df, j, c, x, d):
    name = f"CHANNEL_j{j}_c{c:.4f}_x{x:.4f}_d{d}"
    df = strategy_channel_breakout(df, j, c, x, d)
    return name, df['position']

# Parameter grids
pj_vals = [1, 2, 6, 12, 18, 24, 30, 48, 96, 144, 168]
q_vals = [2, 6, 12, 18, 24, 30, 48, 96, 144, 168, 192]
x_vals_ma = [0, 0.0005, 0.001, 0.005, 0.015]
d_vals_ma = [0, 2, 3, 4, 5]
params_ma = list(itertools.product(pj_vals, q_vals, x_vals_ma, d_vals_ma))

# RSI
h_values_rsi = [2, 6, 12, 14, 18, 24, 30, 48, 96, 168]
v_values_rsi = [10, 15, 20, 25]  
d_values_rsi = [0, 1, 2, 3, 4, 5]
params_rsi = list(itertools.product(h_values_rsi, v_values_rsi, d_values_rsi))

# FILTER
j_values_fr = [1, 2, 6, 12, 24]
x_values_fr = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
y_values_fr = [0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
d_values_fr = [0, 1, 2, 3, 4, 5]
params_fr = list(itertools.product(j_values_fr, x_values_fr, y_values_fr, d_values_fr))

# SR
j_values_sr = [2, 6, 12, 18, 24, 30, 48, 96, 168]
x_values_sr = [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20] 
d_values_sr = [0, 1, 2, 3, 4, 5]
params_sr = list(itertools.product(j_values_sr, x_values_sr, d_values_sr))

# CHANNEL
j_values_ch = [6, 12, 18, 24, 36, 72, 120, 168]
c_values_ch = [0.005, 0.01, 0.05, 0.10, 0.15]
x_values_ch = [0.0005, 0.001, 0.005, 0.01, 0.05]
d_values_ch = [0, 1]
params_ch = list(itertools.product(j_values_ch, c_values_ch, x_values_ch, d_values_ch))

STRATEGIES = [
    ('MA', strategy_ma, params_ma),
    ('RSI', strategy_rsi, params_rsi),
    ('SR', strategy_sr, params_sr),
    ('FR', strategy_filter, params_fr),
    ('CH', strategy_channel, params_ch),
]

def parallel_feature_preparation(df, params_ma, params_rsi, params_sr, params_fr, params_ch):
    tasks = [
        (prepare_moving_averages, params_ma),
        (prepare_rsi, params_rsi),
        (prepare_support_resistance, params_sr),
        (prepare_filter, params_fr),
        (prepare_channel, params_ch)
    ]

    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(func, df.copy(), params) for func, params in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result_df = future.result()
                results.append(result_df)
            except Exception as e:
                print(f"❌ Error during feature preparation: {e}")

    # Keep only new columns from each result_df and concatenate once
    new_columns = [res[[col for col in res.columns if col not in df.columns]] for res in results]
    df = pd.concat([df] + new_columns, axis=1)
    
    return df

def process_strategy(df, tag, func, grid, base, out_dir):
    try:
        results = []
        for params in grid:
            out = func(df, *params)
            if out is not None:
                name, series = out
                results.append(series.rename(name))
        if results:
            combined = pd.concat([df['price']] + results, axis=1)
            combined.to_csv(f"{out_dir}/{base}_{tag}.csv", compression='gzip')
    except Exception as e:
        print(f"❌ Error in strategy {tag} for {base}: {e}")

def process_file(file_path):
    try:
        print(f"🔍 Starting file: {file_path}")
        df = pd.read_csv(file_path)
        
        # Handle unnamed date column
        if 'Unnamed: 0' in df.columns:
            df.rename(columns={'Unnamed: 0': 'date'}, inplace=True)
        elif 'date' not in df.columns:
            raise ValueError(f"'date' column not found in {file_path}")

        # Rename close column to 'price'
        if 'close' in df.columns:
            df.rename(columns={'close': 'price'}, inplace=True)
        elif 'price' not in df.columns:
            raise ValueError(f"'price' column not found in {file_path}")
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date'], inplace=True)
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        df = parallel_feature_preparation(df, params_ma, params_rsi, params_sr, params_fr, params_ch)

        print("Preparing done")

        base = os.path.splitext(os.path.basename(file_path))[0]
        out_dir = 'Parts2'
        os.makedirs(out_dir, exist_ok=True)

        # Parallelize strategy processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = []
            for tag, func, grid in STRATEGIES:
                futures.append(
                    executor.submit(process_strategy, df, tag, func, grid, base, out_dir)
                )
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"❌ Error during strategy execution: {e}")

        print(f"✅ Processed {base}")
    except Exception as e:
         print(f"❌ Error processing {file_path}: {e}")

def main():
    files = glob.glob("/Users/jortvanderweijde/Desktop/Pythontest/Parts/*USD*.csv")
    print(f"Found {len(files)} files.")
    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(process_file, file): file for file in files}
        
        # Wrap the as_completed iterator with tqdm for progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            file = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"❌ Error in future for {file}: {e}")

if __name__ == '__main__':
    # Make multiprocessing safe on all OSes
    multiprocessing.set_start_method("spawn", force=True)
    main()
