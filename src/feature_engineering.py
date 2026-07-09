import numpy as np
import pandas as pd

IN_FILE = "data/preprocessed_data.parquet"
TRAIN_FILE = "data/train_features.parquet"
VAL_FILE = "data/val_features.parquet"
TEST_FILE = "data/test_features.parquet"

def split_features(df, train_frac=0.7, val_frac=0.1):
    """Chronological split to avoid lookahead bias."""
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df

def compute_features(df):
    df = df.copy()
    df['date'] = df.index.date

    # --- log_return: ln(close_t / close_t-1), first bar of session -> ln(close/open) ---
    df['log_return'] = np.log(df.groupby('date')['close'].apply(lambda s: s / s.shift(1))).reset_index(level=0, drop=True)
    first_bar_mask = df['log_return'].isna()
    df.loc[first_bar_mask, 'log_return'] = np.log(df.loc[first_bar_mask, 'close'] / df.loc[first_bar_mask, 'open'])

    # --- overnight_gap: (open_day - close_day-1) / close_day-1 ---
    daily_open = df.groupby('date')['open'].first()
    daily_close = df.groupby('date')['close'].last()
    gap = (daily_open - daily_close.shift(1)) / daily_close.shift(1)
    df['overnight_gap'] = df['date'].map(gap)

    # --- realized_vol_30m: sqrt(sum of squared log_return over last 30 bars), session-reset ---
    def realized_vol(group):
        return np.sqrt(group['log_return'].pow(2).rolling(window=30, min_periods=30).sum())
    df['realized_vol_30m'] = df.groupby('date').apply(realized_vol, include_groups=False).reset_index(level=0, drop=True)

    # --- tod_vol_zscore: rolling z-score of realized_vol_30m at the same minute-of-day, over last 30 trading days ---
    df['minute_of_day'] = df.groupby('date').cumcount()
    tod = df.pivot_table(index='date', columns='minute_of_day', values='realized_vol_30m', aggfunc='first')
    tod_mean = tod.rolling(window=30, min_periods=30).mean().shift(1)
    tod_std = tod.rolling(window=30, min_periods=30).std().shift(1)
    tod_z = (tod - tod_mean) / tod_std
    tod_z_long = tod_z.stack(future_stack=True).rename('tod_vol_zscore')
    tod_z_long.index.names = ['date', 'minute_of_day']
    _orig_index = df.index
    df = df.merge(tod_z_long, on=['date', 'minute_of_day'], how='left')
    df.index = _orig_index

    # --- garman_klass_vol: sqrt(rolling 30m mean of per-bar GK term), session-reset ---
    gk_term = 0.5 * np.log(df['high'] / df['low']) ** 2 - (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2
    df['_gk_term'] = gk_term
    def gk_vol(group):
        return np.sqrt(group['_gk_term'].rolling(window=30, min_periods=30).mean())
    df['garman_klass_vol'] = df.groupby('date').apply(gk_vol, include_groups=False).reset_index(level=0, drop=True)
    df.drop(columns=['_gk_term'], inplace=True)

    # --- bipower_variation_30m: (pi/2) * sum_{i=1..29} |log_return_t-i| * |log_return_t-i-1|, session-reset ---
    def bipower(group):
        abs_ret = group['log_return'].abs()
        prod = abs_ret * abs_ret.shift(1)
        return (np.pi / 2) * prod.rolling(window=29, min_periods=29).sum()
    df['bipower_variation_30m'] = df.groupby('date').apply(bipower, include_groups=False).reset_index(level=0, drop=True)

    # --- range_expansion_ratio: (High_t - Low_t) / ATR_30m, session-reset ---
    def range_expansion(group):
        prev_close = group['close'].shift(1)
        tr = pd.concat([
            group['high'] - group['low'],
            (group['high'] - prev_close).abs(),
            (group['low'] - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_30m = tr.rolling(window=30, min_periods=30).mean()
        return (group['high'] - group['low']) / atr_30m
    df['range_expansion_ratio'] = df.groupby('date').apply(range_expansion, include_groups=False).reset_index(level=0, drop=True)

    # --- dist_from_ma_15m: close_t - SMA_15m(close), session-reset ---
    def dist_from_ma(group):
        ma_15m = group['close'].rolling(window=15, min_periods=15).mean()
        return group['close'] - ma_15m
    df['dist_from_ma_15m'] = df.groupby('date').apply(dist_from_ma, include_groups=False).reset_index(level=0, drop=True)

    # --- rolling_acf_lag1_30m: lag-1 autocorrelation of log_return over last 30 bars, session-reset ---
    def acf_lag1(group):
        return group['log_return'].rolling(window=30, min_periods=30).apply(
            lambda x: pd.Series(x).autocorr(lag=1), raw=False
        )
    df['rolling_acf_lag1_30m'] = df.groupby('date').apply(acf_lag1, include_groups=False).reset_index(level=0, drop=True)

    # --- minutes_from_open: 0 to 375, divided by 375 ---
    df['minutes_from_open'] = df['minute_of_day'] / 375.0

    # --- day_of_week: 1=Monday ... 7=Sunday, divided by 7 ---
    df['day_of_week'] = (pd.DatetimeIndex(df['date']).dayofweek + 1) / 7.0

    warmup_mask = df['minute_of_day'] < 30
    warmup_cols = [
        'realized_vol_30m', 'tod_vol_zscore', 'garman_klass_vol',
        'bipower_variation_30m', 'range_expansion_ratio', 'dist_from_ma_15m',
        'rolling_acf_lag1_30m'
    ]
    df.loc[warmup_mask, warmup_cols] = np.nan

    df.drop(columns=['date', 'minute_of_day'], inplace=True)
    return df

def run_feature_engineering():
    print(f"Loading {IN_FILE} ...")
    df = pd.read_parquet(IN_FILE)

    df = compute_features(df)

    train_df, val_df, test_df = split_features(df)

    train_df.to_parquet(TRAIN_FILE, engine='pyarrow')
    val_df.to_parquet(VAL_FILE, engine='pyarrow')
    test_df.to_parquet(TEST_FILE, engine='pyarrow')

    print(f"Train features saved to {TRAIN_FILE} ({len(train_df)} rows)")
    print(f"Val features saved to {VAL_FILE} ({len(val_df)} rows)")
    print(f"Test features saved to {TEST_FILE} ({len(test_df)} rows)")

    print("\n--- Final Parquet Schema ---")
    pq_df = pd.read_parquet(TRAIN_FILE)
    print(pq_df.dtypes)

    print("\n--- Head Sample ---")
    print(pq_df.head())


if __name__ == "__main__":
    run_feature_engineering()