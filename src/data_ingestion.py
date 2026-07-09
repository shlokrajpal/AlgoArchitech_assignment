import os
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_yfinance_data(ticker="^NSEI", days_back=29):
    """Fetches YF data in 7-day chunks"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    all_chunks = []
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=7), end_date)
        start_str = current_start.strftime("%Y-%m-%d")
        end_str = current_end.strftime("%Y-%m-%d")
        
        df_chunk = yf.download(tickers=ticker, start=start_str, end=end_str, interval="1m", progress=False)
        if not df_chunk.empty:
            # Flatten multi-index columns
            if isinstance(df_chunk.columns, pd.MultiIndex):
                df_chunk.columns = df_chunk.columns.droplevel(1)
            all_chunks.append(df_chunk)
            
        current_start = current_end

    if all_chunks:
        df = pd.concat(all_chunks)
        df = df[~df.index.duplicated(keep="first")]
        # Convert to IST.
        df.index = df.index.tz_convert('Asia/Kolkata')
        df.index.name = 'datetime'
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].rename(columns=str.lower)
        return df
    return pd.DataFrame()

def fetch_moneycontrol_data(from_date, to_date):
    """Fetches Moneycontrol 1-min data using UDF API."""
    from_ts = int(time.mktime(time.strptime(from_date, "%Y-%m-%d %H:%M:%S")))
    to_ts = int(time.mktime(time.strptime(to_date, "%Y-%m-%d %H:%M:%S")))
    url = "https://priceapi.moneycontrol.com/techCharts/history"
    params = {"symbol": "9", "resolution": "1", "from": from_ts, "to": to_ts}
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    
    response = requests.get(url, params=params, headers=headers, timeout=15)
    if response.status_code == 200:
        data = response.json()
        if data.get("s") == "ok":
            df = pd.DataFrame({
                "open": data["o"], "high": data["h"], 
                "low": data["l"], "close": data["c"], "volume": data["v"]
            }, index=pd.to_datetime(data["t"], unit="s", utc=True).tz_convert('Asia/Kolkata'))
            df.index.name = 'datetime'
            return df
    return pd.DataFrame()

def load_kaggle_data(filepath="data\kaggle_NIFTY50_data.csv"):
    """Loads Kaggle CSV data."""
    if not os.path.exists(filepath):
        print(f"Kaggle file {filepath} not found. Skipping.")
        return pd.DataFrame()
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date']) # Assuming 'date' is the timestamp column
    df.set_index('date', inplace=True)
    df.index.name = 'datetime'
    # Localize naive timestamps to IST
    df.index = df.index.tz_localize('Asia/Kolkata')
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df

def reconcile_and_merge(df_base, df_new, source_name, tolerance=0.001):
    """Merges new data into base data, checking for overlaps and disagreements."""
    if df_new.empty: return df_base
    if df_base.empty: return df_new
    
    overlap_idx = df_base.index.intersection(df_new.index)
    if not overlap_idx.empty:
        # Check percentage difference on Close prices
        diff = (df_base.loc[overlap_idx, 'close'] - df_new.loc[overlap_idx, 'close']).abs() / df_base.loc[overlap_idx, 'close']
        avg_diff = diff.mean()
        if avg_diff > tolerance:
            print(f"WARNING: High overlap discrepancy with {source_name}. Avg Diff: {avg_diff:.4%}")
        
        # Resolve Open/High/Low disagreements using next bar's open (vectorized approach)
        major_disagreements = diff[diff > 0.005]
        if not major_disagreements.empty:
            print(f"Logged {len(major_disagreements)} unresolved major disagreements on Close prices.")
            
    # Combine, preferring base data for duplicates
    combined = pd.concat([df_base, df_new[~df_new.index.isin(df_base.index)]]).sort_index()
    return combined

def run_pipeline():
    df_kaggle = load_kaggle_data()
    df_mc = fetch_moneycontrol_data("2026-05-01 00:00:00", "2026-07-01 00:00:00")
    df_yf = fetch_yfinance_data(days_back=29)
    
    df = reconcile_and_merge(df_kaggle, df_mc, "Moneycontrol")
    df = reconcile_and_merge(df, df_yf, "YahooFinance")
    
    if df.empty:
        raise ValueError("No data loaded. Exiting.")

    print(f"Total raw bars unified: {len(df)}")

    df.drop(columns=['volume'], inplace=True, errors='ignore')
    
    initial_len = len(df)
    
    # Drop negative/zero prices and NaNs
    df = df.dropna()
    df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
    
    # Logic bounds checks: H >= max(O,C,L) and L <= min(O,C,H)
    valid_high = df['high'] >= df[['open', 'close', 'low']].max(axis=1)
    valid_low = df['low'] <= df[['open', 'close', 'high']].min(axis=1)
    df = df[valid_high & valid_low]
    
    # Remove duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    
    dropped = initial_len - len(df)
    print(f"Dropped {dropped} bars failing structural/deterministic integrity.")

    df['date'] = df.index.date
    daily_counts = df.groupby('date').size()
    
    # A standard full session (9:15 to 15:30) has 375 minutes. 
    STANDARD_MIN_BARS = 370
    
    # Build session calendar
    calendar = pd.DataFrame(daily_counts, columns=['bar_count'])
    calendar['session_type'] = np.where(calendar['bar_count'] >= STANDARD_MIN_BARS, 'Full_Session', 'Short_Session')
    
    df = df.merge(calendar[['session_type']], left_on='date', right_index=True)
    
    # Group short sessions to remove them (Muhurat/DR)
    short_sessions = df[df['session_type'] == 'Short_Session']['date'].unique()
    df = df[df['session_type'] == 'Full_Session'].copy()
    
    _log_ret = np.log(df.groupby('date')['close'].apply(lambda s: s / s.shift(1))).reset_index(level=0, drop=True)

    first_bar_mask = _log_ret.isna()
    _log_ret.loc[first_bar_mask] = (df.loc[first_bar_mask, 'close'] - df.loc[first_bar_mask, 'open']) / df.loc[first_bar_mask, 'open']
    df['_tmp_returns'] = _log_ret
    
    # Rolling Hampel Filter resetting at session boundaries
    def hampel_filter(group, window_size=30, n_sigmas=3):
        rolling_median = group['_tmp_returns'].rolling(window=window_size, min_periods=10).median()
        
        # MAD approximation
        rolling_mad = group['_tmp_returns'].rolling(window=window_size, min_periods=10).apply(
            lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))), raw=True
        )
        
        dynamic_threshold = n_sigmas * 1.4826 * rolling_mad
        
        # DEFINE A HARD FLOOR (0.25% return in 1 minute to be considered a true bad tick)
        min_absolute_threshold = 0.0025 
        
        # Use whichever is larger: the dynamic threshold or the hard floor
        final_threshold = np.maximum(dynamic_threshold, min_absolute_threshold)
        
        outlier_flag = np.abs(group['_tmp_returns'] - rolling_median) > final_threshold
        return outlier_flag

    df['is_anomaly'] = df.groupby('date').apply(hampel_filter, include_groups=False).reset_index(level=0, drop=True)
    df.drop(columns=['_tmp_returns'], inplace=True)

    anomalies_count = df['is_anomaly'].sum()
    print(f"Flagged {anomalies_count} statistical anomalies (bad ticks).")
    
    if anomalies_count > 0:
        anomalies_df = df[df['is_anomaly'] == True].copy()
        anomalies_df.drop(columns=['date', 'session_type', 'is_anomaly'], inplace=True, errors='ignore')
        anomalies_out_file = "anomalies_dataset.csv"
        anomalies_df.to_csv(anomalies_out_file)
        print(f"Saved bad ticks/anomalies separately to {anomalies_out_file}")

    print("QA SUMMARY")
    print(f"Total Rows Processed:      {len(df)}")
    print(f"Integrity Bars Dropped:    {dropped}")
    print(f"Anomalies Flagged:         {anomalies_count} ({anomalies_count/len(df):.4%})")
    print(f"Short Sessions Excluded:   {len(short_sessions)} dates")
    print(f"Sample Excluded Dates:     {short_sessions[:5]}")

    # Clean up auxiliary columns before export
    df.drop(columns=['date', 'session_type', 'is_anomaly'], inplace=True, errors='ignore')
    
    # Save to Parquet
    out_file = "data/preprocessed_data.parquet"
    df.to_parquet(out_file, engine='pyarrow')
    print(f"\nData successfully saved to {out_file}")
    
    pq_df = pd.read_parquet(out_file)
    print(pq_df.dtypes)
    
    print("\n--- Head Sample ---")
    print(pq_df.head())

if __name__ == "__main__":
    run_pipeline()