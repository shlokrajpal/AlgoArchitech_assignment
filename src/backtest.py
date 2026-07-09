import numpy as np
import pandas as pd

from base_signal import generate_signals

STARTING_CAPITAL = 1_000_000.0

# --- Transaction friction assumption ---
# Combined penalty representing slippage + exchange charges + broker commission,
# applied once per side (entry and exit), i.e. round-trip cost = 2 * FRICTION_PER_SIDE.
FRICTION_PER_SIDE = 0.0005  # 5 bps per side


def _resolve_exit(entry_idx, direction, tp, sl, high, low, open_, close, dates):
    """
    Scan forward from entry_idx+1 for the first bar where TP or SL is touched,
    stopping at the end of the entry's session (no overnight carry, consistent
    with the regime model resetting its state each session). If both are touched
    within the same bar, resolve using an assumed intrabar path: bullish bar
    (close >= open) -> path is open -> low -> high -> close; bearish bar
    (close < open) -> path is open -> high -> low -> close.
    Returns (exit_idx, exit_price, exit_reason) or (None, None, None) if never hit
    within the same session.
    """
    n = len(close)
    entry_date = dates[entry_idx]
    for j in range(entry_idx + 1, n):
        if dates[j] != entry_date:
            break
        hit_tp = (high[j] >= tp) if direction == 1 else (low[j] <= tp)
        hit_sl = (low[j] <= sl) if direction == 1 else (high[j] >= sl)

        if hit_tp and hit_sl:
            bullish = close[j] >= open_[j]
            # bullish: low touched before high -> for a long, SL (below) hit first;
            #          for a short, TP (below) hit first
            # bearish: high touched before low -> for a long, TP (above) hit first;
            #          for a short, SL (above) hit first
            if bullish:
                first = "SL" if direction == 1 else "TP"
            else:
                first = "TP" if direction == 1 else "SL"
            exit_price = sl if first == "SL" else tp
            return j, exit_price, first
        elif hit_tp:
            return j, tp, "TP"
        elif hit_sl:
            return j, sl, "SL"

    return None, None, None


def run_backtest(df: pd.DataFrame, starting_capital: float = STARTING_CAPITAL, position_size_pct: float = 0.01) -> pd.DataFrame:
    signals_df = generate_signals(df)

    # generate_signals() only returns close/regime/signal/tp_price/sl_price, so pull
    # high/low from the original df (same index, guaranteed same row order)
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    close = signals_df["close"].values
    timestamps = signals_df.index
    dates = timestamps.date

    entry_positions = np.where(signals_df["signal"].isin([1, -1]).values)[0]

    trades = []
    capital = starting_capital
    in_position_until = -1  # index up to which we're already in a trade (skip overlapping entries)

    for i in entry_positions:
        if i <= in_position_until:
            continue

        direction = signals_df["signal"].iloc[i]
        tp = signals_df["tp_price"].iloc[i]
        sl = signals_df["sl_price"].iloc[i]
        entry_price = close[i]

        exit_idx, exit_price, reason = _resolve_exit(i, direction, tp, sl, high, low, open_, close, dates)

        if exit_idx is None:
            # neither TP nor SL hit before session end -> flatten at last bar of that session
            entry_date = dates[i]
            same_day_idx = np.where(dates == entry_date)[0]
            exit_idx = same_day_idx[-1]
            exit_price = close[exit_idx]
            reason = "EOD_FLATTEN"

        # Position sized at the specified percentage of available capital (default 1%)
        notional = capital * position_size_pct
        
        raw_return_pct = direction * (exit_price - entry_price) / entry_price
        gross_pnl = notional * raw_return_pct

        friction_cost = notional * FRICTION_PER_SIDE * 2  # entry + exit
        net_pnl = gross_pnl - friction_cost

        capital += net_pnl

        trades.append({
            "entry_time": timestamps[i],
            "exit_time": timestamps[exit_idx],
            "regime": signals_df["regime"].iloc[i],
            "signal": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "exit_reason": reason,
            "gross_pnl": gross_pnl,
            "friction_cost": friction_cost,
            "net_pnl": net_pnl,
            "capital_available": capital,
        })

        in_position_until = exit_idx

    trades_df = pd.DataFrame(trades)
    
    # Check if trades_df is empty to prevent KeyError on empty backtest
    if not trades_df.empty:
        trades_df["cumulative_pnl"] = trades_df["net_pnl"].cumsum()
    else:
        trades_df["cumulative_pnl"] = []

    return trades_df


if __name__ == "__main__":
    df = pd.read_parquet("data/train_features.parquet")
    results = run_backtest(df)
    results.to_csv("data/backtest_results.csv", index=False)
    print(results.head(20))
    print(f"\nTotal trades: {len(results)}")
    
    if not results.empty:
        print(f"Final capital: {results['capital_available'].iloc[-1]:,.2f}")
        print(f"Total net PnL: {results['net_pnl'].sum():,.2f}")
    else:
        print("No trades executed.")