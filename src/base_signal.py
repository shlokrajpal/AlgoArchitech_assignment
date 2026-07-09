import numpy as np
import pandas as pd

from regime_detector import RegimePredictor, MODEL_PATH

# TP/SL distance = TP_SL_VOL_MULT * realized_vol_30m (same units as price, since
# realized_vol_30m is a return-scale quantity -> multiply by close to get price distance)
TP_SL_VOL_MULT = 1.0


def generate_signals(df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """
    Regime 0 (calm / mean-reverting): fade distance from the 15m MA.
    Regime 1 (volatile / momentum): follow distance from the 15m MA.
    
    TP/SL are fixed 1:1, sized off realized_vol_30m (causal, no lookahead).
    """
    df = df.copy()

    predictor = RegimePredictor(model_path=model_path)
    dates = df.index.date
    vol_vals = df["realized_vol_30m"].values
    regimes = np.array([predictor.update(d, v) for d, v in zip(dates, vol_vals)])
    df["regime"] = regimes

    direction = pd.Series(0, index=df.index, dtype=float)
    is_mean_revert = df["regime"] == 0
    is_momentum = df["regime"] == 1

    direction[is_mean_revert] = -np.sign(df.loc[is_mean_revert, "dist_from_ma_15m"])
    direction[is_momentum] = np.sign(df.loc[is_momentum, "dist_from_ma_15m"])

    valid = df["dist_from_ma_15m"].notna() & df["regime"].notna()
    direction[~valid] = np.nan
    df["signal"] = direction

    tp_sl_dist = TP_SL_VOL_MULT * df["realized_vol_30m"] * df["close"]
    df["tp_price"] = np.where(
        df["signal"] == 1, df["close"] + tp_sl_dist,
        np.where(df["signal"] == -1, df["close"] - tp_sl_dist, np.nan)
    )
    df["sl_price"] = np.where(
        df["signal"] == 1, df["close"] - tp_sl_dist,
        np.where(df["signal"] == -1, df["close"] + tp_sl_dist, np.nan)
    )

    return df[["close", "regime", "signal", "tp_price", "sl_price"]]


if __name__ == "__main__":
    df = pd.read_parquet("data/train_features.parquet")
    signals = generate_signals(df)
    print(signals.dropna(subset=["signal"]).head(20))