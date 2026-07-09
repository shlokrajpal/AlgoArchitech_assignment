import os
import datetime
from collections import deque

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm

INPUT_FILE = "data/train_features.parquet"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "regime_pipeline.pkl")
ROLLING_WINDOW = 30 

def train_and_save_model(input_path: str = INPUT_FILE, model_path: str = MODEL_PATH, n_components: int = 4):

    print(f"Loading historical data from {input_path} for training...")
    df = pd.read_parquet(input_path)

    clean_series = df["realized_vol_30m"].dropna()
    log_vol = np.log(clean_series + 1e-8)
    X = log_vol.values.reshape(-1, 1)

    print(f"Training GaussianHMM with {n_components} components...")
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=100, random_state=42)
    model.fit(X)

    learned_means = model.means_.flatten()
    ranked_states = np.argsort(learned_means)
    ranked_means = learned_means[ranked_states]
    gaps = np.diff(ranked_means)

    for rank, (state_idx, mean_val) in enumerate(zip(ranked_states, ranked_means)):
        dwell_bars = 1.0 / max(1e-6, (1.0 - model.transmat_[state_idx, state_idx]))
        print(f"  rank {rank}: state {state_idx}, mean_log_vol={mean_val:.4f}, "
              f"implied_dwell~{dwell_bars:.1f} bars")
    print(f"  gaps between consecutive ranked means: {[round(g, 4) for g in gaps]}")

    # Create the meta-regime mapping:
    # 2 lowest vol states -> 0 (Calm / Mean Reverting)
    # 2 highest vol states -> 1 (Volatile / Momentum)
    meta_mapping = {
        int(ranked_states[0]): 0,
        int(ranked_states[1]): 0,
        int(ranked_states[2]): 1,
        int(ranked_states[3]): 1,
    }
    
    dates = pd.to_datetime(df.loc[clean_series.index].index).date
    temp_pipeline = {"model": model, "meta_mapping": meta_mapping}
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump(temp_pipeline, model_path)  
    predictor = RegimePredictor(model_path=model_path, window=ROLLING_WINDOW)
    meta_states = np.array([
        predictor.update(d, v) for d, v in zip(dates, clean_series.values)
    ])
    valid = ~np.isnan(meta_states)
    n_transitions = int(np.sum(meta_states[valid][1:] != meta_states[valid][:-1]))
    print(f"  Meta-regime transitions on training replay: {n_transitions} ")

    pipeline = {
        "model": model,
        "meta_mapping": meta_mapping,
        "state_mean_log_vol": {int(s): float(m) for s, m in zip(range(n_components), learned_means)},
        "ranked_state_order": [int(s) for s in ranked_states],
        "trained_at": datetime.datetime.now().isoformat(),
        "training_rows": int(len(clean_series)),
    }
    joblib.dump(pipeline, model_path)
    print(f"\nModel and dynamic mapping successfully saved to {model_path}")

    return pipeline


class RegimePredictor:

    def __init__(self, model_path: str = MODEL_PATH, window: int = ROLLING_WINDOW):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pipeline not found at {model_path}. Please run train_and_save_model() first.")
        pipeline = joblib.load(model_path)
        self.model = pipeline["model"]
        self.meta_mapping = pipeline["meta_mapping"]
        self.window = window
        self.reset_session()

    def reset_session(self):
    
        self._buffer = deque(maxlen=self.window)
        self._current_date = None

    def update(self, timestamp, realized_vol_30m_val: float) -> float:
  
        bar_date = pd.Timestamp(timestamp).date()
        if self._current_date is None:
            self._current_date = bar_date
        elif bar_date != self._current_date:
            self.reset_session()
            self._current_date = bar_date

        if pd.isna(realized_vol_30m_val):
            return np.nan

        log_vol = np.log(realized_vol_30m_val + 1e-8)
        self._buffer.append(log_vol)

        X_buffer = np.array(self._buffer, dtype=float).reshape(-1, 1)
        decoded_states = self.model.predict(X_buffer)  # Viterbi over past+current bars only
        hmm_state = int(decoded_states[-1])

        return float(self.meta_mapping[hmm_state])


def predict_regime(realized_vol_30m_val: float, timestamp, predictor: "RegimePredictor" = None,
                    model_path: str = MODEL_PATH) -> float:

    if predictor is None:
        predictor = RegimePredictor(model_path=model_path)
    return predictor.update(timestamp, realized_vol_30m_val)


def predict_regime_batch(df: pd.DataFrame, model_path: str = MODEL_PATH, window: int = ROLLING_WINDOW,
                          vol_col: str = "realized_vol_30m") -> np.ndarray:

    predictor = RegimePredictor(model_path=model_path, window=window)
    dates = df.index.date
    vol_vals = df[vol_col].values

    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        out[i] = predictor.update(dates[i], vol_vals[i])
    return out


if __name__ == "__main__":
    train_and_save_model()