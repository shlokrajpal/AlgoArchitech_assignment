import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS_CSV = "data/backtest_results.csv"
REPORT_IMG = "output/performance_report.png"
REPORT_TXT = "output/performance_summary.txt"


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Drawdown as a fraction of the running peak: 0 at new highs, negative below."""
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown_stats(equity: pd.Series, trade_dates: pd.Series) -> dict:
    """
    Depth: worst peak-to-trough decline (%).
    Duration: longest stretch from a peak to full recovery back to that peak,
    measured in trades and in distinct trading days (market-closed gaps excluded).
    """
    dd = compute_drawdown(equity)
    max_depth = dd.min()
    trough_idx = dd.idxmin()

    # peak is the last index at/before trough where equity == running_max at that point
    running_max = equity.cummax()
    peak_val = running_max.loc[trough_idx]
    peak_idx = equity[equity == peak_val].index[equity[equity == peak_val].index <= trough_idx][0]

    # recovery: first index after trough where equity >= peak_val again
    recovery_candidates = equity[(equity.index > trough_idx) & (equity >= peak_val)]
    recovery_idx = recovery_candidates.index[0] if len(recovery_candidates) > 0 else None

    duration_trades = (recovery_idx - peak_idx) if recovery_idx is not None else (equity.index[-1] - peak_idx)
    recovered = recovery_idx is not None

    peak_date = trade_dates.loc[peak_idx]
    trough_date = trade_dates.loc[trough_idx]
    recovery_date = trade_dates.loc[recovery_idx] if recovered else None
    end_date_for_duration = recovery_date if recovered else trade_dates.iloc[-1]
    duration_days = len(trade_dates[(trade_dates >= peak_date) & (trade_dates <= end_date_for_duration)].unique()) - 1

    return {
        "max_drawdown_pct": max_depth * 100,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "recovery_idx": recovery_idx,
        "recovered": recovered,
        "duration_trades": int(duration_trades),
        "duration_trading_days": int(duration_days),
        "peak_date": peak_date,
        "trough_date": trough_date,
        "recovery_date": recovery_date,
    }


def summary_stats(df: pd.DataFrame, equity: pd.Series, starting_capital: float) -> dict:
    wins = df[df["net_pnl"] > 0]["net_pnl"]
    losses = df[df["net_pnl"] < 0]["net_pnl"]
    gross_profit = wins.sum()
    gross_loss = losses.sum()

    return {
        "total_trades": len(df),
        "win_rate_pct": 100 * len(wins) / len(df) if len(df) > 0 else np.nan,
        "avg_win": wins.mean() if len(wins) > 0 else 0.0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0.0,
        "profit_factor": (gross_profit / abs(gross_loss)) if gross_loss != 0 else np.nan,
        "total_net_pnl": df["net_pnl"].sum(),
        "total_return_pct": 100 * (equity.iloc[-1] / starting_capital - 1),
        "final_capital": equity.iloc[-1],
        "total_friction_paid": df["friction_cost"].sum(),
    }


def generate_report(csv_path: str = RESULTS_CSV, starting_capital: float = 1_000_000.0):
    df = pd.read_csv(csv_path, parse_dates=["entry_time", "exit_time"])
    df = df.sort_values("exit_time").reset_index(drop=True)

    equity = starting_capital + df["net_pnl"].cumsum()
    equity.index = df.index
    trade_dates = df["exit_time"].dt.date

    dd_curve = compute_drawdown(equity)
    dd_info = max_drawdown_stats(equity, trade_dates)
    stats = summary_stats(df, equity, starting_capital)

    # --- Plot: equity curve + rolling drawdown ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(equity.index, equity.values, color="steelblue", linewidth=1.2, label="Equity")
    ax1.axhline(starting_capital, color="gray", linestyle="--", linewidth=0.8, label="Starting capital")
    ax1.scatter([dd_info["peak_idx"]], [equity.loc[dd_info["peak_idx"]]], color="green", zorder=5, label="MDD peak")
    ax1.scatter([dd_info["trough_idx"]], [equity.loc[dd_info["trough_idx"]]], color="red", zorder=5, label="MDD trough")
    ax1.set_ylabel("Capital")
    ax1.set_title("Cumulative Equity Curve")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.fill_between(dd_curve.index, dd_curve.values * 100, 0, color="firebrick", alpha=0.4)
    ax2.plot(dd_curve.index, dd_curve.values * 100, color="firebrick", linewidth=1.0)
    ax2.axhline(dd_info["max_drawdown_pct"], color="black", linestyle=":", linewidth=0.8,
                label=f"Max DD: {dd_info['max_drawdown_pct']:.2f}%")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trade #")
    ax2.set_title("Rolling Drawdown")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(REPORT_IMG, dpi=150)
    plt.close()

    # --- Text summary ---
    lines = []
    lines.append("=== PERFORMANCE REPORT ===\n")
    lines.append(f"Total trades: {stats['total_trades']}")
    lines.append(f"Win rate: {stats['win_rate_pct']:.2f}%")
    lines.append(f"Avg win: {stats['avg_win']:,.2f}   Avg loss: {stats['avg_loss']:,.2f}")
    lines.append(f"Profit factor: {stats['profit_factor']:.3f}")
    lines.append(f"Total net PnL: {stats['total_net_pnl']:,.2f}")
    lines.append(f"Total return: {stats['total_return_pct']:.2f}%")
    lines.append(f"Final capital: {stats['final_capital']:,.2f}")
    lines.append(f"Total friction paid: {stats['total_friction_paid']:,.2f}\n")

    lines.append("--- Max Drawdown ---")
    lines.append(f"Depth: {dd_info['max_drawdown_pct']:.2f}%")
    lines.append(f"Peak date: {dd_info['peak_date']}  ->  Trough date: {dd_info['trough_date']}")

    lines.append(f"Duration: {dd_info['duration_trades']} trades / {dd_info['duration_trading_days']} trading days")

    summary_text = "\n".join(lines)
    with open(REPORT_TXT, "w") as f:
        f.write(summary_text)

    print(summary_text)
    print(f"\nChart saved to {REPORT_IMG}")
    print(f"Summary saved to {REPORT_TXT}")

    return stats, dd_info


if __name__ == "__main__":
    generate_report()