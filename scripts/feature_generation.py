import os
import pandas as pd

SRC = "data/processed/dataset_versions/v1/v0_sample_100.csv"  # produced by data_processing.py
OUT_DIR = "data/processed/dataset_versions/v1"
OUT_FEATS = f"{OUT_DIR}/features_v0_sample_100.parquet"

def _pick(df, primary, alt):
    """Return the column name that exists in df from (primary, alt)."""
    return primary if primary in df.columns else (alt if alt in df.columns else None)

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Map time & ohlcv across raw/processed variants
    ts_col    = _pick(df, "timestamp", "datetime")
    open_col  = _pick(df, "open", "open_price")
    high_col  = _pick(df, "high", "high_price")
    low_col   = _pick(df, "low", "low_price")
    close_col = _pick(df, "close", "close_price")
    vol_col   = "volume" if "volume" in df.columns else None

    missing = [c for c in [ts_col, open_col, high_col, low_col, close_col, vol_col] if c is None]
    if missing:
        raise KeyError(f"Missing required columns (resolved names): {missing}")

    # Ensure timestamp dtype & stock_symbol presence
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.dropna(subset=[ts_col]).copy()
    if "stock_symbol" not in df.columns:
        df["stock_symbol"] = "UNKNOWN"

    df = df.sort_values(["stock_symbol", ts_col]).reset_index(drop=True)

    # === PS features ===
    # rolling_avg_10: moving average of close over last 10 rows (t-10..t)
    df["rolling_avg_10"] = df.groupby("stock_symbol")[close_col].transform(
        lambda x: x.rolling(10, min_periods=1).mean()
    )
    # volume_sum_10: total volume over last 10 rows (t-10..t)
    df["volume_sum_10"] = df.groupby("stock_symbol")[vol_col].transform(
        lambda x: x.rolling(10, min_periods=1).sum()
    )

    # Target: 1 if close[t+5] > close[t] else 0
    future_close = df.groupby("stock_symbol")[close_col].shift(-5)
    df["target"] = (future_close > df[close_col]).astype(int)

    # Standardized output columns for Feast & training
    out = pd.DataFrame({
        "timestamp": df[ts_col],
        "stock_symbol": df["stock_symbol"].astype(str),
        "open_price": df[open_col].astype(float),
        "high_price": df[high_col].astype(float),
        "low_price": df[low_col].astype(float),
        "close_price": df[close_col].astype(float),
        "volume": df[vol_col].astype(float),
        "rolling_avg_10": df["rolling_avg_10"].astype(float),
        "volume_sum_10": df["volume_sum_10"].astype(float),
        "target": df["target"].astype(int),
    })

    # Drop rows where target would be NaN due to shift(-5)
    out = out.dropna().reset_index(drop=True)
    return out

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SRC)
    feats = build_features(df)
    feats.to_parquet(OUT_FEATS, index=False)
    print(f"✅ Wrote features parquet -> {OUT_FEATS} | rows: {len(feats)}")

if __name__ == "__main__":
    main()

