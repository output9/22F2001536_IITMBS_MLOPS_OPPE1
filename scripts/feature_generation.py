import os
import pandas as pd

SRC = "data/processed/dataset_versions/v1/v0_sample_100.csv"
OUT_DIR = "data/processed/dataset_versions/v1"
OUT_FEATS = f"{OUT_DIR}/features_v0_sample_100.parquet"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["stock_symbol", "timestamp"])
    # Basic indicators
    df["ret_1m"] = df.groupby("stock_symbol")["close_price"].pct_change()
    df["ma_5"] = df.groupby("stock_symbol")["close_price"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["ma_15"] = df.groupby("stock_symbol")["close_price"].transform(lambda x: x.rolling(15, min_periods=1).mean())
    df["vol_ma_5"] = df.groupby("stock_symbol")["volume"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    # Drop leading NaNs introduced by pct_change if any
    df = df.dropna(subset=["ret_1m"]).reset_index(drop=True)
    return df

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(SRC, parse_dates=["timestamp"])
    feats = build_features(df)
    # Ensure Feast-friendly dtypes
    feats["stock_symbol"] = feats["stock_symbol"].astype(str)
    feats.to_parquet(OUT_FEATS, index=False)
    print(f"✅ Wrote features parquet -> {OUT_FEATS}")

if __name__ == "__main__":
    main()
