import os
import pandas as pd

RAW_V0 = "data/raw/v0"
PROCESSED_V1 = "data/processed/dataset_versions/v1"

def load_and_process(csv_path, stock_symbol):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "datetime": "timestamp",
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "volume": "volume"
    })
    df["stock_symbol"] = stock_symbol
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df[["timestamp", "stock_symbol", "open_price", "high_price", "low_price", "close_price", "volume"]]
    return df

def main():
    os.makedirs(PROCESSED_V1, exist_ok=True)
    combined = []

    for file in os.listdir(RAW_V0):
        if file.endswith(".csv"):
            stock = file.split("__")[0]
            df = load_and_process(os.path.join(RAW_V0, file), stock)
            combined.append(df)

    final_df = pd.concat(combined)
    sample_df = final_df.head(100)

    final_df.to_csv(f"{PROCESSED_V1}/v0_full.csv", index=False)
    sample_df.to_csv(f"{PROCESSED_V1}/v0_sample_100.csv", index=False)
    print("✅ Data processed and saved successfully.")

if __name__ == "__main__":
    main()
