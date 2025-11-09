from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path="feature_repo")
entity_df = pd.DataFrame({"stock_symbol": ["AARTIIND","ABCAPITAL"]})
training_df = store.get_historical_features(
    entity_df=entity_df.assign(timestamp=pd.Timestamp("2020-01-01")),
    features=[
        "minute_features:open_price",
        "minute_features:close_price",
        "minute_features:ret_1m",
        "minute_features:ma_5",
    ],
).to_df()

print("✅ Feast offline retrieval shape:", training_df.shape)
print(training_df.head())
