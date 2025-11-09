from pathlib import Path
from feast import Entity, FileSource, FeatureView, Field
from feast.types import Float32, Int64
from feast.value_type import ValueType  # <- add this

ROOT = Path(__file__).resolve().parent.parent
features_path = str(ROOT / "data/processed/dataset_versions/v1/features_v0_sample_100.parquet")

stock_entity = Entity(
    name="stock_symbol",
    join_keys=["stock_symbol"],
    value_type=ValueType.STRING,  # <- use enum here
)

source = FileSource(
    path=features_path,
    timestamp_field="timestamp",
)

minute_features_view = FeatureView(
    name="minute_features",
    entities=[stock_entity],
    ttl=None,
    schema=[
        Field(name="open_price", dtype=Float32),
        Field(name="high_price", dtype=Float32),
        Field(name="low_price", dtype=Float32),
        Field(name="close_price", dtype=Float32),
        Field(name="volume", dtype=Int64),
        Field(name="ret_1m", dtype=Float32),
        Field(name="ma_5", dtype=Float32),
        Field(name="ma_15", dtype=Float32),
        Field(name="vol_ma_5", dtype=Float32),
    ],
    source=source,
    online=True,
)
