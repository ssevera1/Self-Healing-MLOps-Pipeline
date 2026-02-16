from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Entity: each row is tied to a user
user = Entity(
    name="user_id",
    join_keys=["user_id"],
    description="Unique identifier for a user",
)

# Offline source backed by a local parquet file
user_transactions_source = FileSource(
    path="data/user_transactions.parquet",
    timestamp_field="event_timestamp",
)

# Feature view exposing transaction-based features
user_transaction_features = FeatureView(
    name="user_transaction_features",
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name="user_transaction_count", dtype=Int64),
        Field(name="user_transaction_amount_avg", dtype=Float32),
        Field(name="user_transaction_amount_max", dtype=Float32),
    ],
    source=user_transactions_source,
    online=True,
)
