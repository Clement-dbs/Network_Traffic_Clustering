import pandas as pd
from pathlib import Path


TP_NUMERIC_FEATURES = [
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
]


def load_data(split="testing", data_dir=None):
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[2] / "data"
    else:
        data_dir = Path(data_dir)

    test_file = data_dir / "UNSW_NB15_testing-set.csv"
    train_file = data_dir / "UNSW_NB15_training-set.csv"

    split = split.lower()

    if split == "testing":
        return pd.read_csv(test_file)
    if split == "training":
        return pd.read_csv(train_file)
    if split == "both":
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        return pd.concat([train_df, test_df], ignore_index=True)

    raise ValueError("split doit etre: 'testing', 'training' ou 'both'")


def get_tp_features(df, include_proto=True):
    cols = TP_NUMERIC_FEATURES.copy()
    if include_proto:
        cols.append("proto")
    return df[cols].copy()
