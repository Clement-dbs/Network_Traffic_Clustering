import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_loader import get_tp_features


def preprocess_data(df, include_proto=True):
    X = get_tp_features(df, include_proto=include_proto).copy()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("encoder", encoder)
        ]), cat_cols)
    ])

    X_processed = preprocessor.fit_transform(X)

    if hasattr(X_processed, "toarray"):
        X_processed = X_processed.toarray()

    return X_processed, preprocessor
