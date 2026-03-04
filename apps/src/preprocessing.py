import pandas as pd
from sklearn.preprocessing import StandardScaler
from data_loader import get_tp_features


def preprocess_data(df, include_proto=True):

    X = get_tp_features(df, include_proto=include_proto).copy()

    if include_proto and "proto" in X.columns:
        X = pd.get_dummies(X, columns=["proto"], drop_first=False)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
