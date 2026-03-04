import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from data_loader import get_tp_features


def preprocess_data(df, include_proto=True):

    X = get_tp_features(df, include_proto=include_proto).copy()

    if include_proto and "proto" in X.columns:
        proto_col = X[["proto"]]
        X_numeric = X.drop(columns=["proto"])

        try:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

        proto_encoded = encoder.fit_transform(proto_col)
        proto_feature_names = encoder.get_feature_names_out(["proto"])
        proto_df = pd.DataFrame(proto_encoded, columns=proto_feature_names, index=X.index)
        X = pd.concat([X_numeric, proto_df], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler
