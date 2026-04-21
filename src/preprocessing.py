import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    X = df.drop("label", axis=1)
    y = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler, le