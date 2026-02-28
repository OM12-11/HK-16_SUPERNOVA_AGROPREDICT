import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_FILE = "master_agri_decision_dataset.csv"
MODEL_FILE = "spoilage_model.pkl"

def risk_level_from_loss(x):
    if x < 10:
        return "Low"
    if x < 25:
        return "Medium"
    return "High"

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found at: {os.path.abspath(DATA_FILE)}")
        sys.exit(1)

    df = pd.read_csv(DATA_FILE)

    required_cols = [
        "Storage_Type",
        "Storage_Duration_days",
        "Temperature_C",
        "Humidity_percent_x",
        "Transport_Time_hours",
        "Daily_Temperature_C",
        "Humidity_percent_y",
        "Rainfall_mm",
        "Wind_Speed_kmh",
        "Soil_Moisture_percent",
        "Extreme_Weather_Event",
        "Loss_Percentage",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print("Missing required columns:", ", ".join(missing))
        sys.exit(1)

    df = df[required_cols].copy()
    df = df.drop_duplicates()

    num_cols = [
        "Storage_Duration_days",
        "Temperature_C",
        "Humidity_percent_x",
        "Transport_Time_hours",
        "Daily_Temperature_C",
        "Humidity_percent_y",
        "Rainfall_mm",
        "Wind_Speed_kmh",
        "Soil_Moisture_percent",
    ]

    for c in num_cols + ["Loss_Percentage"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Storage_Type"] = df["Storage_Type"].astype("string")
    df["Extreme_Weather_Event"] = df["Extreme_Weather_Event"].astype("string")

    df["Storage_Stress_Index"] = df["Temperature_C"] * df["Humidity_percent_x"]
    df["Climate_Risk_Index"] = df["Rainfall_mm"] + df["Wind_Speed_kmh"] + df["Soil_Moisture_percent"]

    feature_num = num_cols + ["Storage_Stress_Index", "Climate_Risk_Index"]
    feature_cat = ["Storage_Type", "Extreme_Weather_Event"]
    target_col = "Loss_Percentage"

    X = df[feature_num + feature_cat].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, feature_num),
            ("cat", categorical_transformer, feature_cat),
        ]
    )

    model = RandomForestRegressor(n_estimators=200, random_state=42)

    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("Model Performance")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")

    preprocess = pipe.named_steps["preprocess"]
    feature_names = []
    try:
        feature_names = list(preprocess.get_feature_names_out())
    except Exception:
        num_names = feature_num
        cat_encoder = preprocess.named_transformers_["cat"].named_steps["encoder"]
        cat_input = feature_cat
        cat_names = list(cat_encoder.get_feature_names_out(cat_input))
        feature_names = num_names + cat_names

    importances = pipe.named_steps["model"].feature_importances_
    fi = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)

    print("Feature Importance (sorted)")
    for name, val in fi:
        print(f"{name}: {val:.6f}")

    pred_df = pd.DataFrame(
        {
            "Actual_Loss_Percentage": y_test.reset_index(drop=True),
            "Predicted_Loss_Percentage": pd.Series(y_pred).round(4),
        }
    )
    pred_df["Predicted_Risk_Level"] = pred_df["Predicted_Loss_Percentage"].apply(risk_level_from_loss)
    print("Sample Predictions (10)")
    print(pred_df.head(10).to_string(index=False))

    joblib.dump(pipe, MODEL_FILE)
    print(f"Model saved to {os.path.abspath(MODEL_FILE)}")

if __name__ == "__main__":
    main()

