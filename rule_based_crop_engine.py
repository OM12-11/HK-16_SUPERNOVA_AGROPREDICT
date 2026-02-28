import os
import sys
import math
import pandas as pd
import numpy as np
import joblib

DATA_FILE = "master_agri_decision_dataset.csv"
MODEL_FILE = "spoilage_model.pkl"

DEFAULT_THRESHOLDS = {
    "nitrogen_low": 60.0,
    "rainfall_high": 800.0,
    "rainfall_low": 400.0,
    "temp_high": 28.0,
    "temp_moderate_low": 18.0,
    "ph_low": 6.0,
    "ph_high": 7.5,
    "ph_very_high": 8.0,
}

DURABLE_CROPS = ["Wheat", "Pulses", "Millets", "Barley", "Oilseeds"]
PERISHABLE_CROPS = ["Tomato", "Leafy Vegetables"]
CANDIDATE_CROPS = [
    "Legumes",
    "Pulses",
    "Wheat",
    "Maize",
    "Millets",
    "Barley",
    "Rice",
    "Paddy",
    "Oilseeds",
    "Tomato",
    "Leafy Vegetables",
]

def risk_from_loss(x):
    if x < 10:
        return "Low"
    if x < 25:
        return "Medium"
    return "High"

def ensure_spoilage_outputs(df):
    needs_pred = ("Predicted_Loss_Percentage" not in df.columns) or ("Risk_Level" not in df.columns)
    if not needs_pred:
        return df
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError("Missing spoilage_model.pkl and dataset lacks predicted columns")
    model = joblib.load(MODEL_FILE)
    required_features = [
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
        "Storage_Stress_Index",
        "Climate_Risk_Index",
    ]
    df = df.copy()
    if "Storage_Stress_Index" not in df.columns and "Temperature_C" in df.columns and "Humidity_percent_x" in df.columns:
        df["Storage_Stress_Index"] = pd.to_numeric(df["Temperature_C"], errors="coerce") * pd.to_numeric(df["Humidity_percent_x"], errors="coerce")
    if "Climate_Risk_Index" not in df.columns and all(col in df.columns for col in ["Rainfall_mm", "Wind_Speed_kmh", "Soil_Moisture_percent"]):
        df["Climate_Risk_Index"] = pd.to_numeric(df["Rainfall_mm"], errors="coerce") + pd.to_numeric(df["Wind_Speed_kmh"], errors="coerce") + pd.to_numeric(df["Soil_Moisture_percent"], errors="coerce")
    miss = [c for c in required_features if c not in df.columns]
    if miss:
        raise KeyError("Missing features for prediction: " + ", ".join(miss))
    y_pred = model.predict(df[required_features])
    df["Predicted_Loss_Percentage"] = y_pred
    df["Risk_Level"] = df["Predicted_Loss_Percentage"].apply(risk_from_loss)
    return df

def calculate_soil_score(row, t):
    scores = {c: 6.0 for c in CANDIDATE_CROPS}
    n = row.get("Nitrogen_N_kg_per_ha", np.nan)
    try:
        n = float(n)
    except Exception:
        n = np.nan
    if not math.isnan(n) and n < t["nitrogen_low"]:
        scores["Legumes"] = 9.0
        scores["Pulses"] = 8.0
        scores["Wheat"] = 6.0
    else:
        for c in ["Wheat", "Maize", "Rice", "Barley"]:
            scores[c] = max(scores[c], 8.0)
    prev = str(row.get("Previous_Crop", "")).strip().lower()
    if prev == "rice":
        scores["Legumes"] = max(scores["Legumes"], 9.0)
        scores["Pulses"] = max(scores["Pulses"], 8.0)
        scores["Rice"] = 3.0
        scores["Paddy"] = 3.0
    if prev == "wheat":
        scores["Pulses"] = max(scores["Pulses"], 8.0)
        scores["Oilseeds"] = max(scores["Oilseeds"], 8.0)
    ph = row.get("Soil_pH", np.nan)
    try:
        ph = float(ph)
    except Exception:
        ph = np.nan
    if not math.isnan(ph):
        if ph < t["ph_low"]:
            scores["Millets"] = max(scores["Millets"], 9.0)
            scores["Pulses"] = max(scores["Pulses"], 8.0)
        elif t["ph_low"] <= ph <= t["ph_high"]:
            for c in ["Wheat", "Maize", "Rice"]:
                scores[c] = max(scores[c], 8.0)
        elif ph > t["ph_very_high"]:
            scores["Barley"] = max(scores["Barley"], 8.5)
    out = pd.Series(scores)
    out = out.clip(lower=0.0, upper=10.0)
    return out

def calculate_climate_score(row, t):
    scores = {c: 6.0 for c in CANDIDATE_CROPS}
    r = row.get("Rainfall_mm", np.nan)
    temp = row.get("Daily_Temperature_C", np.nan)
    try:
        r = float(r)
        temp = float(temp)
    except Exception:
        r = np.nan
        temp = np.nan
    if not math.isnan(r) and not math.isnan(temp):
        if r > t["rainfall_high"] and temp > t["temp_high"]:
            scores["Paddy"] = 9.0
            scores["Maize"] = 9.0
        elif r < t["rainfall_low"] and temp > t["temp_high"]:
            scores["Millets"] = 9.0
        elif t["rainfall_low"] <= r <= t["rainfall_high"]:
            scores["Wheat"] = max(scores["Wheat"], 8.0)
    out = pd.Series(scores)
    out = out.clip(lower=0.0, upper=10.0)
    return out

def calculate_storage_score(row):
    scores = {c: 6.0 for c in CANDIDATE_CROPS}
    risk = str(row.get("Risk_Level", "")).strip().upper()
    if risk == "HIGH":
        for c in DURABLE_CROPS:
            scores[c] = 9.0
        for c in PERISHABLE_CROPS:
            scores[c] = 3.0
    elif risk == "LOW":
        for c in PERISHABLE_CROPS:
            scores[c] = 9.0
        for c in DURABLE_CROPS:
            scores[c] = max(scores[c], 7.0)
    else:
        for c in DURABLE_CROPS:
            scores[c] = max(scores[c], 7.5)
        for c in PERISHABLE_CROPS:
            scores[c] = max(scores[c], 5.0)
    out = pd.Series(scores)
    out = out.clip(lower=0.0, upper=10.0)
    return out

def calculate_final_score(soil_scores, climate_scores, storage_scores):
    fs = 0.4 * soil_scores + 0.3 * climate_scores + 0.3 * storage_scores
    return fs

def explanation_for_row(row, t, top_crop):
    reasons = []
    n = row.get("Nitrogen_N_kg_per_ha", np.nan)
    try:
        n = float(n)
    except Exception:
        n = np.nan
    if not math.isnan(n) and n < t["nitrogen_low"] and top_crop in ["Legumes", "Pulses"]:
        reasons.append("low nitrogen soil")
    ph = row.get("Soil_pH", np.nan)
    try:
        ph = float(ph)
    except Exception:
        ph = np.nan
    if not math.isnan(ph):
        if ph < t["ph_low"] and top_crop in ["Millets", "Pulses"]:
            reasons.append("acidic soil pH")
        elif t["ph_low"] <= ph <= t["ph_high"] and top_crop in ["Wheat", "Maize", "Rice", "Paddy"]:
            reasons.append("neutral soil pH")
        elif ph > t["ph_very_high"] and top_crop in ["Barley"]:
            reasons.append("alkaline soil pH")
    risk = str(row.get("Risk_Level", "")).strip().upper()
    if risk == "HIGH" and top_crop in DURABLE_CROPS:
        reasons.append("high spoilage risk")
    if risk == "LOW" and top_crop in PERISHABLE_CROPS:
        reasons.append("low spoilage risk")
    r = row.get("Rainfall_mm", np.nan)
    temp = row.get("Daily_Temperature_C", np.nan)
    try:
        r = float(r)
        temp = float(temp)
    except Exception:
        r = np.nan
        temp = np.nan
    if not math.isnan(r) and not math.isnan(temp):
        if r > t["rainfall_high"] and temp > t["temp_high"] and top_crop in ["Paddy", "Maize", "Rice"]:
            reasons.append("high rainfall and high temperature")
        elif r < t["rainfall_low"] and temp > t["temp_high"] and top_crop in ["Millets"]:
            reasons.append("low rainfall and high temperature")
        elif t["rainfall_low"] <= r <= t["rainfall_high"] and top_crop in ["Wheat"]:
            reasons.append("moderate rainfall conditions")
    if not reasons:
        return f"{top_crop} recommended based on overall balanced suitability."
    return f"{top_crop} recommended due to " + ", ".join(reasons) + "."

def main():
    if not os.path.exists(DATA_FILE):
        print(f"Dataset not found at: {os.path.abspath(DATA_FILE)}")
        sys.exit(1)
    df = pd.read_csv(DATA_FILE)
    limit = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--limit" and i + 2 <= len(sys.argv[1:]):
            try:
                limit = int(sys.argv[1:][i + 1])
            except Exception:
                limit = None
    if limit is not None and limit > 0:
        df = df.head(limit)
    needed = [
        "Soil_Type",
        "Soil_pH",
        "Nitrogen_N_kg_per_ha",
        "Phosphorus_P_kg_per_ha",
        "Potassium_K_kg_per_ha",
        "Organic_Carbon_percent",
        "Land_Degradation_Index",
        "Previous_Crop",
        "Rainfall_mm",
        "Daily_Temperature_C",
        "Seasonal_Forecast",
    ]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        print("Missing required columns:", ", ".join(miss))
        sys.exit(1)
    df = ensure_spoilage_outputs(df)
    t = DEFAULT_THRESHOLDS
    soil_scores_df = df.apply(lambda r: calculate_soil_score(r, t), axis=1, result_type="expand")
    climate_scores_df = df.apply(lambda r: calculate_climate_score(r, t), axis=1, result_type="expand")
    storage_scores_df = df.apply(lambda r: calculate_storage_score(r), axis=1, result_type="expand")
    final_scores_df = calculate_final_score(soil_scores_df, climate_scores_df, storage_scores_df)
    top_crops = final_scores_df.idxmax(axis=1)
    df_out = df.copy()
    df_out["Recommended_New_Crop"] = top_crops.values
    def pick_score(row, source_df, crop):
        col = crop
        return row[col] if col in row else np.nan
    merged = pd.concat([df_out, soil_scores_df.add_prefix("soil__"), climate_scores_df.add_prefix("clim__"), storage_scores_df.add_prefix("stor__"), final_scores_df.add_prefix("final__")], axis=1)
    merged["Soil_Score"] = merged.apply(lambda r: r.get("soil__" + r["Recommended_New_Crop"], np.nan), axis=1)
    merged["Climate_Score"] = merged.apply(lambda r: r.get("clim__" + r["Recommended_New_Crop"], np.nan), axis=1)
    merged["Storage_Risk_Score"] = merged.apply(lambda r: r.get("stor__" + r["Recommended_New_Crop"], np.nan), axis=1)
    merged["Final_Score"] = merged.apply(lambda r: r.get("final__" + r["Recommended_New_Crop"], np.nan), axis=1)
    merged["Recommendation_Explanation"] = merged.apply(lambda r: explanation_for_row(r, t, r["Recommended_New_Crop"]), axis=1)
    cols_show = [
        "Recommended_New_Crop",
        "Soil_Score",
        "Climate_Score",
        "Storage_Risk_Score",
        "Final_Score",
        "Recommendation_Explanation",
    ]
    print("Sample recommendations (10)")
    print(merged[cols_show].head(10).to_string(index=False))

if __name__ == "__main__":
    main()
