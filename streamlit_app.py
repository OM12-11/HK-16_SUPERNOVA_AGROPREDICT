import os
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import altair as alt
import price_forecast as pf
import rule_based_crop_engine as rbe
import requests
from concurrent.futures import ThreadPoolExecutor
from streamlit_folium import st_folium
import folium

st.set_page_config(page_title="AGROPREDICT Dashboard", layout="wide")

model_path = "spoilage_model.pkl"
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

st.title("AGROPREDICT Dashboard")
st.subheader("Spoilage Prediction and Rule-Based Crop Recommendation")

with st.sidebar:
    st.header("Thresholds")
    t = dict(rbe.DEFAULT_THRESHOLDS)
    t["nitrogen_low"] = st.number_input("Nitrogen low threshold", value=float(t["nitrogen_low"]))
    t["rainfall_high"] = st.number_input("Rainfall high threshold", value=float(t["rainfall_high"]))
    t["rainfall_low"] = st.number_input("Rainfall low threshold", value=float(t["rainfall_low"]))
    t["temp_high"] = st.number_input("Temperature high threshold", value=float(t["temp_high"]))
    t["ph_low"] = st.number_input("pH low", value=float(t["ph_low"]))
    t["ph_high"] = st.number_input("pH high", value=float(t["ph_high"]))
    t["ph_very_high"] = st.number_input("pH very high", value=float(t["ph_very_high"]))
    st.subheader("🔐 API Keys (Required for Real-Time Fetch)")
    weather_api_key = st.text_input("OpenWeatherMap API Key", type="password", value=st.session_state.get("weather_api_key", ""))
    if weather_api_key:
        st.session_state["weather_api_key"] = weather_api_key

@st.cache_data(ttl=3600)
def fetch_openweather(lat, lon, api_key):
    if not api_key:
        raise Exception("OpenWeatherMap Error: missing_api_key")
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": float(lat), "lon": float(lon), "appid": api_key, "units": "metric"}
    r = requests.get(url, params=params, timeout=10)
    if r.status_code != 200:
        raise Exception(f"OpenWeatherMap Error: {r.status_code} - {r.text}")
    return r.json()

@st.cache_data(ttl=3600)
def fetch_nasa_power(lat, lon):
    end = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=7)
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = {
        "parameters": "T2M,PRECTOTCORR",
        "community": "AG",
        "longitude": float(lon),
        "latitude": float(lat),
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise Exception(f"NASA POWER Error: {r.status_code} - {r.text}")
    return r.json()

@st.cache_data(ttl=3600)
def fetch_soilgrids(lat, lon):
    url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    base = [
        ("lon", float(lon)),
        ("lat", float(lat)),
        ("depth", "0-5cm"),
        ("value", "mean"),
    ]
    props = ["phh2o", "ocd", "clay", "sand", "silt", "nitrogen"]
    params = base + [("property", p) for p in props]
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        raise Exception(f"SoilGrids API Error: {r.status_code} - {r.text}")
    return r.json()

def derive_extreme(cond, temp_c, rain_mm, wind_kmh, thresholds):
    if cond in ["Thunderstorm", "Squall"]:
        return "Storm"
    if cond in ["Tornado"]:
        return "Cyclone"
    if cond in ["Rain", "Drizzle"] and isinstance(rain_mm, (int, float)) and rain_mm >= float(thresholds["rainfall_high"]):
        return "Flood"
    if isinstance(temp_c, (int, float)) and temp_c >= float(thresholds["temp_high"]):
        return "Heatwave"
    if isinstance(rain_mm, (int, float)) and rain_mm < float(thresholds["rainfall_low"]):
        return "Drought"
    return "None"

def derive_seasonal(rain_mm, thresholds):
    if not isinstance(rain_mm, (int, float)) or np.isnan(rain_mm):
        return "Normal"
    if rain_mm < float(thresholds["rainfall_low"]):
        return "Below Normal"
    if rain_mm > float(thresholds["rainfall_high"]):
        return "Above Normal"
    return "Normal"

def normalize_longitude(lon):
    return ((float(lon) + 180.0) % 360.0) - 180.0

def clean_value(value, min_allowed=None):
    try:
        v = float(value)
    except Exception:
        return None
    if v is None or np.isnan(v):
        return None
    if v <= -999:
        return None
    if min_allowed is not None:
        try:
            m = float(min_allowed)
        except Exception:
            m = None
        if m is not None and v < m:
            return None
    return v

def validate_coordinates(lat, lon):
    if lat is None or lon is None:
        raise Exception("Please select a location on the map.")
    try:
        lat = float(str(lat).strip())
        lon = float(str(lon).strip())
    except (ValueError, TypeError):
        raise Exception("Coordinates must be numeric.")
    if lat < -90 or lat > 90:
        raise Exception(f"Invalid latitude: {lat}. Must be between -90 and 90.")
    if lon < -180 or lon > 180:
        raise Exception(f"Invalid longitude: {lon}. Must be between -180 and 180.")
    return lat, lon

def fetch_weather(lat, lon, api_key):
    return fetch_openweather(lat, lon, api_key)

def fetch_soil(lat, lon):
    return fetch_soilgrids(lat, lon)

def fetch_nasa(lat, lon):
    return fetch_nasa_power(lat, lon)

def normalize_data(weather, soil, nasa, thresholds):
    temp_c = float(weather.get("main", {}).get("temp", np.nan))
    hum = float(weather.get("main", {}).get("humidity", np.nan))
    wind_ms = float(weather.get("wind", {}).get("speed", np.nan))
    rain = 0.0
    if isinstance(weather.get("rain"), dict):
        rain = float(weather["rain"].get("1h", 0.0))
    cond = None
    if isinstance(weather.get("weather"), list) and weather["weather"]:
        cond = weather["weather"][0].get("main")
    d = nasa.get("properties", {}).get("parameter", {})
    t2m = pd.Series(d.get("T2M", {}))
    pre = pd.Series(d.get("PRECTOTCORR", {}))
    daily_t = None if t2m.empty else float(pd.to_numeric(t2m, errors="coerce").mean())
    rainfall_latest = None if pre.empty else float(pd.to_numeric(pre, errors="coerce").iloc[-1])
    layers = soil.get("properties", {}).get("layers", [])
    soil_pH = None
    oc_percent = None
    for layer in layers:
        name = layer.get("name")
        vs = layer.get("values", {})
        v = None
        if isinstance(vs, dict):
            v = vs.get("avg") or vs.get("median") or vs.get("value")
        if name == "phh2o":
            soil_pH = None if v is None else float(v)
        if name == "ocd":
            oc_gkg = None if v is None else float(v)
            oc_percent = None if oc_gkg is None else oc_gkg / 10.0
    wind_kmh = None if np.isnan(wind_ms) else float(wind_ms) * 3.6
    extreme = derive_extreme(cond, temp_c, rainfall_latest, wind_kmh, thresholds)
    seasonal = derive_seasonal(rainfall_latest, thresholds)
    return {
        "Temperature_C": temp_c,
        "Daily_Temperature_C": daily_t,
        "Humidity_percent": hum,
        "Wind_Speed_kmh": wind_kmh,
        "Rainfall_mm": rainfall_latest if rainfall_latest is not None else rain,
        "Soil_pH": soil_pH,
        "Organic_Carbon_percent": oc_percent,
        "Extreme_Weather_Event": extreme,
        "Seasonal_Forecast": seasonal,
    }

def normalize_partial(weather, soil, nasa, thresholds):
    out = {}
    temp_c = None
    hum = None
    wind_ms = None
    rain_w = None
    cond = None
    if isinstance(weather, dict):
        temp_c = weather.get("main", {}).get("temp")
        hum = weather.get("main", {}).get("humidity")
        wind_ms = weather.get("wind", {}).get("speed")
        if isinstance(weather.get("rain"), dict):
            rain_w = weather["rain"].get("1h", 0.0)
        if isinstance(weather.get("weather"), list) and weather["weather"]:
            cond = weather["weather"][0].get("main")
    daily_t = None
    rain_n = None
    if isinstance(nasa, dict):
        d = nasa.get("properties", {}).get("parameter", {})
        t2m = pd.Series(d.get("T2M", {}))
        pre = pd.Series(d.get("PRECTOTCORR", {}))
        daily_t = None if t2m.empty else float(pd.to_numeric(t2m, errors="coerce").mean())
        rain_n = None if pre.empty else float(pd.to_numeric(pre, errors="coerce").iloc[-1])
    soil_pH = None
    oc_percent = None
    soil_type = None
    nitrogen_kg_ha = None
    if isinstance(soil, dict):
        layers = soil.get("properties", {}).get("layers", [])
        clay = None
        sand = None
        for layer in layers:
            name = layer.get("name")
            vs = layer.get("values", {})
            v = None
            if isinstance(vs, dict):
                v = vs.get("mean") or vs.get("avg") or vs.get("median") or vs.get("value")
            if name == "phh2o":
                soil_pH = None if v is None else float(v)
            if name == "ocd":
                oc_gkg = None if v is None else float(v)
                oc_percent = None if oc_gkg is None else oc_gkg / 10.0
            if name == "clay":
                clay = None if v is None else float(v)
            if name == "sand":
                sand = None if v is None else float(v)
            if name == "nitrogen":
                n_gkg = None if v is None else float(v)
                nitrogen_kg_ha = None if n_gkg is None else float(n_gkg) * 10.0
        if clay is not None and clay > 40:
            soil_type = "Clay"
        elif sand is not None and sand > 70:
            soil_type = "Sandy"
        elif clay is not None or sand is not None:
            soil_type = "Loam"
    wind_kmh = None if wind_ms is None or np.isnan(wind_ms) else float(wind_ms) * 3.6
    rain_final = rain_n if isinstance(rain_n, (int, float)) else (rain_w if isinstance(rain_w, (int, float)) else None)
    extreme = derive_extreme(cond, temp_c, rain_final, wind_kmh, thresholds) if (cond or temp_c or rain_final or wind_kmh) else None
    seasonal = derive_seasonal(rain_final, thresholds) if isinstance(rain_final, (int, float)) else None
    if isinstance(temp_c, (int, float)):
        out["Temperature_C"] = float(temp_c)
    if isinstance(daily_t, (int, float)):
        out["Daily_Temperature_C"] = float(daily_t)
    if isinstance(hum, (int, float)):
        out["Humidity_percent"] = float(hum)
    if isinstance(wind_kmh, (int, float)):
        out["Wind_Speed_kmh"] = float(wind_kmh)
    if isinstance(rain_final, (int, float)):
        out["Rainfall_mm"] = float(rain_final)
    if isinstance(soil_pH, (int, float)):
        out["Soil_pH"] = float(soil_pH)
    if isinstance(oc_percent, (int, float)):
        out["Organic_Carbon_percent"] = float(oc_percent)
    if isinstance(nitrogen_kg_ha, (int, float)):
        out["Nitrogen_N_kg_per_ha"] = float(nitrogen_kg_ha)
    if isinstance(soil_type, str):
        out["Soil_Type"] = soil_type
    if isinstance(extreme, str):
        out["Extreme_Weather_Event"] = extreme
    if isinstance(seasonal, str):
        out["Seasonal_Forecast"] = seasonal
    return out
def update_session_state(data):
    numeric_min = {
        "Temperature_C": None,
        "Daily_Temperature_C": None,
        "Humidity_percent": 0.0,
        "Wind_Speed_kmh": 0.0,
        "Rainfall_mm": 0.0,
        "Soil_Moisture_percent": 0.0,
        "Soil_pH": 0.0,
        "Organic_Carbon_percent": 0.0,
        "Nitrogen_N_kg_per_ha": 0.0,
        "Phosphorus_P_kg_per_ha": 0.0,
        "Potassium_K_kg_per_ha": 0.0,
    }
    for k, v in data.items():
        if k in numeric_min:
            cleaned = clean_value(v, numeric_min[k])
            if cleaned is None:
                continue
            if k == "Humidity_percent":
                st.session_state["Humidity_percent_x"] = float(cleaned)
                st.session_state["Humidity_percent_y"] = float(cleaned)
            st.session_state[k] = float(cleaned)
        else:
            if v is None:
                continue
            st.session_state[k] = v

def fetch_geospatial_sequential(lat, lon, weather_key, thresholds):
    if not (-90.0 <= float(lat) <= 90.0 and -180.0 <= float(lon) <= 180.0):
        raise Exception("Invalid coordinates")
    weather = None
    soil = None
    nasa = None
    w_err = None
    s_err = None
    n_err = None
    try:
        weather = fetch_weather(lat, lon, weather_key)
    except Exception as e:
        w_err = str(e)
    try:
        soil = fetch_soil(lat, lon)
    except Exception as e:
        s_err = str(e)
    try:
        nasa = fetch_nasa(lat, lon)
    except Exception as e:
        n_err = str(e)
    if weather is None and soil is None and nasa is None:
        errs = ", ".join([x for x in [w_err, s_err, n_err] if x])
        raise Exception(errs or "All APIs failed")
    data = normalize_partial(weather, soil, nasa, thresholds)
    return data

tab1, tab2, tab3 = st.tabs(["Single Record", "Batch Upload", "Market Optimizer"])

def predict_and_score(df_row, thresholds):
    df_row = df_row.copy()
    df_row["Storage_Stress_Index"] = pd.to_numeric(df_row.get("Temperature_C", np.nan), errors="coerce") * pd.to_numeric(df_row.get("Humidity_percent_x", np.nan), errors="coerce")
    df_row["Climate_Risk_Index"] = pd.to_numeric(df_row.get("Rainfall_mm", np.nan), errors="coerce") + pd.to_numeric(df_row.get("Wind_Speed_kmh", np.nan), errors="coerce") + pd.to_numeric(df_row.get("Soil_Moisture_percent", np.nan), errors="coerce")
    if model is not None:
        req = [
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
        for c in req:
            if c not in df_row.columns:
                df_row[c] = np.nan
        y_pred = model.predict(df_row[req])[0]
        risk = rbe.risk_from_loss(y_pred)
    else:
        y_pred = np.nan
        risk = "UNKNOWN"
    r_series = df_row.iloc[0]
    r_series["Predicted_Loss_Percentage"] = y_pred
    r_series["Risk_Level"] = risk
    soil_scores = rbe.calculate_soil_score(r_series, thresholds)
    climate_scores = rbe.calculate_climate_score(r_series, thresholds)
    storage_scores = rbe.calculate_storage_score(r_series)
    final_scores = rbe.calculate_final_score(soil_scores, climate_scores, storage_scores)
    top_crop = final_scores.idxmax()
    explanation = rbe.explanation_for_row(r_series, thresholds, top_crop)
    return y_pred, risk, soil_scores, climate_scores, storage_scores, final_scores, top_crop, explanation

with tab1:
    st.write("📍 Select Farm Location")
    if "selected_lat" not in st.session_state:
        st.session_state["selected_lat"] = None
        st.session_state["selected_lon"] = None
    m = folium.Map(location=[20, 0], zoom_start=2)
    map_data = st_folium(m, height=300, returned_objects=["last_clicked"])
    if isinstance(map_data, dict) and isinstance(map_data.get("last_clicked"), dict):
        lat_val = map_data["last_clicked"].get("lat")
        lon_val = map_data["last_clicked"].get("lng")
        try:
            st.session_state["lat"] = float(str(lat_val).strip())
            st.session_state["lon"] = normalize_longitude(float(str(lon_val).strip()))
            st.session_state["selected_lat"] = st.session_state["lat"]
            st.session_state["selected_lon"] = st.session_state["lon"]
        except Exception:
            pass
    st.write(f"Coordinates: {st.session_state.get('lat')} , {st.session_state.get('lon')}")
    fetch_btn = st.button("🔄 Fetch Data from Location")
    if fetch_btn:
        lat_raw = st.session_state.get("lat")
        lon_raw = st.session_state.get("lon")
        weather_key = st.session_state.get("weather_api_key")
        try:
            lat, lon = validate_coordinates(lat_raw, lon_raw)
        except Exception as e:
            st.error(f"Geospatial fetch failed: {e}")
            st.info("You may enter values manually below.")
        else:
            if not weather_key:
                st.warning("OpenWeatherMap API key required.")
            else:
                try:
                    with st.spinner("Fetching real-time geospatial data..."):
                        data = fetch_geospatial_sequential(lat, lon, weather_key, t)
                    missing_fields = []
                    expected_fields = [
                        "Temperature_C","Humidity_percent","Wind_Speed_kmh","Rainfall_mm",
                        "Soil_Moisture_percent","Soil_Type","Soil_pH","Organic_Carbon_percent","Nitrogen_N_kg_per_ha",
                        "Daily_Temperature_C"
                    ]
                    for f in expected_fields:
                        if data.get(f) is None:
                            missing_fields.append(f)
                    if missing_fields:
                        st.warning(f"Some fields could not be auto-fetched: {', '.join(missing_fields)}")
                        st.info("You can manually enter these values below.")
                    update_session_state(data)
                    st.success("Geospatial data fetched successfully!")
                except Exception as e:
                    st.error(f"Geospatial fetch failed: {e}")
                    st.info("You may enter values manually below.")
    st.write("Enter a new record to predict spoilage risk and recommend a crop.")
    cols1 = st.columns(3)
    with cols1[0]:
        Storage_Type = st.selectbox("Storage_Type", ["Open Storage", "Warehouse", "Cold Storage"])
        Storage_Duration_days = st.number_input("Storage_Duration_days", min_value=0.0, value=10.0)
        Temperature_C = st.number_input("Temperature_C", value=float(st.session_state.get("Temperature_C", 25.0)))
        Humidity_percent_x = st.number_input("Humidity_percent_x", value=float(st.session_state.get("Humidity_percent_x", 60.0)))
    with cols1[1]:
        Transport_Time_hours = st.number_input("Transport_Time_hours", min_value=0.0, value=5.0)
        Daily_Temperature_C = st.number_input("Daily_Temperature_C", value=float(st.session_state.get("Daily_Temperature_C", 28.0)))
        Humidity_percent_y = st.number_input("Humidity_percent_y", value=float(st.session_state.get("Humidity_percent_y", 55.0)))
        Rainfall_mm = st.number_input("Rainfall_mm", min_value=0.0, value=float(st.session_state.get("Rainfall_mm", 600.0)))
    with cols1[2]:
        Wind_Speed_kmh = st.number_input("Wind_Speed_kmh", min_value=0.0, value=float(st.session_state.get("Wind_Speed_kmh", 10.0)))
        Soil_Moisture_percent = st.number_input("Soil_Moisture_percent", min_value=0.0, value=float(st.session_state.get("Soil_Moisture_percent", 20.0)))
        Extreme_Weather_Event = st.selectbox("Extreme_Weather_Event", ["None", "Drought", "Heatwave", "Flood", "Cyclone", "Storm"], index=["None","Drought","Heatwave","Flood","Cyclone","Storm"].index(st.session_state.get("Extreme_Weather_Event", "None")))
        Seasonal_Forecast = st.text_input("Seasonal_Forecast", value=str(st.session_state.get("Seasonal_Forecast", "Normal")))
    cols2 = st.columns(3)
    with cols2[0]:
        Soil_Type = st.text_input("Soil_Type", value=str(st.session_state.get("Soil_Type", "Loam")))
        Soil_pH = st.number_input("Soil_pH", min_value=0.0, value=float(st.session_state.get("Soil_pH", 6.8)))
    with cols2[1]:
        Nitrogen_N_kg_per_ha = st.number_input("Nitrogen_N_kg_per_ha", min_value=0.0, value=float(st.session_state.get("Nitrogen_N_kg_per_ha", 70.0)))
        Phosphorus_P_kg_per_ha = st.number_input("Phosphorus_P_kg_per_ha", min_value=0.0, value=float(st.session_state.get("Phosphorus_P_kg_per_ha", 40.0)))
    with cols2[2]:
        Potassium_K_kg_per_ha = st.number_input("Potassium_K_kg_per_ha", min_value=0.0, value=float(st.session_state.get("Potassium_K_kg_per_ha", 30.0)))
        Organic_Carbon_percent = st.number_input("Organic_Carbon_percent", min_value=0.0, value=float(st.session_state.get("Organic_Carbon_percent", 0.8)))
    Land_Degradation_Index = st.number_input("Land_Degradation_Index", min_value=0.0, value=0.2)
    Previous_Crop = st.selectbox("Previous_Crop", ["None", "Rice", "Wheat", "Maize", "Pulses", "Oilseeds"])
    if st.button("Predict and Recommend"):
        row_df = pd.DataFrame([{
            "Storage_Type": Storage_Type,
            "Storage_Duration_days": Storage_Duration_days,
            "Temperature_C": Temperature_C,
            "Humidity_percent_x": Humidity_percent_x,
            "Transport_Time_hours": Transport_Time_hours,
            "Daily_Temperature_C": Daily_Temperature_C,
            "Humidity_percent_y": Humidity_percent_y,
            "Rainfall_mm": Rainfall_mm,
            "Wind_Speed_kmh": Wind_Speed_kmh,
            "Soil_Moisture_percent": Soil_Moisture_percent,
            "Extreme_Weather_Event": Extreme_Weather_Event,
            "Soil_Type": Soil_Type,
            "Soil_pH": Soil_pH,
            "Nitrogen_N_kg_per_ha": Nitrogen_N_kg_per_ha,
            "Phosphorus_P_kg_per_ha": Phosphorus_P_kg_per_ha,
            "Potassium_K_kg_per_ha": Potassium_K_kg_per_ha,
            "Organic_Carbon_percent": Organic_Carbon_percent,
            "Land_Degradation_Index": Land_Degradation_Index,
            "Previous_Crop": Previous_Crop,
            "Seasonal_Forecast": Seasonal_Forecast,
        }])
        y_pred, risk, soil_scores, climate_scores, storage_scores, final_scores, top_crop, explanation = predict_and_score(row_df, t)
        st.metric("Predicted Loss %", None if np.isnan(y_pred) else round(float(y_pred), 4))
        st.metric("Predicted Risk Level", risk)
        st.subheader("Recommendation")
        st.write(f"Recommended_New_Crop: {top_crop}")
        chosen = pd.DataFrame({
            "Soil_Score": [soil_scores[top_crop]],
            "Climate_Score": [climate_scores[top_crop]],
            "Storage_Risk_Score": [storage_scores[top_crop]],
            "Final_Score": [final_scores[top_crop]],
            "Recommendation_Explanation": [explanation],
        })
        st.table(chosen)
        st.subheader("Top 5 crops by Final Score")
        top5 = final_scores.sort_values(ascending=False).head(5)
        st.table(pd.DataFrame({"Crop": top5.index, "Final_Score": top5.values}))

with tab2:
    st.write("Upload CSV with same columns as your merged dataset.")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        try:
            df_in = rbe.ensure_spoilage_outputs(df_in)
        except Exception as e:
            st.warning(str(e))
            if model is not None:
                pass
        soil_scores_df = df_in.apply(lambda r: rbe.calculate_soil_score(r, t), axis=1, result_type="expand")
        climate_scores_df = df_in.apply(lambda r: rbe.calculate_climate_score(r, t), axis=1, result_type="expand")
        storage_scores_df = df_in.apply(lambda r: rbe.calculate_storage_score(r), axis=1, result_type="expand")
        final_scores_df = rbe.calculate_final_score(soil_scores_df, climate_scores_df, storage_scores_df)
        top_crops = final_scores_df.idxmax(axis=1)
        df_out = df_in.copy()
        df_out["Recommended_New_Crop"] = top_crops.values
        merged = pd.concat([df_out, soil_scores_df.add_prefix("soil__"), climate_scores_df.add_prefix("clim__"), storage_scores_df.add_prefix("stor__"), final_scores_df.add_prefix("final__")], axis=1)
        merged["Soil_Score"] = merged.apply(lambda r: r.get("soil__" + r["Recommended_New_Crop"], np.nan), axis=1)
        merged["Climate_Score"] = merged.apply(lambda r: r.get("clim__" + r["Recommended_New_Crop"], np.nan), axis=1)
        merged["Storage_Risk_Score"] = merged.apply(lambda r: r.get("stor__" + r["Recommended_New_Crop"], np.nan), axis=1)
        merged["Final_Score"] = merged.apply(lambda r: r.get("final__" + r["Recommended_New_Crop"], np.nan), axis=1)
        merged["Recommendation_Explanation"] = merged.apply(lambda r: rbe.explanation_for_row(r, t, r["Recommended_New_Crop"]), axis=1)
        st.subheader("Sample (10)")
        st.dataframe(merged[["Recommended_New_Crop","Soil_Score","Climate_Score","Storage_Risk_Score","Final_Score","Recommendation_Explanation"]].head(10))
        csv = merged.to_csv(index=False).encode("utf-8")
        st.download_button("Download results CSV", data=csv, file_name="agropredict_recommendations.csv", mime="text/csv")

with tab3:
    st.write("Market optimization and price forecasting")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("Price history input")
        ph_file = st.file_uploader("Upload price history CSV", type=["csv"], key="ph")
        ph_df = None
        date_col = st.text_input("Date column", value="date")
        price_col = st.text_input("Price column", value="price")
        region = st.selectbox("Region", ["North", "South", "East", "West", "Central"], index=0)
        demand_index = st.slider("Demand index", min_value=-10.0, max_value=10.0, value=0.0)
        supply_index = st.slider("Supply index", min_value=-10.0, max_value=10.0, value=0.0)
        horizon_short = st.slider("Short-term horizon (days)", min_value=7, max_value=30, value=14)
        horizon_medium_months = st.slider("Medium-term horizon (months)", min_value=3, max_value=6, value=3)
        if ph_file is not None:
            ph_df = pd.read_csv(ph_file)
        else:
            ph_df = pd.DataFrame({
                "date": pd.date_range(pd.Timestamp.today() - pd.Timedelta(days=90), periods=90, freq="D"),
                "price": np.linspace(1800, 2000, 90) + 50 * np.sin(np.linspace(0, 8 * np.pi, 90))
            })
    with col_b:
        st.write("Transportation and logistics")
        load_kg = st.number_input("Load (kg)", min_value=1.0, value=1000.0)
        fuel_cost_per_km = st.number_input("Fuel cost per km", min_value=0.0, value=5.0)
        vehicle_type = st.selectbox("Vehicle type", ["Tractor", "Mini-truck", "Truck"], index=1)
        labor_charges = st.number_input("Labor charges", min_value=0.0, value=800.0)
    st.write("Market options")
    markets = pd.DataFrame([
        {"name": "Govt Mandi A", "type": "Government Mandi", "price_factor": 1.00, "buyer_rating": 8.5, "payment_security": 9.0, "price_consistency": 8.0, "transaction_transparency": 8.5, "distance_km": 25.0},
        {"name": "APMC Market B", "type": "APMC", "price_factor": 1.02, "buyer_rating": 7.8, "payment_security": 8.5, "price_consistency": 7.5, "transaction_transparency": 8.0, "distance_km": 45.0},
        {"name": "FPO Hub C", "type": "FPO", "price_factor": 0.98, "buyer_rating": 8.8, "payment_security": 8.8, "price_consistency": 8.2, "transaction_transparency": 8.7, "distance_km": 30.0},
        {"name": "Agri Marketplace D", "type": "Marketplace", "price_factor": 1.05, "buyer_rating": 7.2, "payment_security": 8.0, "price_consistency": 7.0, "transaction_transparency": 7.8, "distance_km": 60.0},
    ])
    st.dataframe(markets[["name","type","distance_km","buyer_rating","payment_security","price_consistency","transaction_transparency"]])
    run_btn = st.button("Run Optimization")
    if run_btn:
        short_df, trend_short, s_short = pf.forecast(ph_df, date_col, price_col, horizon_short, region, demand_index, supply_index)
        medium_days = int(horizon_medium_months * 30)
        medium_df, trend_medium, s_medium = pf.forecast(ph_df, date_col, price_col, medium_days, region, demand_index, supply_index)
        latest_short = float(short_df["yhat"].iloc[-1])
        ranked = pf.rank_markets(latest_short, markets)
        sel_market = ranked.iloc[0]
        distance_km = float(sel_market["distance_km"])
        total_cost, per_kg = pf.transport_cost(distance_km, fuel_cost_per_km, vehicle_type, load_kg, labor_charges)
        transport_info = {"total_cost": total_cost, "cost_per_kg": per_kg}
        profit_df, best_row, risk = pf.optimize_profit(ranked, load_kg, transport_info)
        st.subheader("Price forecast")
        base = alt.Chart(short_df).encode(x=alt.X("date:T"))
        band = base.mark_area(opacity=0.2).encode(y="yhat_lower:Q", y2="yhat_upper:Q", color=alt.value("#99c"))
        line = base.mark_line(color="#3366cc").encode(y="yhat:Q")
        st.altair_chart(band + line, use_container_width=True)
        st.metric("Trend direction (short)", trend_short)
        st.metric("Trend direction (medium)", trend_medium)
        st.subheader("Trusted market recommendation")
        st.table(ranked[["name","type","expected_price","distance_km","buyer_rating","payment_security"]].head(5))
        st.subheader("Transportation cost")
        st.table(pd.DataFrame({"Estimated total cost": [total_cost], "Cost per kg": [per_kg]}))
        st.subheader("Profit optimization")
        st.table(pd.DataFrame({
            "Recommended market": [best_row["name"]],
            "Ideal time to sell": ["Short-term" if trend_short == "Bullish" else "Medium-term" if trend_medium == "Bullish" else "Stable window"],
            "Expected selling price": [float(best_row["expected_price"])],
            "Estimated net profit": [float(best_row["net_profit"])],
            "Risk indicator": [risk],
        }))
