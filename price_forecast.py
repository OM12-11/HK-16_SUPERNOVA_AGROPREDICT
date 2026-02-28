import math
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def _validate_ts(df: pd.DataFrame, date_col: str, price_col: str) -> pd.DataFrame:
    if date_col not in df.columns or price_col not in df.columns:
        raise KeyError("missing date or price column")
    out = df[[date_col, price_col]].dropna()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna()
    out = out.sort_values(date_col)
    return out

def _build_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    df["t"] = (df[date_col] - df[date_col].min()).dt.days.astype(int)
    df["dow"] = df[date_col].dt.dayofweek.astype(int)
    df["month"] = df[date_col].dt.month.astype(int)
    df["sin7"] = np.sin(2 * np.pi * df["t"] / 7.0)
    df["cos7"] = np.cos(2 * np.pi * df["t"] / 7.0)
    df["sin30"] = np.sin(2 * np.pi * df["t"] / 30.0)
    df["cos30"] = np.cos(2 * np.pi * df["t"] / 30.0)
    return df

def forecast(df_in: pd.DataFrame, date_col: str, price_col: str, horizon_days: int, region: str | None = None, demand_index: float = 0.0, supply_index: float = 0.0) -> Tuple[pd.DataFrame, str, float]:
    df = _validate_ts(df_in, date_col, price_col)
    df_f = _build_features(df, date_col)
    X_num = df_f[["t", "sin7", "cos7", "sin30", "cos30"]]
    X_cat = df_f[["dow", "month"]]
    y = pd.to_numeric(df[price_col], errors="coerce")
    y = y.values
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    pre = ColumnTransformer(transformers=[("num", "passthrough", [0, 1, 2, 3, 4]), ("cat", enc, [5, 6])])
    model = LinearRegression()
    pipe = Pipeline(steps=[("prep", pre), ("model", model)])
    pipe.fit(pd.concat([X_num, X_cat], axis=1).values, y)
    resid = y - pipe.predict(pd.concat([X_num, X_cat], axis=1).values)
    s = float(np.nanstd(resid)) if len(resid) else 0.0
    last_date = df[date_col].max()
    fut_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")
    fut = pd.DataFrame({date_col: fut_dates})
    fut_f = _build_features(fut, date_col)
    Xn = fut_f[["t", "sin7", "cos7", "sin30", "cos30"]]
    Xc = fut_f[["dow", "month"]]
    yhat = pipe.predict(pd.concat([Xn, Xc], axis=1).values)
    adj = 1.0 + 0.01 * float(demand_index - supply_index)
    region_factor = 0.0
    if isinstance(region, str):
        key = region.strip().lower()
        if key in ["north", "west"]:
            region_factor = 0.01
        elif key in ["south", "east"]:
            region_factor = -0.005
    yhat = yhat * (1.0 + region_factor) * adj
    ci_k = 1.645
    lower = yhat - ci_k * s
    upper = yhat + ci_k * s
    trend = "Stable"
    if len(yhat) >= 2:
        d = yhat[-1] - yhat[0]
        if d > 0.0 and abs(d) > 0.01:
            trend = "Bullish"
        elif d < 0.0 and abs(d) > 0.01:
            trend = "Bearish"
    out = pd.DataFrame({date_col: fut_dates, "yhat": yhat, "yhat_lower": lower, "yhat_upper": upper})
    return out, trend, s

def rank_markets(forecast_price: float, markets_df: pd.DataFrame) -> pd.DataFrame:
    df = markets_df.copy()
    df["expected_price"] = float(forecast_price) * df["price_factor"]
    df["reliability_score"] = 0.4 * df["price_consistency"] + 0.3 * df["buyer_rating"] + 0.3 * df["payment_security"]
    df["rank_score"] = 0.6 * df["reliability_score"] + 0.4 * df["transaction_transparency"]
    df = df.sort_values("rank_score", ascending=False)
    return df

def transport_cost(distance_km: float, fuel_cost_per_km: float, vehicle_type: str, load_kg: float, labor_charges: float) -> Tuple[float, float]:
    vt = str(vehicle_type or "").strip().lower()
    mult = 1.0
    if vt == "tractor":
        mult = 0.9
    elif vt == "mini-truck":
        mult = 1.0
    elif vt == "truck":
        mult = 1.2
    base = float(distance_km) * float(fuel_cost_per_km) * mult
    total = base + float(labor_charges)
    per_kg = total / max(float(load_kg), 1.0)
    return total, per_kg

def optimize_profit(markets_ranked: pd.DataFrame, load_kg: float, transport_info: Dict[str, float]) -> Tuple[pd.DataFrame, pd.Series, str]:
    df = markets_ranked.copy()
    df["transport_total"] = transport_info["total_cost"]
    df["transport_per_kg"] = transport_info["cost_per_kg"]
    df["net_profit"] = float(load_kg) * (df["expected_price"] - df["transport_per_kg"])
    best = df.sort_values("net_profit", ascending=False).iloc[0]
    risk = "Low"
    if best["reliability_score"] < 6.0:
        risk = "Medium"
    if best["reliability_score"] < 4.5:
        risk = "High"
    return df, best, risk
