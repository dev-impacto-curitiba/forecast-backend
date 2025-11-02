# -*- coding: utf-8 -*-
"""
FastAPI - Risco por Bairros (Canoas)
------------------------------------
Endpoints:
- /health
- /v1/meta
- /v1/risk/by_bairro            (JSON tabular com filtros)
- /v1/risk/by_bairro/csv        (CSV)
- /v1/risk/by_bairro/top        (Top-N piores por data)
- /v1/geo/canoas/bairros_risk   (GeoJSON para mapa por data, com include de campos)
- /v1/bairros/detail            (Detalhe de um bairro em uma data; opcional dinâmica de U para a data)
- /v1/filters                   (Esquema de filtros e campos disponíveis)

Suposições de arquivos (produzidos pelos scripts anteriores):
- data/hazard/hazard_forecast.csv       -> contém pelo menos: date, H_score (e, se possível, fatores p6_pct, a72_pct, ...)
- data/u/canoas_bairros_u.csv           -> contém por bairro: U_static, U_t (dinâmico do dia do cálculo), Fragilidade_t, subíndices e métricas brutas
- data/u/canoas_bairros_u.geojson       -> geometria dos bairros + mesmas propriedades

Observação sobre "momento" (date):
- Para mapear toda a cidade (todos os bairros), usamos por padrão U_static ou U_t pré-calculado (dryness_date).
  Recalcular U para TODOS os bairros em datas arbitrárias pode ser pesado (múltiplas chamadas à Open-Meteo).
  Por isso, este endpoint usa U_static/U_t de arquivo e H_score do CSV (por data).
- No detalhe do bairro (/v1/bairros/detail) existe a opção de aplicar "dynamic=1" para recalcular U(t) para aquela data (se estiver no horizonte do forecast).
"""

from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
import yaml, json, requests
from datetime import date, datetime, timedelta, timezone

# Opcional (apenas se quiser ativar dinâmica pontual via Open-Meteo no detail)
try:
    import openmeteo_requests
    import requests_cache
    from retry_requests import retry
    OM_AVAILABLE = True
except Exception:
    OM_AVAILABLE = False

ROOT = Path("services").resolve()
DATA = ROOT / "data"
HAZARD_CSV = DATA/"hazard"/"hazard_forecast.csv"
U_CSV      = DATA/"u"/"canoas_bairros_u.csv"
U_GEOJSON  = DATA/"u"/"canoas_bairros_u.geojson"
WEIGHTS_YAML = ROOT/"configs"/"weights.yaml"

app = FastAPI(title="Canoas - Risco por Bairros API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Utils -------------------

def load_weights() -> Dict[str, Any]:
    if WEIGHTS_YAML.exists():
        return yaml.safe_load(WEIGHTS_YAML.read_text(encoding="utf-8"))
    # Defaults
    return {
        "hazard_levels": {"green_max": 0.33, "yellow_max": 0.66},
        "hazard_daily_weights": {"p6":0.25,"a72":0.25,"sm":0.15,"etd":0.10,"p1":0.10,"pp":0.05,"rd":0.10},
        "u_weights": {"perm":0.40,"macro":0.25,"cob":0.20,"micro":0.15},
    }

def try_load_hazard() -> pd.DataFrame:
    if not HAZARD_CSV.exists():
        raise HTTPException(404, detail="hazard_forecast.csv não encontrado em data/hazard/")
    df = pd.read_csv(HAZARD_CSV)
    if "date" not in df.columns or "H_score" not in df.columns:
        raise HTTPException(500, detail="hazard_forecast.csv deve conter colunas 'date' e 'H_score'.")
    df["date"] = pd.to_datetime(df["date"])
    return df

def try_load_u() -> (pd.DataFrame, gpd.GeoDataFrame):
    if not U_CSV.exists() or not U_GEOJSON.exists():
        raise HTTPException(404, detail="Arquivos de U não encontrados (CSV/GeoJSON). Rode o gerador de U por bairros.")
    dfU = pd.read_csv(U_CSV)
    gdfU = gpd.read_file(U_GEOJSON)
    # Normaliza CRS
    if gdfU.crs is None:
        gdfU.set_crs(epsg=4326, inplace=True)
    else:
        gdfU = gdfU.to_crs(epsg=4326)
    # Campo de nome do bairro
    name_field = None
    for cand in ["bairro","name","NOME","BAIRRO","Bairro"]:
        if cand in gdfU.columns:
            name_field = cand
            break
    if name_field is None:
        name_field = "bairro"
        gdfU[name_field] = gdfU["OBJECTID"].astype(str)
    # Garante presença de 'bairro' string nos dois dataframes
    if "bairro" not in dfU.columns and name_field in dfU.columns:
        dfU.rename(columns={name_field:"bairro"}, inplace=True)
    if "bairro" not in gdfU.columns and name_field in gdfU.columns:
        gdfU.rename(columns={name_field:"bairro"}, inplace=True)
    dfU["bairro"] = dfU["bairro"].astype(str)
    gdfU["bairro"] = gdfU["bairro"].astype(str)
    return dfU, gdfU

def clip_hazard_by_date(dfH: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    out = dfH.copy()
    if start:
        out = out[out["date"] >= pd.to_datetime(start)]
    if end:
        out = out[out["date"] <= pd.to_datetime(end)]
    return out.sort_values("date").reset_index(drop=True)

def latest_hazard_date(dfH: pd.DataFrame) -> pd.Timestamp:
    return dfH["date"].max()

def select_hazard_for_date(dfH: pd.DataFrame, d: Optional[str]) -> pd.DataFrame:
    if d:
        target = pd.to_datetime(d).date()
        sel = dfH[dfH["date"].dt.date == target]
        if sel.empty:
            raise HTTPException(404, detail=f"Data {target} não encontrada no hazard CSV.")
        return sel[["date","H_score"]].copy()
    # default: última data
    last = latest_hazard_date(dfH)
    return dfH[dfH["date"]==last][["date","H_score"]].copy()

def bucket_risk(x: float, thresholds: Dict[str,float]) -> str:
    g = thresholds["green_max"]
    y = thresholds["yellow_max"]
    if x < g: return "green"
    if x < y: return "yellow"
    return "red"

def apply_filters(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Filtra por risk_level, score range e subcampos (hazard/infra)."""
    out = df.copy()
    # Risk level (lista CSV)
    levels = params.get("risk_level")
    if levels:
        allowed = set([s.strip().lower() for s in levels.split(",")])
        out = out[out["Risk_level"].str.lower().isin(allowed)]
    # Score range
    min_r = params.get("min_risk"); max_r = params.get("max_risk")
    if min_r is not None:
        out = out[out["Risk_score"] >= float(min_r)]
    if max_r is not None:
        out = out[out["Risk_score"] <= float(max_r)]
    # Hazard factors (se existirem no hazard CSV)
    for k in ["p6_pct","a72_pct","sm_norm","et_deficit","p1_pct","pp_unit","rd_norm"]:
        lo = params.get(f"min_{k}"); hi = params.get(f"max_{k}")
        if lo is not None and k in out.columns: out = out[out[k] >= float(lo)]
        if hi is not None and k in out.columns: out = out[out[k] <= float(hi)]
    # Infra subindices
    for k in ["u_cobertura","u_micro","u_macro","u_permeabilidade"]:
        lo = params.get(f"min_{k}"); hi = params.get(f"max_{k}")
        if lo is not None and k in out.columns: out = out[out[k] >= float(lo)]
        if hi is not None and k in out.columns: out = out[out[k] <= float(hi)]
    return out.reset_index(drop=True)

# -------------- Dinâmica pontual de U (opcional) --------------
TZ = "America/Sao_Paulo"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
if OM_AVAILABLE:
    cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
    retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
    om_client = openmeteo_requests.Client(session=retry_session)

def compute_dryness_for_date(lat: float, lon: float, target_date: pd.Timestamp) -> Dict[str, float]:
    """
    Recalcula dryness (sm + ET) para uma data específica:
    - Se target_date ∈ [hoje-2d, hoje+16d] -> usa forecast (past_days + forecast_days)
    - Caso contrário -> retorna None para indicar fora do alcance (frontend pode usar U_static)
    """
    if not OM_AVAILABLE:
        return {"sm_norm": None, "et_scaled": None, "dryness": None}

    today = pd.Timestamp(date.today(), tz=timezone.utc).tz_convert(TZ).date()
    td = pd.to_datetime(target_date).date()

    # janela suportada
    if td < (today - timedelta(days=2)) or td > (today + timedelta(days=16)):
        return {"sm_norm": None, "et_scaled": None, "dryness": None}

    hourly_vars = ["evapotranspiration","soil_moisture_0_to_1cm"]
    params = {
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "past_days": 2, "forecast_days": 16,
        "hourly": hourly_vars
    }
    resp = om_client.weather_api(FORECAST_URL, params=params)[0]
    h = resp.Hourly()
    times = pd.date_range(
        start=pd.to_datetime(h.Time(), unit="s", utc=True),
        end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=h.Interval()),
        inclusive="left"
    )
    df = pd.DataFrame({"time": times})
    for i in range(h.VariablesLength()):
        name = hourly_vars[i] if i < len(hourly_vars) else f"var_{i}"
        try:
            df[name] = h.Variables(i).ValuesAsNumpy()
        except Exception:
            df[name] = np.nan
    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date

    # calcula ET e SM para a data-alvo
    et24 = float(df[df["date"] == td]["evapotranspiration"].sum()) if "evapotranspiration" in df else np.nan
    sm6  = float(df[df["date"] == td].tail(6)["soil_moisture_0_to_1cm"].mean()) if "soil_moisture_0_to_1cm" in df else np.nan

    # normalizações (âncoras)
    sm_lo, sm_hi = 0.10, 0.45
    et_lo, et_hi = 1.0, 6.0

    def clamp01(x): return float(max(0.0, min(1.0, x)))
    def scale(x, lo, hi):
        s = (x - lo) / (hi - lo) if hi != lo else 0.0
        return clamp01(s)

    sm_norm = scale(sm6, sm_lo, sm_hi) if sm6 == sm6 else None
    et_scaled = scale(et24, et_lo, et_hi) if et24 == et24 else None
    if sm_norm is None or et_scaled is None:
        return {"sm_norm": None, "et_scaled": None, "dryness": None}
    dryness = 0.5*(1.0 - sm_norm) + 0.5*et_scaled
    return {"sm_norm": sm_norm, "et_scaled": et_scaled, "dryness": clamp01(dryness)}

# -------------- Endpoints --------------

@app.get("/health")
def health(): return {"status": "ok"}

@app.get("/v1/meta")
def meta():
    w = load_weights()
    dfH = try_load_hazard()
    return {
        "city": "canoas",
        "timezone": "America/Sao_Paulo",
        "hazard_date_min": dfH["date"].min().date().isoformat(),
        "hazard_date_max": dfH["date"].max().date().isoformat(),
        "thresholds": w["hazard_levels"],
        "weights": {"hazard": w["hazard_daily_weights"], "u": w["u_weights"]}
    }

@app.get("/v1/risk/by_bairro")
def risk_by_bairro(date_str: Optional[str] = Query(None, alias="date")):
    dfH = try_load_hazard(); dfU, _ = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date_str is None else dfH[dfH["date"].dt.date==pd.to_datetime(date_str).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d = df_sel["date"].iloc[0]
    df = dfU.copy()
    df["U"] = df.get("U_t", df.get("U_static", 0)).fillna(0)
    df["U_valid"] = df["U"] > 0
    df.loc[~df["U_valid"], ["Risk_score","Risk_level"]] = [np.nan,"no_data"]
    dfv = df[df["U_valid"]].copy()
    dfv["Fragilidade"] = 1 - dfv["U"]
    dfv["Risk_score"] = (H * dfv["Fragilidade"]).clip(0,1)
    dfv["Risk_level"] = dfv["Risk_score"].apply(lambda x: bucket_risk(x,thr))
    df.update(dfv)
    df["date"] = d.date().isoformat(); df["H_score"] = H
    return df[["bairro","date","H_score","U","U_valid","Risk_score","Risk_level"]].to_dict(orient="records")

@app.get("/v1/risk/by_bairro/top")
def risk_top(
    date: Optional[str] = Query(None, description="Data específica (YYYY-MM-DD)"),
    n: int = Query(5, description="Número de bairros a retornar")
):
    """
    Retorna os N piores bairros por score de risco na data informada.
    Ignora bairros com U==0 (no_data).
    """
    # passa a data diretamente para o cálculo
    rows = risk_by_bairro(date_str=date)
    df = pd.DataFrame(rows)

    # ignora bairros sem dado (U_valid=False ou Risk_score NaN)
    df = df[df.get("U_valid", True)]
    df = df.dropna(subset=["Risk_score"])

    # ordena e retorna top N
    df = df.sort_values("Risk_score", ascending=False).head(n)
    return df.to_dict(orient="records")

@app.get("/v1/geo/canoas/bairros_risk")
def geo_bairros_risk(date: Optional[str] = None):
    dfH = try_load_hazard(); dfU, gdfU = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date is None else dfH[dfH["date"].dt.date==pd.to_datetime(date).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d = df_sel["date"].iloc[0]
    df = dfU.copy(); df["U"] = df.get("U_t", df.get("U_static", 0)).fillna(0)
    df["U_valid"] = df["U"] > 0
    df.loc[~df["U_valid"], ["Risk_score","Risk_level"]] = [np.nan,"no_data"]
    dfv = df[df["U_valid"]].copy()
    dfv["Fragilidade"] = 1 - dfv["U"]
    dfv["Risk_score"] = (H * dfv["Fragilidade"]).clip(0,1)
    dfv["Risk_level"] = dfv["Risk_score"].apply(lambda x: bucket_risk(x,thr))
    df.update(dfv)
    gdf = gdfU.merge(df[["bairro","U","U_valid","Risk_score","Risk_level"]], on="bairro", how="left")
    gdf["date"] = d.date().isoformat()
    gj = json.loads(gdf.to_json())
    return gj

@app.get("/v1/bairros/detail")
def bairro_detail(bairro: str, date: Optional[str] = None):
    dfH = try_load_hazard(); dfU, _ = try_load_u(); thr = load_weights()["hazard_levels"]
    df_sel = dfH if date is None else dfH[dfH["date"].dt.date==pd.to_datetime(date).date()]
    if df_sel.empty: df_sel = dfH[dfH["date"]==dfH["date"].max()]
    H = float(df_sel["H_score"].iloc[0]); d = df_sel["date"].iloc[0]
    row = dfU[dfU["bairro"].astype(str)==str(bairro)].head(1)
    if row.empty: raise HTTPException(404,f"Bairro '{bairro}' não encontrado.")
    U = float(row.get("U_t", row.get("U_static", 0)).iloc[0])
    if U == 0 or pd.isna(U):
        return {"bairro": bairro, "date": d.date().isoformat(), "status": "no_data"}
    Frag = 1-U; Risk = float(np.clip(H*Frag,0,1)); level = bucket_risk(Risk,thr)
    return {"bairro": bairro, "date": d.date().isoformat(),"H_score":H,"U":U,
            "Fragilidade":Frag,"Risk_score":Risk,"Risk_level":level}

@app.get("/v1/filters")
def filters_schema():
    return {
        "risk_level": {"type":"enum","values":["green","yellow","red","no_data"]},
        "score_range": {"type":"range","field":"Risk_score","min":0,"max":1},
        "infra_subindices": {
            "u_cobertura":{"type":"range","min":0,"max":1},
            "u_micro":{"type":"range","min":0,"max":1},
            "u_macro":{"type":"range","min":0,"max":1},
            "u_permeabilidade":{"type":"range","min":0,"max":1}
        }
    }

