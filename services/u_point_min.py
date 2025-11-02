# u_point_min.py
# -*- coding: utf-8 -*-
import math, json, requests
import pandas as pd, numpy as np
from shapely.geometry import Point, LineString, Polygon, shape
from shapely.ops import unary_union
from pyproj import Transformer
import openmeteo_requests, requests_cache
from retry_requests import retry
from datetime import datetime, timezone, timedelta

# ---------- Configurações ----------
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]
TZ = "America/Sao_Paulo"

# Âncoras de normalização (edite se desejar)
ANCHORS = {
    "dens_pav_km_km2": (4.0, 18.0),
    "dreno_km_km2": (0.05, 0.50),
    "canal_km_km2": (0.10, 1.00),
    "frac_verde": (0.05, 0.30),
    "sm_clamp": (0.10, 0.45),
    "et_day": (1.0, 6.0)
}

# Pesos U_min (re-normalizamos se faltar termo)
WEIGHTS = {"perm": 0.40, "macro": 0.25, "cob": 0.20, "micro": 0.15}

# ---------- Utils ----------
def clamp01(x): return max(0.0, min(1.0, float(x)))

def scale_linear(x, lo, hi, invert=False):
    if hi == lo: return 0.0
    s = (x - lo) / (hi - lo)
    s = clamp01(s)
    return 1.0 - s if invert else s

def projectors(lat, lon):
    # UTM 22S (Canoas/RS) — bom para a região Sul (ajuste se for longe do RS)
    # fallback: WebMercator
    try:
        to_xy = Transformer.from_crs("EPSG:4326", "EPSG:31982", always_xy=True)
        to_ll = Transformer.from_crs("EPSG:31982", "EPSG:4326", always_xy=True)
        return to_xy, to_ll
    except Exception:
        to_xy = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        return to_xy, to_ll

def circle_buffer(lat, lon, radius_km):
    to_xy, to_ll = projectors(lat, lon)
    x, y = to_xy.transform(lon, lat)
    poly = Point(x, y).buffer(radius_km * 1000.0)  # metros
    return poly, to_ll

def overpass(query):
    for u in OVERPASS_URLS:
        try:
            r = requests.post(u, data={"data": query}, timeout=120)
            if r.ok:
                return r.json()
        except Exception:
            continue
    raise RuntimeError("Overpass API sem resposta (tente novamente mais tarde).")

def lines_length_km(elems, buf_poly, to_xy):
    total = 0.0
    for e in elems:
        if "geometry" not in e: continue
        coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(coords) < 2: continue
        xys = [to_xy.transform(x, y) for x, y in coords]
        line = LineString(xys)
        inter = line.intersection(buf_poly)
        total += inter.length if not inter.is_empty else 0.0
    return total / 1000.0  # km

def polygons_area_km2(elems, buf_poly, to_xy):
    total = 0.0
    for e in elems:
        if "geometry" not in e: continue
        coords = [(p["lon"], p["lat"]) for p in e["geometry"]]
        if len(coords) < 3: continue
        xys = [to_xy.transform(x, y) for x, y in coords]
        poly = Polygon(xys)
        if not poly.is_valid: poly = poly.buffer(0)
        inter = poly.intersection(buf_poly)
        total += inter.area if not inter.is_empty else 0.0
    return total / 1e6  # km²

# ---------- OSM (Overpass) ----------
def fetch_osm_metrics(lat, lon, radius_km):
    buf_poly, to_ll = circle_buffer(lat, lon, radius_km)
    to_xy, _ = projectors(lat, lon)  # inverso já temos
    # query com around:R (metros)
    R = int(radius_km * 1000)
    q = f"""
    [out:json][timeout:60];
    (
      // vias pavimentadas (proxy de cobertura)
      way(around:{R},{lat},{lon})["highway"]["surface"~"asphalt|paved|concrete"];
      // drenos/valas (micro)
      way(around:{R},{lat},{lon})["waterway"~"drain|ditch"];
      // canais (macro)
      way(around:{R},{lat},{lon})["waterway"="canal"];
      // verdes (permeabilidade): landuse/natural/leisure
      way(around:{R},{lat},{lon})["landuse"~"grass|forest|meadow|recreation_ground|park"];
      way(around:{R},{lat},{lon})["natural"~"wood|scrub|grassland|heath|wetland"];
      way(around:{R},{lat},{lon})["leisure"="park"];
      // bombas (macro)
      node(around:{R},{lat},{lon})["man_made"="pumping_station"];
    );
    out tags geom;
    """
    data = overpass(q)
    elements = data.get("elements", [])

    # separa por tipo/tag
    paved = [e for e in elements if e.get("type") == "way" and e.get("tags", {}).get("highway") and e.get("tags", {}).get("surface")]
    drain  = [e for e in elements if e.get("type") == "way" and e.get("tags", {}).get("waterway") in ("drain", "ditch")]
    canal  = [e for e in elements if e.get("type") == "way" and e.get("tags", {}).get("waterway") == "canal"]
    greens = [e for e in elements if e.get("type") == "way" and (
        e.get("tags", {}).get("landuse") in ("grass","forest","meadow","recreation_ground","park")
        or e.get("tags", {}).get("natural") in ("wood","scrub","grassland","heath","wetland")
        or e.get("tags", {}).get("leisure") == "park"
    )]
    pumps = [e for e in elements if e.get("type") == "node" and e.get("tags", {}).get("man_made") == "pumping_station"]

    # projeta buffer para o CRS métrico
    lon0, lat0 = lon, lat
    to_xy, _ = projectors(lat0, lon0)
    x0, y0 = to_xy.transform(lon0, lat0)
    buf_xy = Point(x0, y0).buffer(radius_km * 1000.0)

    # métricas
    paved_km  = lines_length_km(paved, buf_xy, to_xy)
    drain_km  = lines_length_km(drain, buf_xy, to_xy)
    canal_km  = lines_length_km(canal, buf_xy, to_xy)
    green_km2 = polygons_area_km2(greens, buf_xy, to_xy)
    area_km2  = buf_xy.area / 1e6
    pumps_n   = len(pumps)

    return {
        "area_km2": area_km2,
        "paved_km": paved_km,
        "drain_km": drain_km,
        "canal_km": canal_km,
        "green_km2": green_km2,
        "pumps_n": pumps_n
    }

# ---------- Open-Meteo (dinâmica) ----------
cache_session = requests_cache.CachedSession('.cache', expire_after=1800)
retry_session = retry(cache_session, retries=3, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_dryness(lat, lon):
    hourly_vars = ["evapotranspiration","soil_moisture_0_to_1cm"]
    params = {
        "latitude": lat, "longitude": lon, "timezone": TZ,
        "forecast_days": 1, "past_days": 2, "hourly": hourly_vars
    }
    resp = om.weather_api(FORECAST_URL, params=params)[0]
    h = resp.Hourly()
    times = pd.date_range(start=pd.to_datetime(h.Time(), unit="s", utc=True),
                          end=pd.to_datetime(h.TimeEnd(), unit="s", utc=True),
                          freq=pd.Timedelta(seconds=h.Interval()), inclusive="left")
    df = pd.DataFrame({"time": times})
    for i in range(h.VariablesLength()):
        name = hourly_vars[i] if i < len(hourly_vars) else f"var_{i}"
        try:
            df[name] = h.Variables(i).ValuesAsNumpy()
        except Exception:
            df[name] = np.nan

    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date

    # ET do último dia completo
    last_day = df["date"].iloc[-1]
    et24 = float(df[df["date"] == last_day]["evapotranspiration"].sum()) if "evapotranspiration" in df else np.nan
    # SM média das últimas 6h
    sm6  = float(df.tail(6)["soil_moisture_0_to_1cm"].mean()) if "soil_moisture_0_to_1cm" in df else np.nan

    # normalizações
    sm_lo, sm_hi = ANCHORS["sm_clamp"]
    et_lo, et_hi = ANCHORS["et_day"]
    sm_norm = scale_linear(sm6, sm_lo, sm_hi)           # 0..1 (alto = mais úmido)
    et_scaled = scale_linear(et24, et_lo, et_hi)        # 0..1 (alto = mais seco)
    dryness = 0.5*(1.0 - sm_norm) + 0.5*et_scaled       # alto = mais seco

    return {
        "et24_mm": et24 if et24 == et24 else None,
        "sm6_m3m3": sm6 if sm6 == sm6 else None,
        "sm_norm": clamp01(sm_norm if sm_norm == sm_norm else 0.5),
        "et_scaled": clamp01(et_scaled if et_scaled == et_scaled else 0.5),
        "dryness": clamp01(dryness if dryness == dryness else 0.5)
    }

# ---------- U point-based ----------
def compute_U_point(lat, lon, radius_km=2.0, delta=0.10):
    osm = fetch_osm_metrics(lat, lon, radius_km)
    area = max(1e-6, osm["area_km2"])

    dens_pav = osm["paved_km"] / area
    dreno_km2 = osm["drain_km"] / area
    canal_km2 = osm["canal_km"] / area
    frac_verde = min(1.0, osm["green_km2"] / area)
    has_pump = 1.0 if osm["pumps_n"] > 0 else 0.0

    # normalizações
    u_cob  = scale_linear(dens_pav, *ANCHORS["dens_pav_km_km2"])
    u_micro= scale_linear(dreno_km2, *ANCHORS["dreno_km_km2"])
    u_macro= clamp01(0.5*has_pump + 0.5*scale_linear(canal_km2, *ANCHORS["canal_km_km2"]))
    u_perm = scale_linear(frac_verde, *ANCHORS["frac_verde"])

    # pesos (re-normaliza se faltar algo)
    weights = WEIGHTS.copy()
    avail = {"perm": True, "macro": True, "cob": True, "micro": True}
    s = sum(weights[k] for k, ok in avail.items() if ok)
    weights = {k: (weights[k]/s if avail[k] else 0.0) for k in weights}

    U_static = clamp01(weights["perm"]*u_perm + weights["macro"]*u_macro +
                       weights["cob"]*u_cob + weights["micro"]*u_micro)

    # ajuste dinâmico com Open-Meteo
    dyn = fetch_dryness(lat, lon)
    dryness = dyn["dryness"]
    U_t = clamp01(U_static + delta*(dryness - 0.5))
    frag_t = clamp01(1.0 - U_t)

    return {
        "inputs": {
            "lat": lat, "lon": lon, "radius_km": radius_km,
            "area_km2": round(area, 4),
            "paved_km": round(osm["paved_km"], 3),
            "drain_km": round(osm["drain_km"], 3),
            "canal_km": round(osm["canal_km"], 3),
            "green_km2": round(osm["green_km2"], 3),
            "pumps_n": int(osm["pumps_n"])
        },
        "densities": {
            "dens_pav_km_km2": round(dens_pav, 3),
            "dreno_km_km2": round(dreno_km2, 3),
            "canal_km_km2": round(canal_km2, 3),
            "frac_verde": round(frac_verde, 3)
        },
        "subindices": {
            "u_cobertura": round(u_cob, 3),
            "u_micro": round(u_micro, 3),
            "u_macro": round(u_macro, 3),
            "u_permeabilidade": round(u_perm, 3)
        },
        "weights": weights,
        "U_static": round(U_static, 3),
        "dynamic": {
            "sm_norm": dyn["sm_norm"], "et_scaled": dyn["et_scaled"],
            "dryness": dyn["dryness"], "delta": delta
        },
        "U_t": round(U_t, 3),
        "Fragilidade_t": round(frag_t, 3)
    }

if __name__ == "__main__":
    # Exemplo para Canoas (centro): lat=-29.918, lon=-51.185
    lat, lon = -29.918, -51.185
    out = compute_U_point(lat, lon, radius_km=2.0, delta=0.10)
    print(json.dumps(out, ensure_ascii=False, indent=2))
