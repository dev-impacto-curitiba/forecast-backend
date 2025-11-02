# ================================
# Open-Meteo: Flood + Hourly Weather (RS)
# Cria dataset diário para prever enchentes
# ================================
import openmeteo_requests
import pandas as pd
import numpy as np
import requests_cache
from retry_requests import retry
from datetime import date

# -----------------
# Cliente com cache/retry
# -----------------
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)

FLOOD_URL   = "https://flood-api.open-meteo.com/v1/flood"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"  # histórico horário estável
TZ = "America/Sao_Paulo"

# -----------------
# Escolha sua área/tempo
# -----------------
lat, lon = -30.03, -51.22  # Porto Alegre, RS (ajuste conforme sua bacia/ponto de interesse)
start_date = "2024-01-01"
end_date   = str(date.today())

# -----------------
# Variáveis HORÁRIAS (Open-Meteo)
# (mapeadas para os nomes oficiais da API)
# -----------------
hourly_vars = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "apparent_temperature",
    "precipitation_probability",
    "precipitation",
    "rain",
    "showers",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "visibility",
    "evapotranspiration",
    "et0_fao_evapotranspiration",
    "vapor_pressure_deficit",
    "wind_speed_10m",
    "wind_speed_80m",
    "wind_speed_120m",
    "wind_speed_180m",
    "wind_direction_120m",
    "wind_direction_180m",
    "wind_gusts_10m",
    "temperature_80m",
    "temperature_120m",
    "temperature_180m",
    "soil_temperature_0cm",
    "soil_temperature_6cm",
    "soil_temperature_18cm",
    "soil_temperature_54cm",
    "soil_moisture_0_1cm",
    "soil_moisture_1_3cm",
    "soil_moisture_3_9cm",
    "soil_moisture_9_27cm",
    "soil_moisture_27_81cm",
]

def fetch_hourly_df(lat, lon, start_date, end_date, variables):
    """Baixa dados HORÁRIOS do Archive API e retorna DataFrame com timezone em UTC (coluna 'time')."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TZ,
        "hourly": variables
    }
    responses = om.weather_api(ARCHIVE_URL, params=params)
    resp = responses[0]
    hourly = resp.Hourly()

    # Monta tabela temporal
    time_index = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )
    df = pd.DataFrame({"time": time_index})

    # Preenche colunas dinamicamente (nem todas podem vir)
    n = hourly.VariablesLength()
    # A ordem na resposta deve bater com a ordem pedida em "variables"
    for i in range(n):
        try:
            vals = hourly.Variables(i).ValuesAsNumpy()
            if i < len(variables):
                name = variables[i]
            else:
                name = f"var_{i}"
            df[name] = vals
        except Exception:
            # ignora variável faltante/sem dados
            pass

    # Normaliza para timezone local (opcional)
    df["time_local"] = df["time"].dt.tz_convert(TZ)
    df["date"] = df["time_local"].dt.date
    return df

# -----------------
# Variáveis DIÁRIAS (Flood API)
# -----------------
# Observação: nem todas as variáveis “extras” existem em todos os locais; manter conjunto essencial.
flood_daily_vars = [
    "river_discharge",
    "river_discharge_mean",
    "river_discharge_median",
    "river_discharge_min",
    "river_discharge_max",
    "river_discharge_p25",
    "river_discharge_p75"
    # Se a sua área suportar, você pode tentar:
    # "river_discharge_anomaly",
    # "river_discharge_climatology",
    # "river_discharge_return_period"
]

def fetch_flood_daily_df(lat, lon, start_date, end_date, variables):
    """Baixa dados DIÁRIOS de descarga (Flood API) e retorna DataFrame diário."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TZ,
        "daily": variables
    }
    responses = om.weather_api(FLOOD_URL, params=params)
    resp = responses[0]
    daily = resp.Daily()

    # eixo temporal
    date_index = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left"
    ).tz_convert(TZ).date

    out = pd.DataFrame({"date": date_index})
    # Prenche cada variável que chegou
    for i in range(daily.VariablesLength()):
        col = variables[i] if i < len(variables) else f"flood_var_{i}"
        try:
            out[col] = daily.Variables(i).ValuesAsNumpy()
        except Exception:
            pass

    out["latitude"] = resp.Latitude()
    out["longitude"] = resp.Longitude()
    return out

# -----------------
# Agregação: de horário -> diário (para casar com Flood)
# -----------------
def agg_hourly_to_daily(weather_hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Agrupa por 'date' com regras sensatas: soma de precip, ET/ET0; máx de rajada; médias do restante."""
    df = weather_hourly_df.copy()

    # mapeia regras de agregação (se a coluna existir)
    sum_cols = [c for c in df.columns if c in {
        "precipitation", "rain", "showers",
        "evapotranspiration", "et0_fao_evapotranspiration"
    }]
    max_cols = [c for c in df.columns if c in {"wind_gusts_10m"}]
    mean_cols = [c for c in df.columns
                 if c not in (["time","time_local","date"] + sum_cols + max_cols)
                 and c != "weather_code"]  # weather_code pode ser tratado à parte

    agg = {}
    for c in sum_cols:  agg[c] = "sum"
    for c in max_cols:  agg[c] = "max"
    for c in mean_cols: agg[c] = "mean"

    # Para weather_code, pega o modo (mais frequente) do dia, se existir
    if "weather_code" in df.columns:
        def mode_or_nan(x):
            m = x.mode()
            return m.iloc[0] if not m.empty else np.nan
        agg["weather_code"] = mode_or_nan

    daily = df.groupby("date", as_index=False).agg(agg)
    return daily

# -----------------
# Execução
# -----------------
if __name__ == "__main__":
    # 1) Meteorologia/solo horário -> diário
    wx_hourly = fetch_hourly_df(lat, lon, start_date, end_date, hourly_vars)
    wx_daily  = agg_hourly_to_daily(wx_hourly)
    wx_daily.to_csv("rs_weather_daily.csv", index=False)

    # 2) Hidrologia diária (Flood)
    flood_daily = fetch_flood_daily_df(lat, lon, start_date, end_date, flood_daily_vars)
    flood_daily.to_csv("rs_flood_daily.csv", index=False)

    # 3) Merge (left join nas datas da hidrologia)
    merged = flood_daily.merge(wx_daily, on="date", how="left")
    merged.to_csv("rs_flood_weather_merged.csv", index=False)

    print(f"Salvos: rs_weather_daily.csv ({len(wx_daily)} linhas), "
          f"rs_flood_daily.csv ({len(flood_daily)} linhas), "
          f"rs_flood_weather_merged.csv ({len(merged)} linhas))")
