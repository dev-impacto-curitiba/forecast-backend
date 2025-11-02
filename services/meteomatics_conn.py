import datetime as dt
import pandas as pd
import numpy as np
import meteomatics.api as api
from scipy.stats import percentileofscore
from config import USERNAME, PASSWORD


# -----------------------
# Helpers de tempo (UTC)
# -----------------------
def to_utc(x: dt.datetime) -> dt.datetime:
    """Garante que x é timezone-aware em UTC."""
    if x.tzinfo is None:
        return x.replace(tzinfo=dt.UTC)
    return x.astimezone(dt.UTC)


# -----------------------
# Meteomatics helpers
# -----------------------
def get_precip_data(lat, lon, start, end, interval, username, password, model="mix"):
    """Baixa precipitação 1h em mm e retorna DataFrame com colunas ['time','precip_mm'] sempre em UTC."""
    # Garante UTC nos limites de consulta
    start = to_utc(start)
    end = to_utc(end)

    df = api.query_time_series(
        [(lat, lon)],
        start,
        end,
        interval,
        ["precip_1h:mm"],
        username,
        password,
        model=model
    ).reset_index()

    # Detecta coluna de tempo e precipitação de forma robusta
    time_col = "validdate" if "validdate" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        raise ValueError(f"Não encontrei coluna de tempo em {list(df.columns)}")

    precip_col = None
    for c in df.columns:
        c_low = str(c).lower()
        if "precip" in c_low and "1h" in c_low:
            precip_col = c
            break
    # fallback: primeira coluna que contenha "precip"
    if precip_col is None:
        for c in df.columns:
            if "precip" in str(c).lower():
                precip_col = c
                break

    if precip_col is None:
        raise ValueError(f"Não encontrei coluna de precipitação em {list(df.columns)}")

    # Padroniza nomes e garante timezone UTC-aware no pandas
    df.rename(columns={time_col: "time", precip_col: "precip_mm"}, inplace=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)

    # Se por acaso vier alguma unidade na coluna, apenas garante tipo numérico
    df["precip_mm"] = pd.to_numeric(df["precip_mm"], errors="coerce").fillna(0.0)

    return df[["time", "precip_mm"]]


def rolling_sum(series: pd.Series, hours: int) -> pd.Series:
    """Soma móvel com janela de 'hours' horas (freq horária)."""
    return series.rolling(hours, min_periods=1).sum()


def get_percentile(value: float, distribution: pd.Series) -> float:
    """Percentil no intervalo [0,1]."""
    if len(distribution) == 0:
        return 0.0
    return percentileofscore(distribution, value) / 100.0


# -----------------------
# Núcleo do cálculo (modo conta gratuita)
# -----------------------
def compute_H(lat, lon, username, password, t0, horizons,
              w6h=0.5, w72h=0.5, output_csv="risk_fluvial_basic.csv"):
    """Versão compatível com a conta gratuita: antecedente e baseline de 24h, previsões dentro de poucos dias."""
    interval = dt.timedelta(hours=1)

    # Garante t0 UTC-aware
    t0 = to_utc(t0)

    # Forecast (t0 -> t0 + maxΔ)
    forecast_end = t0 + dt.timedelta(hours=max(horizons))
    df_fore = get_precip_data(lat, lon, t0, forecast_end, interval, username, password)

    # Antecedente: últimas 24h
    antecedent_start = t0 - dt.timedelta(hours=24)
    df_ant = get_precip_data(lat, lon, antecedent_start, t0, interval, username, password)
    ant24h_mm = float(df_ant["precip_mm"].sum())

    # Baseline limitado: últimas 24h
    df_base = df_ant.copy()
    base_6h = rolling_sum(df_base["precip_mm"], 6).dropna()
    base_24h = rolling_sum(df_base["precip_mm"], 24).dropna()
    ant24h_pct = get_percentile(ant24h_mm, base_24h)

    results = []
    for Δ in horizons:
        # Limite temporal (UTC-aware)
        limit = t0 + dt.timedelta(hours=Δ)
        # df_fore["time"] é UTC-aware; limit também -> comparação válida
        df_h = df_fore[df_fore["time"] <= limit]

        if df_h.empty:
            peak6h_mm = 0.0
            peak6h_pct = 0.0
        else:
            peaks_6h = rolling_sum(df_h["precip_mm"], 6)
            peak6h_mm = float(peaks_6h.max() if not peaks_6h.empty else 0.0)
            peak6h_pct = get_percentile(peak6h_mm, base_6h)

        H = w6h * peak6h_pct + w72h * ant24h_pct

        results.append({
            "horizon_h": int(Δ),
            "peak6h_mm_forecast": round(peak6h_mm, 2),
            "peak6h_pct": round(peak6h_pct, 3),
            "ant24h_mm_observed": round(ant24h_mm, 2),
            "ant24h_pct": round(ant24h_pct, 3),
            "H_score": round(H, 3)
        })

    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Arquivo salvo: {output_csv}")
    print(df_out)
    return df_out


def main():
    # Usa datetime-aware em UTC (sem DeprecationWarning)
    t0 = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0)

    # Sua coordenada de exemplo (RS)
    lat, lon = -29.918, -51.185

    compute_H(
        lat=lat,
        lon=lon,
        username=USERNAME,
        password=PASSWORD,
        t0=t0,
        horizons=[6, 24, 48, 72],
        w6h=0.5,
        w72h=0.5,
        output_csv="risk_fluvial_basic.csv"
    )


if __name__ == "__main__":
    main()
