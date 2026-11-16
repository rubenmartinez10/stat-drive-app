"""Reentrena el modelo con viajes_base.csv + las respuestas del Google Form (si FORM_CSV_URL existe)."""
import os, json, numpy as np, pandas as pd
from datetime import datetime, timezone

COLS = ["distancia_km", "hora_salida", "incidencias_n", "es_regreso", "es_otra"]
NEEDED = ["tipo", "hora_salida", "distancia_km", "incidencias_n", "real_min"]

def hora_decimal(v):
    if isinstance(v, (int, float)): return float(v)
    t = pd.to_datetime(str(v), errors="coerce")
    return np.nan if pd.isna(t) else t.hour + t.minute / 60

def cargar():
    base = pd.read_csv("viajes_base.csv")[NEEDED].astype(
        {"hora_salida": float, "distancia_km": float, "incidencias_n": float, "real_min": float})
    url = os.environ.get("FORM_CSV_URL")
    if not url:
        return base, 0
    f = pd.read_csv(url)
    f.columns = [str(c).strip().lower() for c in f.columns]
    faltan = [c for c in NEEDED if c not in f.columns]
    if faltan:
        print(f"ADVERTENCIA: faltan columnas {faltan} en la hoja. Columnas encontradas: {list(f.columns)}")
        print("Se entrena solo con los datos base.")
        return base, 0
    f = f[NEEDED].copy()
    f["hora_salida"] = f["hora_salida"].map(hora_decimal)
    for c in ["distancia_km", "incidencias_n", "real_min"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["tipo"] = f["tipo"].astype(str).str.strip().str.lower()
    f = f.dropna()
    ok = (f["tipo"].isin(["ida", "regreso", "otra"]) & f["distancia_km"].between(0.5, 150)
          & f["real_min"].between(1, 400) & f["incidencias_n"].between(0, 10)
          & f["hora_salida"].between(0, 24))
    print(f"Form: {len(f)} filas completas, {int(ok.sum())} válidas")
    f = f[ok].astype(float, errors="ignore")
    if f.empty:
        return base, 0
    out = pd.concat([base, f], ignore_index=True)
    return out.astype({"hora_salida": float, "distancia_km": float, "incidencias_n": float, "real_min": float}), len(f)

def X_de(d):
    return np.column_stack([np.ones(len(d)), d["distancia_km"], d["hora_salida"], d["incidencias_n"],
                            (d["tipo"] == "regreso").astype(int), (d["tipo"] == "otra").astype(int)]).astype(float)

df, n_form = cargar()
X, y = X_de(df), df["real_min"].values.astype(float)
coef = np.linalg.lstsq(X, y, rcond=None)[0]

# error con validación cruzada de 10 pliegues
rng = np.random.default_rng(42); idx = rng.permutation(len(df)); errs = np.zeros(len(df))
for k in range(10):
    te = idx[k::10]; tr = np.setdiff1d(idx, te)
    c = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
    errs[te] = y[te] - X[te] @ c

modelo = {"intercepto": coef[0], **dict(zip(COLS, coef[1:])),
          "rmse_cv": float(np.sqrt((errs ** 2).mean())), "mae_cv": float(np.abs(errs).mean()),
          "n": len(df), "n_form": n_form,
          "dist_min": float(df["distancia_km"].min()), "dist_max": float(df["distancia_km"].max()),
          "actualizado": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}
json.dump({k: (round(v, 4) if isinstance(v, float) else v) for k, v in modelo.items()},
          open("model.json", "w"), indent=2)
print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in modelo.items()}, indent=2))