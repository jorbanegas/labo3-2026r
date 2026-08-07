# -*- coding: utf-8 -*-
"""Regresion lineal a nivel producto — la version de la catedra, portable y con palancas.

Reproduce src/Estadistica/z403_RegresionLineal.ipynb (escrito para Colab) para que corra
en la VM y su prediccion entre en el flujo de mezclas.

Por que importa: este modelo de DOCE variables saca 0.231 en Kaggle, mejor que el
pipeline de LightGBM con 600 features (0.256 mezclado). Sus diferencias estructurales:

  1. Trabaja a nivel PRODUCTO: suma sobre todos los clientes de entrada.
  2. Se queda solo con los 780 productos a entregar.
  3. Usa 12 lags de tn y nada mas. Sin deltas, rachas ni agregados por categoria.
  4. Entrena con UNA fila por producto, del periodo 201812, para predecir desde 201912.
     Mismo mes del calendario: aprende "como es febrero visto desde diciembre", en vez
     de promediar todos los regimenes estacionales del anio.

El punto 4 es probablemente el que mas explica la diferencia con el LightGBM, que
entrena con todos los meses mezclados y ni siquiera sabe en que mes esta parado.

    python linreg.py                        # la version de la catedra
    python linreg.py --productos todos      # entrenar con los 780, no con los 182
    python linreg.py --meses 201812,201712  # dos diciembres de entrenamiento
    python linreg.py --lags 6

Escribe exp/<nombre>/submission_202002.csv, listo para mezclar.py y para subir.
"""
import argparse
import io
import sys
from pathlib import Path

for _s in ("stdout", "stderr"):
    _f = getattr(sys, _s)
    if getattr(_f, "encoding", "").lower() not in ("utf-8", "utf8") and hasattr(_f, "buffer"):
        setattr(sys, _s, io.TextIOWrapper(_f.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True))

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import labo3 as L  # noqa: E402
import polars as pl  # noqa: E402

# La lista que el notebook de la catedra llama "la salsa magica": 182 productos
# elegidos a mano sobre los que se ajusta la regresion.
PRODUCTOS_MAGICOS = [
    20001, 20002, 20005, 20013, 20033, 20037, 20038, 20043, 20044, 20045, 20046, 20052,
    20055, 20058, 20059, 20069, 20070, 20072, 20073, 20075, 20080, 20091, 20094, 20099,
    20107, 20114, 20120, 20132, 20137, 20139, 20142, 20144, 20146, 20148, 20151, 20153,
    20157, 20158, 20161, 20162, 20166, 20167, 20189, 20198, 20201, 20202, 20203, 20208,
    20226, 20228, 20231, 20233, 20253, 20254, 20256, 20269, 20270, 20271, 20275, 20276,
    20277, 20278, 20288, 20298, 20315, 20317, 20320, 20322, 20335, 20337, 20338, 20344,
    20348, 20350, 20353, 20359, 20385, 20390, 20398, 20402, 20403, 20406, 20411, 20416,
    20417, 20418, 20419, 20421, 20422, 20424, 20428, 20429, 20443, 20456, 20466, 20469,
    20479, 20497, 20500, 20509, 20514, 20517, 20524, 20532, 20549, 20551, 20560, 20561,
    20565, 20568, 20579, 20583, 20585, 20586, 20589, 20599, 20606, 20614, 20624, 20632,
    20642, 20646, 20653, 20655, 20657, 20660, 20661, 20663, 20666, 20677, 20680, 20684,
    20696, 20699, 20713, 20737, 20744, 20745, 20765, 20768, 20773, 20777, 20786, 20789,
    20800, 20807, 20812, 20818, 20830, 20832, 20838, 20847, 20855, 20863, 20864, 20882,
    20883, 20906, 20913, 20914, 20919, 20922, 20925, 20937, 20945, 20956, 20961, 20965,
    20970, 20976, 20986, 20996, 21016, 21038, 21048, 21049, 21077, 21080, 21088, 21118,
    21170, 21200,
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--productos", choices=["magicos", "todos"], default="magicos",
                    help="entrenar con los 182 elegidos a mano o con los 780")
    ap.add_argument("--meses", default="201812",
                    help="periodos de entrenamiento separados por coma (origenes)")
    ap.add_argument("--lags", type=int, default=12, help="cuantos lags de tn usar")
    ap.add_argument("--horizonte", type=int, default=2)
    ap.add_argument("--origen", type=int, default=201912,
                    help="periodo desde el que se predice")
    ap.add_argument("--nombre", default=None, help="carpeta de salida dentro de exp/")
    args = ap.parse_args()

    import statsmodels.api as sm

    meses_tr = [int(x) for x in args.meses.split(",")]

    # ── 1. Ventas por producto y periodo (se colapsa la dimension cliente) ──
    ventas = (pl.read_csv(L.DIR_RAW / "sell-in.txt.gz", separator="\t")
                .group_by("product_id", "periodo")
                .agg(pl.col("tn").sum().alias("tn")))

    oficial = pl.read_csv(L.DIR_RAW / "product_id_apredecir201912.txt",
                          separator="\t").select("product_id").unique()
    ventas = (ventas.join(oficial, on="product_id", how="inner")
                    .sort(["product_id", "periodo"]))
    print(f"{ventas['product_id'].n_unique()} productos de la lista oficial, "
          f"{ventas.height:,} filas producto-mes")

    # ── 2. Lags. shift(-h) trae el valor de h meses ADELANTE: esa es la clase ──
    lags = [-args.horizonte, *range(args.lags)]
    tabla = tabla_lags(ventas, lags, args.horizonte)
    campos = [f"tn_{k}" for k in range(args.lags)]

    # ── 3. Entrenamiento ────────────────────────────────────────────────────
    filtro = pl.col("periodo").is_in(meses_tr)
    if args.productos == "magicos":
        filtro = filtro & pl.col("product_id").is_in(PRODUCTOS_MAGICOS)
    dtrain = tabla.filter(filtro).drop_nulls(campos + ["clase"])
    if dtrain.is_empty():
        raise SystemExit(f"Sin filas de entrenamiento para {meses_tr}. "
                         f"Con {args.lags} lags hace falta historia suficiente.")
    print(f"Entrenamiento: {dtrain.height} filas  (meses {meses_tr}, "
          f"productos '{args.productos}')")

    X = sm.add_constant(dtrain.select(campos).to_pandas())
    modelo = sm.OLS(dtrain["clase"].to_pandas(), X).fit()
    print(f"R2 = {modelo.rsquared:.4f}   R2 ajustado = {modelo.rsquared_adj:.4f}")

    # ── 4. Prediccion desde el origen, solo donde hay historia completa ──────
    dfuture = tabla.filter(pl.col("periodo") == args.origen).drop_nulls(campos)
    pred = modelo.predict(sm.add_constant(dfuture.select(campos).to_pandas()))
    reg = dfuture.select("product_id").with_columns(pl.Series("tn_pred", pred.values))
    print(f"Prediccion por regresion: {reg.height} productos")

    # ── 5. Respaldo para los que no tienen historia completa ────────────────
    # Promedio de los ultimos 12 meses. Sin esto faltarian productos en la entrega.
    ini = (args.origen // 100 - 1) * 100 + args.origen % 100 + 1
    promedio = (ventas.filter(pl.col("periodo").is_between(ini, args.origen))
                      .group_by("product_id").agg(pl.col("tn").mean().alias("tn")))

    final = (promedio.join(reg, on="product_id", how="left")
                     .with_columns(pl.coalesce(["tn_pred", "tn"]).alias("tn"))
                     .select(["product_id", "tn"]))
    print(f"Respaldo por promedio  : {max(0, final.height - reg.height)} productos")

    # ── 6. La entrega tiene que ser EXACTAMENTE la lista oficial ────────────
    final = (oficial.join(final, on="product_id", how="left")
                    .with_columns(pl.col("tn").fill_null(0.0).clip(0.0))
                    .sort("product_id"))
    if final.height != oficial.height:
        raise SystemExit(f"{final.height} filas contra {oficial.height} oficiales")

    nombre = args.nombre or (f"linreg_{args.productos}_{args.lags}lags_"
                             + "-".join(str(m) for m in meses_tr))
    dst = L.RUTA_EXP / nombre
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "submission_202002.csv"
    final.write_csv(path)

    print(f"\n{final.height} productos   tn total {final['tn'].sum():,.0f}")
    print(f"Guardado: {path}")
    print("\nPara subirlo:\n")
    print(f"kaggle competitions submit -c labo-iii-2026-rosario -f {path} -m \"{nombre}\"")


def tabla_lags(ventas: "pl.DataFrame", lags: list, horizonte: int) -> "pl.DataFrame":
    """Una fila por (producto, periodo) con tn_0..tn_N y la clase a t+horizonte."""
    return (ventas.with_columns([
                pl.col("tn").shift(lag).over("product_id").alias(f"tn_{lag}")
                for lag in lags])
                  .rename({f"tn_{-horizonte}": "clase"}))


if __name__ == "__main__":
    main()
