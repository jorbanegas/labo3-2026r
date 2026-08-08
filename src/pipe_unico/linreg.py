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
    python linreg.py --reg ridge --alpha 10 # encoger los coeficientes

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
    ap.add_argument("--productos",
                    choices=["magicos", "todos", "topvol", "estables"],
                    default="magicos",
                    help="que productos entran al ajuste: los 182 a mano, todos, "
                         "los de mayor volumen, o los de serie mas regular")
    ap.add_argument("--n", type=int, default=182,
                    help="cuantos productos con --productos topvol o estables")
    ap.add_argument("--meses", default="201812",
                    help="periodos de entrenamiento separados por coma (origenes)")
    ap.add_argument("--lags", type=int, default=12, help="cuantos lags de tn usar")
    ap.add_argument("--horizonte", type=int, default=2)
    ap.add_argument("--origen", type=int, default=201912,
                    help="periodo desde el que se predice")
    ap.add_argument("--reg", choices=["ols", "ridge", "lasso"], default="ols",
                    help="ols = la version de la catedra (sin regularizacion)")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="fuerza de la regularizacion (solo con --reg ridge|lasso)")
    ap.add_argument("--nombre", default=None, help="carpeta de salida dentro de exp/")
    args = ap.parse_args()

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
    # ── Que productos entran al ajuste ──────────────────────────────────────
    # Vale 0.04 en Kaggle: los 182 elegidos a mano dan 0.231 y los 780 dan 0.271.
    # La razon probable es que el WAPE es suma|error|/suma|real|, asi que lo dominan
    # los productos de mayor volumen; y OLS es sensible a outliers, de modo que meter
    # series erraticas al ajuste corre los coeficientes en favor de productos que casi
    # no pesan en el resultado. 'topvol' y 'estables' son versiones explicitas de esa
    # misma idea, sin depender de una lista sin criterio documentado.
    elegidos = seleccionar_productos(ventas, args.productos, args.n, args.origen)
    filtro = pl.col("periodo").is_in(meses_tr)
    if elegidos is not None:
        filtro = filtro & pl.col("product_id").is_in(elegidos)
        print(f"Productos para el ajuste: {len(elegidos)} ('{args.productos}')")
    dtrain = tabla.filter(filtro).drop_nulls(campos + ["clase"])
    if dtrain.is_empty():
        raise SystemExit(f"Sin filas de entrenamiento para {meses_tr}. "
                         f"Con {args.lags} lags hace falta historia suficiente.")
    print(f"Entrenamiento: {dtrain.height} filas  (meses {meses_tr}, "
          f"productos '{args.productos}')")

    predecir = ajustar(dtrain.select(campos).to_pandas(), dtrain["clase"].to_pandas(),
                       args.reg, args.alpha)

    # ── 4. Prediccion desde el origen, solo donde hay historia completa ──────
    dfuture = tabla.filter(pl.col("periodo") == args.origen).drop_nulls(campos)
    pred = predecir(dfuture.select(campos).to_pandas())
    reg = dfuture.select("product_id").with_columns(pl.Series("tn_pred", pred))
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

    _suf = f"{args.n}" if args.productos in ("topvol","estables") else ""
    # El tag de regularizacion se omite con 'ols' a proposito: asi la corrida de la
    # catedra conserva el nombre que ya tiene en exp/ (y en las mezclas) y no aparece
    # duplicada. Con ridge/lasso el alpha entra en el nombre, porque dos alphas
    # distintos son dos modelos distintos y no pueden pisarse la carpeta.
    _tag_reg = "" if args.reg == "ols" else f"_{args.reg}{args.alpha:g}"
    nombre = args.nombre or (f"linreg_{args.productos}{_suf}_{args.lags}lags_"
                             + "-".join(str(m) for m in meses_tr) + _tag_reg)
    dst = L.RUTA_EXP / nombre
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "submission_202002.csv"
    final.write_csv(path)

    print(f"\n{final.height} productos   tn total {final['tn'].sum():,.0f}")
    print(f"Guardado: {path}")
    print("\nPara subirlo:\n")
    print(f"kaggle competitions submit -c labo-iii-2026-rosario -f {path} -m \"{nombre}\"")


def ajustar(X, y, metodo: str, alpha: float):
    """Ajusta el modelo y devuelve una funcion que predice sobre un DataFrame igual.

    'ols' es la version de la catedra y no cambia en nada: statsmodels, sin penalizar.

    Por que regularizar puede pagar aca: son 13 parametros sobre 182 filas, y las 12
    features son lags de la misma serie, o sea que estan fuertemente correlacionadas
    entre si. Ese es el regimen clasico de varianza alta en OLS -- coeficientes grandes
    y de signos alternados que se cancelan en train y no generalizan.

    Ridge y Lasso se ajustan sobre variables ESTANDARIZADAS, y tambien se estandariza
    la clase. Sin eso alpha no significaria nada: las features son toneladas, el termino
    de error va en toneladas al cuadrado y aplasta cualquier penalizacion de magnitud
    razonable, con lo que alpha=1 y alpha=100 devolverian los mismos coeficientes que
    OLS. Estandarizando, la grilla de alphas es interpretable y comparable entre
    corridas. El intercepto nunca se penaliza (fit_intercept=True).
    """
    import numpy as np

    if metodo == "ols":
        import statsmodels.api as sm
        modelo = sm.OLS(y, sm.add_constant(X)).fit()
        print(f"R2 = {modelo.rsquared:.4f}   R2 ajustado = {modelo.rsquared_adj:.4f}")
        return lambda Xn: np.asarray(modelo.predict(sm.add_constant(Xn)))

    from sklearn.linear_model import Lasso, Ridge

    # Desvio 0 (una columna constante) dividiria por cero: se deja en 1, que equivale
    # a no escalar esa columna. Su coeficiente queda absorbido por el intercepto.
    mx = X.mean()
    sx = X.std(ddof=0).replace(0.0, 1.0)
    my = float(y.mean())
    sy = float(y.std(ddof=0))
    if sy == 0.0:
        sy = 1.0

    est = (Ridge if metodo == "ridge" else Lasso)(alpha=alpha, fit_intercept=True)
    est.fit((X - mx) / sx, (y - my) / sy)

    _nz = int((est.coef_ != 0).sum())
    print(f"{metodo} alpha={alpha:g}   coeficientes no nulos: {_nz}/{len(est.coef_)}"
          f"   |w| medio {np.abs(est.coef_).mean():.4f}")

    def _predecir(Xn):
        return np.asarray(est.predict((Xn - mx) / sx)) * sy + my

    return _predecir


def seleccionar_productos(ventas, criterio: str, n: int, origen: int):
    """product_ids que entran al ajuste. None = todos."""
    if criterio == "todos":
        return None
    if criterio == "magicos":
        return PRODUCTOS_MAGICOS

    # Los ultimos 12 meses hasta el origen: es la ventana que el modelo va a usar.
    ini = (origen // 100 - 1) * 100 + origen % 100 + 1
    v = ventas.filter(pl.col("periodo").is_between(ini, origen))

    if criterio == "topvol":
        # Los n de mayor volumen. Son los que dominan el denominador del WAPE, asi que
        # ajustar sobre ellos alinea el modelo con lo que la metrica premia.
        return (v.group_by("product_id").agg(pl.col("tn").sum().alias("v"))
                 .sort("v", descending=True).head(n)["product_id"].to_list())

    # 'estables': serie completa y de variacion moderada. El coeficiente de variacion
    # (desvio/media) mide lo erratico de la serie; los mas erraticos son los que
    # ensucian un ajuste por minimos cuadrados.
    est = (v.group_by("product_id")
            .agg(pl.len().alias("meses"),
                 pl.col("tn").mean().alias("m"),
                 pl.col("tn").std().alias("s"))
            .filter((pl.col("meses") == 12) & (pl.col("m") > 0))
            .with_columns((pl.col("s") / pl.col("m")).alias("cv"))
            .sort("cv"))
    return est.head(n)["product_id"].to_list()


def tabla_lags(ventas: "pl.DataFrame", lags: list, horizonte: int) -> "pl.DataFrame":
    """Una fila por (producto, periodo) con tn_0..tn_N y la clase a t+horizonte."""
    return (ventas.with_columns([
                pl.col("tn").shift(lag).over("product_id").alias(f"tn_{lag}")
                for lag in lags])
                  .rename({f"tn_{-horizonte}": "clase"}))


if __name__ == "__main__":
    main()
