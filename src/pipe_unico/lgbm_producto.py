# -*- coding: utf-8 -*-
"""LightGBM a nivel producto, construido con las lecciones que dejo linreg.py.

El pipeline grande (pipe_unico) llega a 0.264 con 600 features a nivel producto-cliente.
Una regresion lineal de 12 variables a nivel producto llega a 0.231. Las diferencias
estructurales entre los dos, y que este script adopta:

  1. NIVEL PRODUCTO. Se suma sobre los clientes de entrada. El WAPE de la competencia
     se mide por producto, asi que modelar el par producto-cliente agrega varianza en
     una dimension que la metrica despues colapsa igual.
  2. SOLO LOS 780 productos a entregar.
  3. POCAS FEATURES. 12 lags de tn -- un anio exacto. El barrido lo confirmo: 6 lags
     dan 0.347, 12 dan 0.231, 24 dan 0.303. Doce es el ciclo estacional completo.
  4. EL MES DEL ANIO COMO VARIABLE. Esto es lo que le falta al pipeline grande, que
     excluye 'periodo' de las features y por lo tanto NO SABE en que mes esta parado.
     linreg lo resuelve por otra via: entrenando solo con origenes de diciembre.

Aca las dos vias son palancas: --meses todos (con 'mes' de feature) o --meses diciembre.

    python lgbm_producto.py                          # todos los origenes + mes
    python lgbm_producto.py --meses diciembre        # solo origenes de diciembre
    python lgbm_producto.py --productos magicos      # la seleccion de la catedra
    python lgbm_producto.py --semillas 1,2,3         # ensemble de 3 semillas
    python lgbm_producto.py --extras                 # suma medias moviles y tendencia

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
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from linreg import PRODUCTOS_MAGICOS, seleccionar_productos, tabla_lags  # noqa: E402


def idx(p: int) -> int:
    """Periodo AAAAMM a indice de mes absoluto, para poder restar meses."""
    return (p // 100) * 12 + p % 100


def main() -> None:
    ap = argparse.ArgumentParser()
    # ── Datos, iguales a linreg.py ──────────────────────────────────────────
    ap.add_argument("--productos", choices=["magicos", "todos", "topvol", "estables"],
                    default="todos", help="que productos entran al ajuste")
    ap.add_argument("--n", type=int, default=182, help="cuantos con topvol o estables")
    ap.add_argument("--meses", default="todos",
                    help="'todos' | 'diciembre' | lista de periodos separados por coma")
    ap.add_argument("--lags", type=int, default=12)
    ap.add_argument("--horizonte", type=int, default=2)
    ap.add_argument("--origen", type=int, default=201912)
    ap.add_argument("--extras", action="store_true",
                    help="sumar medias moviles y pendiente ademas de los lags")
    # ── LightGBM ────────────────────────────────────────────────────────────
    ap.add_argument("--objetivo", default="regression",
                    choices=["regression", "regression_l1", "tweedie", "poisson"])
    ap.add_argument("--arboles", type=int, default=2000, help="techo; corta por early stopping")
    ap.add_argument("--hojas", type=int, default=31)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--min-hojas", dest="min_hojas", type=int, default=20,
                    help="min_child_samples: subilo si el dataset es chico")
    ap.add_argument("--semillas", default="102191",
                    help="una o mas semillas separadas por coma; se promedian")
    # Sin submuestreo, cambiar la semilla NO cambia el modelo: LightGBM es determinista
    # salvo por el sorteo de filas y columnas. Con estos dos por debajo de 1.0 cada
    # semilla ve datos distintos y promediarlas reduce varianza de verdad.
    ap.add_argument("--filas", type=float, default=0.8, help="bagging_fraction")
    ap.add_argument("--cols", type=float, default=0.8, help="feature_fraction")
    ap.add_argument("--val-meses", dest="val_meses", type=int, default=3,
                    help="cuantos origenes finales se reservan para el early stopping")
    ap.add_argument("--nombre", default=None)
    args = ap.parse_args()

    import lightgbm as lgb

    semillas = [int(s) for s in args.semillas.split(",")]

    # ── 1. Ventas por producto y periodo ────────────────────────────────────
    ventas = (pl.read_csv(L.DIR_RAW / "sell-in.txt.gz", separator="\t")
                .group_by("product_id", "periodo")
                .agg(pl.col("tn").sum().alias("tn")))
    oficial = pl.read_csv(L.DIR_RAW / "product_id_apredecir201912.txt",
                          separator="\t").select("product_id").unique()
    ventas = (ventas.join(oficial, on="product_id", how="inner")
                    .sort(["product_id", "periodo"]))
    print(f"{ventas['product_id'].n_unique()} productos, {ventas.height:,} filas producto-mes")

    # ── 2. Panel con lags y el mes del anio ─────────────────────────────────
    lags = [-args.horizonte, *range(args.lags)]
    tabla = tabla_lags(ventas, lags, args.horizonte)
    campos = [f"tn_{k}" for k in range(args.lags)]

    # 'mes' es LA feature que le falta al pipeline grande: sin ella el modelo no
    # puede distinguir "predecir febrero desde diciembre" de "predecir agosto desde
    # junio", y termina promediando regimenes estacionales distintos.
    tabla = tabla.with_columns((pl.col("periodo") % 100).cast(pl.Int8).alias("mes"))
    campos_modelo = campos + ["mes"]

    if args.extras:
        # Pocas y con sentido: nivel reciente a tres escalas y pendiente del anio.
        tabla = tabla.with_columns([
            pl.mean_horizontal([pl.col(f"tn_{k}") for k in range(3)]).alias("media3"),
            pl.mean_horizontal([pl.col(f"tn_{k}") for k in range(6)]).alias("media6"),
            pl.mean_horizontal([pl.col(f"tn_{k}") for k in range(args.lags)]).alias("media12"),
            (pl.col("tn_0") - pl.col(f"tn_{args.lags - 1}")).alias("pendiente"),
        ])
        campos_modelo += ["media3", "media6", "media12", "pendiente"]

    # ── 3. Que filas entran al entrenamiento ────────────────────────────────
    elegidos = seleccionar_productos(ventas, args.productos, args.n, args.origen)
    cond = pl.col("clase").is_not_null()
    for c in campos:
        cond = cond & pl.col(c).is_not_null()
    if elegidos is not None:
        cond = cond & pl.col("product_id").is_in(elegidos)

    if args.meses == "diciembre":
        cond = cond & (pl.col("mes") == args.origen % 100)
        print(f"Solo origenes del mes {args.origen % 100} (alineacion estacional estricta)")
    elif args.meses != "todos":
        cond = cond & pl.col("periodo").is_in([int(m) for m in args.meses.split(",")])

    datos = tabla.filter(cond).sort("periodo")
    if datos.is_empty():
        raise SystemExit("Sin filas de entrenamiento con esos filtros.")

    # ── 4. Corte temporal para el early stopping ────────────────────────────
    # Los ultimos origenes van a validacion, con un hueco de 'horizonte' meses para
    # que la clase de train no invada el periodo de validacion.
    origenes = sorted(datos["periodo"].unique().to_list())
    if len(origenes) > args.val_meses + 2:
        val_orig = origenes[-args.val_meses:]
        tope = idx(min(val_orig)) - args.horizonte
        tr_orig = [p for p in origenes if idx(p) <= tope]
    else:
        # Con pocos origenes (--meses diciembre deja uno o dos) no hay corte posible:
        # se entrena con todo y se usa el techo de arboles fijo, sin early stopping.
        val_orig, tr_orig = [], origenes
        print(f"[AVISO] solo {len(origenes)} origen(es) disponibles: no hay corte "
              f"temporal posible,")
        print(f"        asi que se entrenan los {args.arboles} arboles sin early "
              f"stopping. Con")
        print(f"        pocas filas eso sobreajusta: baja --arboles (100-300) o sube "
              f"--min-hojas.")

    dtr = datos.filter(pl.col("periodo").is_in(tr_orig))
    dva = datos.filter(pl.col("periodo").is_in(val_orig)) if val_orig else None
    print(f"Entrenamiento: {dtr.height:,} filas  ({len(tr_orig)} origenes)")
    if dva is not None:
        print(f"Validacion   : {dva.height:,} filas  (origenes {val_orig})")
    print(f"Features     : {len(campos_modelo)}  ->  {campos_modelo[:4]} ...")

    # ── 5. Entrenamiento, promediando semillas ──────────────────────────────
    Xtr, ytr = dtr.select(campos_modelo).to_pandas(), dtr["clase"].to_numpy()
    dfuture = tabla.filter(pl.col("periodo") == args.origen).drop_nulls(campos)
    Xfu = dfuture.select(campos_modelo).to_pandas()

    preds, rondas = [], []
    for s in semillas:
        params = dict(objective=args.objetivo, metric="mae", verbosity=-1,
                      boosting_type="gbdt", learning_rate=args.lr,
                      num_leaves=args.hojas, min_child_samples=args.min_hojas,
                      seed=s, bagging_seed=s, feature_fraction_seed=s,
                      bagging_fraction=args.filas, bagging_freq=1,
                      feature_fraction=args.cols,
                      n_jobs=-1, deterministic=True)
        ds_tr = lgb.Dataset(Xtr, label=ytr, categorical_feature=["mes"])
        if dva is not None:
            ds_va = lgb.Dataset(dva.select(campos_modelo).to_pandas(),
                                label=dva["clase"].to_numpy(),
                                categorical_feature=["mes"], reference=ds_tr)
            m = lgb.train(params, ds_tr, num_boost_round=args.arboles,
                          valid_sets=[ds_va],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            rondas.append(m.best_iteration)
        else:
            m = lgb.train(params, ds_tr, num_boost_round=args.arboles)
            rondas.append(args.arboles)
        preds.append(m.predict(Xfu))
        print(f"   semilla {s}: {rondas[-1]} arboles")

    reg = dfuture.select("product_id").with_columns(
        pl.Series("tn_pred", np.mean(preds, axis=0)))
    print(f"Prediccion por modelo : {reg.height} productos")

    # ── 6. Respaldo por promedio, igual que linreg ──────────────────────────
    ini = (args.origen // 100 - 1) * 100 + args.origen % 100 + 1
    promedio = (ventas.filter(pl.col("periodo").is_between(ini, args.origen))
                      .group_by("product_id").agg(pl.col("tn").mean().alias("tn")))
    final = (promedio.join(reg, on="product_id", how="left")
                     .with_columns(pl.coalesce(["tn_pred", "tn"]).alias("tn"))
                     .select(["product_id", "tn"]))
    print(f"Respaldo por promedio : {max(0, final.height - reg.height)} productos")

    # ── 7. Exactamente la lista oficial ─────────────────────────────────────
    final = (oficial.join(final, on="product_id", how="left")
                    .with_columns(pl.col("tn").fill_null(0.0).clip(0.0))
                    .sort("product_id"))
    if final.height != oficial.height:
        raise SystemExit(f"{final.height} filas contra {oficial.height} oficiales")

    _n = f"{args.n}" if args.productos in ("topvol", "estables") else ""
    nombre = args.nombre or (f"lgbmprod_{args.productos}{_n}_{args.lags}lags_"
                             f"{args.meses}_{args.objetivo}"
                             + ("_extras" if args.extras else "")
                             + (f"_{len(semillas)}sem" if len(semillas) > 1 else ""))
    dst = L.RUTA_EXP / nombre
    dst.mkdir(parents=True, exist_ok=True)
    path = dst / "submission_202002.csv"
    final.write_csv(path)

    print(f"\n{final.height} productos   tn total {final['tn'].sum():,.0f}")
    print(f"Guardado: {path}")
    print(f"\nkaggle competitions submit -c labo-iii-2026-rosario -f {path} -m \"{nombre}\"")


if __name__ == "__main__":
    main()
