# -*- coding: utf-8 -*-
"""Promedia las predicciones de varios experimentos en una sola entrega.

Por que conviene: con 11 experimentos subidos, el rango de puntajes de Kaggle es
0.264-0.331 y nueve caen entre 0.264 y 0.299. Las palancas ya no mueven la aguja.
Promediar modelos DIVERSOS si: sus errores no estan correlacionados y se cancelan
entre si. Y no cuesta computo: las predicciones ya estan calculadas.

    python mezclar.py --listar
    python mezclar.py tgtFilter cli1de2 grpProducto     # por fragmento del nombre
    python mezclar.py --todos --metodo mediana
    python mezclar.py A B C --pesos 2,1,1               # el primero pesa doble

Escribe exp/_mezclas/<nombre>.csv, listo para subir con la CLI de kaggle.
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
import pandas as pd  # noqa: E402

DIR_MEZCLAS = L.RUTA_EXP / "_mezclas"


# Un WAPE por encima de esto no es un modelo malo, es un desborde numerico (zscore
# sobre series casi constantes da ~1e9). Con --metodo mediana queda descartado por
# votacion, pero con media UN SOLO modelo asi destruye la mezcla entera: la media de
# los 14 dio 1.6e10 en Kaggle por culpa de ese unico experimento.
WAPE_ABSURDO = 5.0


def entregas(incluir_rotos: bool = False) -> dict:
    """{nombre_experimento: path del submission mas reciente}."""
    import json
    out, descartados = {}, []
    for d in sorted(L.RUTA_EXP.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        csvs = sorted(d.glob("submission_*.csv"))
        if not csvs:
            continue
        w = None
        res = d / "resultado.json"
        if res.exists():
            try:
                w = json.load(open(res, encoding="utf-8")).get("wape_test")
            except json.JSONDecodeError:
                pass
        if w is not None and w > WAPE_ABSURDO and not incluir_rotos:
            descartados.append((d.name, w))
            continue
        out[d.name] = csvs[-1]
    if descartados:
        print(f"[excluido] {len(descartados)} experimento(s) con wape_test absurdo "
              f"(desborde numerico, no modelo malo):")
        for n, w in descartados:
            print(f"      {w:.3e}   {n[:64]}")
        print("      Se incluyen con --incluir-rotos, pero con --metodo media "
              "arruinan la mezcla.\n")
    return out


def elegir(disponibles: dict, patrones: list) -> list:
    """Cada patron tiene que casar con EXACTAMENTE un experimento."""
    elegidos = []
    for p in patrones:
        m = [n for n in disponibles if p.lower() in n.lower()]
        if not m:
            raise SystemExit(f"'{p}' no casa con ningun experimento. Corre --listar.")
        # Un nombre completo gana sobre las coincidencias parciales. Sin esto, un
        # experimento cuyo nombre es prefijo de otro (linreg_..._201812 contra
        # linreg_..._201812-201712) seria imposible de elegir: no hay ningun
        # fragmento que identifique al corto sin casar tambien con el largo.
        exactos = [n for n in m if n.lower() == p.lower()]
        if exactos:
            elegidos.append(exactos[0])
            continue
        if len(m) > 1:
            print(f"'{p}' casa con {len(m)}:")
            for x in m:
                print(f"      {x}")
            raise SystemExit("Se mas especifico: cada patron tiene que elegir uno solo.")
        elegidos.append(m[0])
    return elegidos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("patrones", nargs="*", help="fragmentos del nombre de cada experimento")
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--todos", action="store_true", help="mezclar todas las entregas")
    ap.add_argument("--metodo", choices=["media", "mediana"], default="media",
                    help="mediana es mas robusta si sospechas de un modelo raro")
    ap.add_argument("--pesos", default=None,
                    help="pesos separados por coma, uno por experimento (solo con media)")
    ap.add_argument("--nombre", default=None, help="nombre del archivo de salida")
    ap.add_argument("--incluir-rotos", dest="rotos", action="store_true",
                    help="no excluir los experimentos con wape absurdo")
    args = ap.parse_args()

    disp = entregas(args.rotos)
    if not disp:
        raise SystemExit(f"No hay ningun submission_*.csv en {L.RUTA_EXP}")

    if args.listar or not (args.patrones or args.todos):
        print(f"{len(disp)} entregas disponibles:\n")
        for n, p in disp.items():
            filas = sum(1 for _ in open(p)) - 1
            print(f"   {filas:>5} filas   {n[:78]}")
        print("\nElegi por fragmento del nombre, ej:")
        print("   python mezclar.py tgtFilter cli1de2 grpProducto")
        return

    nombres = list(disp) if args.todos else elegir(disp, args.patrones)

    pesos = [1.0] * len(nombres)
    if args.pesos:
        pesos = [float(x) for x in args.pesos.split(",")]
        if len(pesos) != len(nombres):
            raise SystemExit(f"{len(pesos)} pesos para {len(nombres)} experimentos")
        if args.metodo != "media":
            raise SystemExit("Los pesos solo aplican con --metodo media")

    print(f"Mezclando {len(nombres)} entregas por {args.metodo}:\n")
    marcos = []
    for n, w in zip(nombres, pesos):
        df = pd.read_csv(disp[n])
        if list(df.columns) != ["product_id", "tn"]:
            raise SystemExit(f"{n}: columnas inesperadas {list(df.columns)}")
        print(f"   peso {w:>4.1f}   {len(df):>5} filas   tn total {df['tn'].sum():>12,.0f}   {n[:52]}")
        marcos.append(df.set_index("product_id")["tn"].rename(n))

    tabla = pd.concat(marcos, axis=1)

    # Todas las entregas tienen que cubrir los mismos productos: si una fue generada
    # antes del arreglo de las 780 filas, la mezcla saldria con huecos.
    if tabla.isna().any().any():
        faltan = tabla.isna().sum()
        print("\n[AVISO] hay productos que no estan en todas las entregas:")
        for n, c in faltan[faltan > 0].items():
            print(f"      {c:>4} faltantes en {n[:60]}")
        print("      Para esos productos se promedia solo con las que si lo tienen.")

    if args.metodo == "mediana":
        mezcla = tabla.median(axis=1, skipna=True)
    else:
        # Media ponderada ignorando los faltantes: cada producto se divide por la
        # suma de los pesos que efectivamente aportaron.
        w = pd.Series(pesos, index=tabla.columns)
        num = tabla.mul(w, axis=1).sum(axis=1, skipna=True)
        den = tabla.notna().mul(w, axis=1).sum(axis=1)
        mezcla = num / den

    salida = (mezcla.clip(lower=0).rename("tn").reset_index()
                    .sort_values("product_id").reset_index(drop=True))

    # ── La entrega tiene que ser EXACTAMENTE la lista oficial ───────────────
    # Al concatenar entregas se toma la UNION de los product_id. Si alguna es vieja
    # (anterior al arreglo de las 780 filas) trae productos de mas, y Kaggle rechaza
    # el archivo con "Submission must have N rows". Por eso se reindexa aca tambien.
    oficial = pd.read_csv(L.DIR_RAW / "product_id_apredecir201912.txt", sep="\t")
    oficial = oficial[["product_id"]].drop_duplicates()
    antes = len(salida)
    salida = oficial.merge(salida, on="product_id", how="left")
    sobran, faltan = antes - len(oficial), int(salida["tn"].isna().sum())
    if sobran > 0:
        print(f"\n{sobran} producto(s) predicho(s) que NO estan en la lista oficial: se descartan")
    if faltan:
        print(f"{faltan} producto(s) de la lista sin prediccion en ninguna entrega: van en 0")
        salida["tn"] = salida["tn"].fillna(0.0)
    salida = salida.sort_values("product_id").reset_index(drop=True)
    if len(salida) != len(oficial):
        raise SystemExit(f"La mezcla quedo con {len(salida)} filas y la lista oficial "
                         f"tiene {len(oficial)}. Kaggle la rechazaria.")

    # Correlacion entre modelos: si estan todos por encima de 0.99, son casi el mismo
    # y la mezcla no va a aportar mucho. Lo que hace ganar una mezcla es la diversidad.
    corr = tabla.corr()
    vals = [corr.iloc[i, j] for i in range(len(corr)) for j in range(i + 1, len(corr))]
    if vals:
        print(f"\nCorrelacion entre modelos: min {min(vals):.3f}   media "
              f"{sum(vals)/len(vals):.3f}   max {max(vals):.3f}")
        if min(vals) > 0.99:
            print("   ^ son casi identicos: la mezcla va a aportar poco. Busca "
                  "modelos mas distintos (otra granularidad, otro target).")

    DIR_MEZCLAS.mkdir(parents=True, exist_ok=True)
    nombre = args.nombre or f"mezcla_{len(nombres)}modelos_{args.metodo}.csv"
    path = DIR_MEZCLAS / nombre
    salida.to_csv(path, index=False)

    print(f"\n{len(salida):,} productos   tn total {salida['tn'].sum():,.0f}")
    print(f"Guardado: {path}")
    print(f"\nPara subirlo:\n\nkaggle competitions submit -c labo-iii-2026-rosario \\\\\n"
          f"  -f {path} \\\\\n  -m \"mezcla {args.metodo} de {len(nombres)} modelos\"")


if __name__ == "__main__":
    main()
