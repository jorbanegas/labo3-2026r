# -*- coding: utf-8 -*-
"""Sube a Kaggle los submissions de los experimentos, del mejor al peor.

Kaggle limita los envios por dia (habitualmente 5). Este script:
  - ordena por wape_test, para que si el limite corta, lo bueno ya se mando;
  - lleva registro de lo enviado y no repite;
  - detecta el rechazo por limite diario y para, en vez de seguir chocando.

    python subir.py --listar      # que hay, que se subio, que falta
    python subir.py               # sube los pendientes (respeta --max)
    python subir.py --max 5       # cuantos como maximo en esta corrida
    python subir.py --incluir-rotos   # sube tambien los de wape absurdo
"""
import argparse
import io
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

for _s in ("stdout", "stderr"):
    _f = getattr(sys, _s)
    if getattr(_f, "encoding", "").lower() not in ("utf-8", "utf8") and hasattr(_f, "buffer"):
        setattr(sys, _s, io.TextIOWrapper(_f.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True))

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import labo3 as L  # noqa: E402

COMPETENCIA = "labo-iii-2026-rosario"
REGISTRO = L.RUTA_EXP / "_corridas" / "submits.json"

# Un WAPE por encima de esto no es un modelo malo, es un desborde numerico (el caso
# zscore dio 4.6e9). Subirlo solo gasta uno de los envios del dia.
WAPE_ABSURDO = 5.0


def candidatos(incluir_rotos: bool = False) -> list:
    """(wape_test, carpeta, csv) de cada experimento entregable, mejor primero."""
    filas = []
    for d in sorted(L.RUTA_EXP.iterdir()):
        if not d.is_dir():
            continue
        res = d / "resultado.json"
        if not res.exists():
            continue
        csvs = sorted(d.glob("submission_*.csv"))
        if not csvs:
            continue
        try:
            r = json.load(open(res, encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        w = r.get("wape_test")
        if w is None:
            continue
        if w > WAPE_ABSURDO and not incluir_rotos:
            continue
        filas.append((float(w), d.name, csvs[-1]))
    return sorted(filas)


def kaggle_cli(args: list) -> tuple:
    try:
        r = subprocess.run(["kaggle"] + args, capture_output=True, text=True, timeout=600)
        return r.returncode == 0, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError:
        return False, "No esta instalada la CLI de kaggle: pip install kaggle"
    except Exception as e:                       # noqa: BLE001
        return False, str(e)


def es_limite_diario(salida: str) -> bool:
    s = salida.lower()
    return any(t in s for t in ("maximum number of daily submissions",
                                "submission limit", "too many submissions",
                                "429", "exceeded"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true")
    ap.add_argument("--max", type=int, default=5,
                    help="cuantos subir como maximo en esta corrida (default 5)")
    ap.add_argument("--incluir-rotos", dest="rotos", action="store_true",
                    help=f"subir tambien los de wape_test > {WAPE_ABSURDO}")
    args = ap.parse_args()

    reg = {}
    if REGISTRO.exists():
        try:
            reg = json.load(open(REGISTRO, encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    cand = candidatos(args.rotos)
    if not cand:
        print("No hay experimentos con resultado.json y submission_*.csv")
        return

    print(f"{len(cand)} experimentos entregables (mejor wape_test primero):\n")
    for w, nom, _ in cand:
        print(f"   {'SUBIDO' if nom in reg else '  --  '}  test {w:.5f}  {nom[:66]}")

    rotos = [c for c in candidatos(True) if c[0] > WAPE_ABSURDO]
    if rotos and not args.rotos:
        print(f"\n{len(rotos)} excluido(s) por wape absurdo (desborde numerico, no modelo malo):")
        for w, nom, _ in rotos:
            print(f"      test {w:.3e}  {nom[:60]}")
        print("   Se suben igual con --incluir-rotos, pero gastan un envio del dia.")

    pendientes = [c for c in cand if c[1] not in reg]
    print(f"\nPendientes: {len(pendientes)}")
    if args.listar:
        return
    if not pendientes:
        print("Nada que subir.")
        return

    if not (Path.home() / ".kaggle" / "kaggle.json").exists():
        print("Faltan credenciales: subi kaggle.json al bucket y corre labo3.instalar_kaggle()")
        return

    enviados = 0
    for w, nom, csv in pendientes:
        if enviados >= args.max:
            print(f"\nLlegue al maximo de {args.max} envios en esta corrida.")
            print(f"Quedan {len(pendientes) - enviados}. Volve a correr esto manana.")
            break

        print(f"\n{'='*70}\nSubiendo (test {w:.5f}): {nom[:60]}\n   {csv.name}")
        ok, salida = kaggle_cli(["competitions", "submit", "-c", COMPETENCIA,
                                 "-f", str(csv), "-m", f"{nom[:80]} | wape_test={w:.5f}"])
        print(salida.strip() or "(sin respuesta)")

        if ok:
            reg[nom] = {"wape_test": w, "archivo": str(csv),
                        "fecha": datetime.now().isoformat(timespec="seconds")}
            L.escribir_json(reg, REGISTRO)
            enviados += 1
        elif es_limite_diario(salida):
            # No tiene sentido seguir: los proximos van a chocar contra lo mismo.
            print("\nKaggle corto por limite diario de envios.")
            print(f"Se subieron {enviados}. Volve a correr esto manana y sigue donde quedo.")
            break
        else:
            print("   [aviso] fallo este envio; se sigue con el siguiente.")

    print(f"\n{'='*70}\nEnviados en esta corrida: {enviados}")
    print(f"Registro: {REGISTRO}")


if __name__ == "__main__":
    main()
