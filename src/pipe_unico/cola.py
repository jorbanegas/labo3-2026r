# -*- coding: utf-8 -*-
"""Corre una cola de experimentos, uno tras otro, sin supervision.

Pensado para spot VMs: cada experimento se ejecuta con papermill sobre
pipe_unico.ipynb cambiando solo las palancas indicadas. Si Google mata la maquina,
se vuelve a lanzar exactamente el mismo comando: los experimentos ya terminados se
saltean y el que estaba a medias retoma desde su ultimo checkpoint.

    python cola.py                 # corre la cola definida abajo
    python cola.py --listar        # muestra que hay hecho y que falta
    python cola.py --rehacer       # ignora el registro y corre todo de nuevo

Para que sobreviva a que cierres el browser:

    nohup python cola.py > ~/buckets/b1/exp/cola.log 2>&1 &
    tail -f ~/buckets/b1/exp/cola.log
"""
import argparse
import io
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Los notebooks imprimen tablas de polars con caracteres de dibujo. Si la consola no
# es UTF-8 (Windows por defecto), eso hace fallar el experimento por un problema de
# impresion, no de calculo. errors='replace' para que nunca corte una corrida.
for _s in ("stdout", "stderr"):
    _f = getattr(sys, _s)
    if getattr(_f, "encoding", "").lower() not in ("utf-8", "utf8") and hasattr(_f, "buffer"):
        setattr(sys, _s, io.TextIOWrapper(_f.buffer, encoding="utf-8",
                                          errors="replace", line_buffering=True))

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI))
import labo3 as L  # noqa: E402

NOTEBOOK = AQUI / "pipe_unico.ipynb"
DIR_CORRIDAS = L.RUTA_EXP / "_corridas"     # notebooks ejecutados, uno por experimento
REGISTRO = L.RUTA_EXP / "_corridas" / "registro.json"


# ══════════════════════════════════════════════════════════════════════════
# LA COLA
# ══════════════════════════════════════════════════════════════════════════
# Una palanca por experimento respecto de la BASE. Es la unica forma de saber a que
# atribuir una mejora: si cambias dos cosas y el WAPE baja, no sabes cual fue.
#
# El 'nombre' es solo para el log y el registro; el experimento real se identifica
# por su carpeta en exp/, que sale de las palancas.
BASE = {
    'n_trials': 40,
    'submit':   False,          # nunca subir automaticamente desde la cola
}

COLA = [
    ("base",                {}),
    ("target_delta",        {'target': 'clase_tn_delta'}),
    ("target_nivel",        {'target': 'clase_tn'}),
    ("norm_zscore",         {'metodo_norm': 'zscore'}),
    ("clientes_todos",      {'filtro_clientes': 'todos'}),
    ("clientes_1de2",       {'clientes_n': 2}),
    ("solo_producto",       {'agrupamiento': 'B'}),
    ("solo_target_prods",   {'solo_target': True}),
    ("lags_12",             {'max_lags': 12}),
    ("regularizacion_fuerte", {'regularizacion': 'fuerte'}),
    ("arboles_1000",        {'techo_arboles': 1000}),
]


# ══════════════════════════════════════════════════════════════════════════
CANDADO = L.RUTA_EXP / "_corridas" / "cola.lock"


def memoria_libre_gb() -> float:
    """GB disponibles segun /proc/meminfo. -1 si no se puede leer (no-Linux)."""
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1024**2
    except OSError:
        pass
    return -1.0


def tomar_candado(forzar: bool = False) -> None:
    """Impide dos colas a la vez, o una cola mientras corre el notebook a mano.

    Dos procesos cargando el dataset completo no entran en 64 GB. El sistema no mata
    a ninguno porque hay swap: los dos siguen vivos avanzando a paso de tortuga, que
    es peor que un error porque no se nota hasta horas despues.
    """
    CANDADO.parent.mkdir(parents=True, exist_ok=True)
    if CANDADO.exists() and not forzar:
        try:
            info = json.load(open(CANDADO, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            info = {}
        print(f"Ya hay una cola corriendo (o quedo un candado viejo):\n"
              f"   {info}\n\n"
              f"Si Google mato esa maquina, el candado quedo huerfano y se borra con:\n"
              f"   rm {CANDADO}\n"
              f"o se ignora con:  python cola.py --forzar-candado")
        sys.exit(1)

    libre = memoria_libre_gb()
    if libre >= 0:
        print(f"Memoria disponible: {libre:.1f} GB")
        if libre < 20:
            print("[AVISO] Menos de 20 GB libres. Si tenes el notebook abierto con su\n"
                  "        kernel vivo, cerralo (Kernel -> Shut Down) antes de seguir:\n"
                  "        dos procesos con el dataset entero no entran y todo se va a swap.")

    import socket
    L.escribir_json({"pid": __import__("os").getpid(), "host": socket.gethostname(),
                     "desde": datetime.now().isoformat(timespec="seconds")}, CANDADO)


def cargar_registro() -> dict:
    if REGISTRO.exists():
        try:
            return json.load(open(REGISTRO, encoding="utf-8"))
        except json.JSONDecodeError:
            print("[aviso] registro ilegible, se empieza uno nuevo")
    return {}


def correr_uno(nombre: str, overrides: dict) -> dict:
    """Ejecuta el notebook con estas palancas. Devuelve el resumen del resultado."""
    import papermill as pm

    DIR_CORRIDAS.mkdir(parents=True, exist_ok=True)
    salida = DIR_CORRIDAS / f"{nombre}.ipynb"
    params = dict(BASE)
    params.update(overrides)

    t0 = time.time()
    pm.execute_notebook(
        str(NOTEBOOK), str(salida),
        parameters={"OVERRIDES": params},
        cwd=str(AQUI),
        progress_bar=False,
        log_output=True,
        stdout_file=sys.stdout, stderr_file=sys.stderr,
    )
    return {"ok": True, "segundos": round(time.time() - t0, 1),
            "notebook": str(salida), "overrides": params,
            "fecha": datetime.now().isoformat(timespec="seconds")}


def resultados() -> list:
    """Lee los resultado.json de todos los experimentos que haya en el bucket."""
    filas = []
    for d in sorted(L.RUTA_EXP.iterdir()):
        f = d / "resultado.json"
        if d.is_dir() and f.exists():
            try:
                r = json.load(open(f, encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            filas.append((r.get("wape_test"), r.get("wape_val"), d.name))
    return sorted(filas, key=lambda x: (x[0] is None, x[0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--listar", action="store_true", help="solo mostrar el estado")
    ap.add_argument("--rehacer", action="store_true", help="ignorar el registro")
    ap.add_argument("--forzar-candado", dest="forzar_candado", action="store_true",
                    help="correr aunque exista el candado de otra cola")
    args = ap.parse_args()

    reg = {} if args.rehacer else cargar_registro()

    if args.listar:
        print(f"Cola: {len(COLA)} experimentos\n")
        for nombre, ov in COLA:
            print(f"   {'HECHO' if reg.get(nombre, {}).get('ok') else '  --  '}  "
                  f"{nombre:24} {ov}")
        print("\nResultados en el bucket (mejor primero):")
        for wt, wv, n in resultados():
            print(f"   test {wt if wt is None else f'{wt:.5f}'}   "
                  f"val {wv if wv is None else f'{wv:.5f}'}   {n[:70]}")
        return

    tomar_candado(args.forzar_candado)
    L.limpiar_tmp()
    pendientes = [(n, o) for n, o in COLA if not reg.get(n, {}).get("ok")]
    print(f"Cola: {len(COLA)} experimentos, {len(pendientes)} pendientes\n")

    for i, (nombre, ov) in enumerate(COLA, 1):
        if reg.get(nombre, {}).get("ok"):
            print(f"[{i}/{len(COLA)}] {nombre}: ya estaba hecho, se saltea")
            continue
        print(f"\n{'='*70}\n[{i}/{len(COLA)}] {nombre}   {ov}\n{'='*70}", flush=True)
        try:
            reg[nombre] = correr_uno(nombre, ov)
            print(f"[{i}/{len(COLA)}] {nombre}: OK en {reg[nombre]['segundos']:,.0f}s")
        except KeyboardInterrupt:
            print("\nInterrumpido a mano.")
            break
        except Exception as e:
            # Un experimento que falla no puede frenar la cola: se registra y sigue.
            # Al relanzar se reintenta, y como los checkpoints estan, arranca avanzado.
            print(f"[{i}/{len(COLA)}] {nombre}: FALLO -> {type(e).__name__}: {e}")
            traceback.print_exc()
            reg[nombre] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                           "fecha": datetime.now().isoformat(timespec="seconds")}
        finally:
            L.escribir_json(reg, REGISTRO)   # el registro se guarda SIEMPRE

    print(f"\n{'='*70}\nResultados (mejor wape_test primero):")
    for wt, wv, n in resultados():
        print(f"   test {wt if wt is None else f'{wt:.5f}'}   "
              f"val {wv if wv is None else f'{wv:.5f}'}   {n[:70]}")


if __name__ == "__main__":
    main()
