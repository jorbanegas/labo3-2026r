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

def rango_meses(desde: int, hasta: int) -> list:
    """Periodos AAAAMM consecutivos: rango_meses(201701, 201703) -> [201701, 201702, 201703].
    Misma definicion que en el notebook; hace falta aca porque BASE la usa."""
    a, b = (desde // 100) * 12 + desde % 100, (hasta // 100) * 12 + hasta % 100
    return [((m - 1) // 12) * 100 + ((m - 1) % 12) + 1 for m in range(a, b + 1)]


BASE = {
    'n_trials': 40,
    'submit':   False,          # nunca subir automaticamente desde la cola

    # ── VALIDACION ANCHA ────────────────────────────────────────────────────
    # La tanda anterior mostro que wape_val no predecia wape_test: correlacion de
    # Spearman -0.315, o sea que por ranking iba levemente AL REVES. Con 2 meses de
    # validacion, Optuna optimizaba ruido y sus 40 trials valian lo mismo que 40
    # tiros al azar.
    #
    # Con 6 meses de val la senial es mucho mas estable. Los meses respetan el gap
    # obligatorio del horizonte 2:
    #     max(train) 201901 + 2 = 201903 <= min(val) 201903   OK
    #     max(val)   201908 + 2 = 201910 <= min(test) 201910  OK
    #
    # El test sigue siendo UN mes y no hay forma de agrandarlo: 201911 y 201912 son
    # los meses de inferencia (los ultimos 2, con horizonte 2) y no pueden solaparse.
    # Asi que wape_test sigue siendo ruidoso; el que mejora es el criterio de Optuna.
    'meses_train': rango_meses(201701, 201901),
    'meses_val':   rango_meses(201903, 201908),
    'meses_test':  [201910],
}

# ── QUE DIJERON LOS RESULTADOS DE KAGGLE ─────────────────────────────────────
# Las metricas internas NO predicen el puntaje de Kaggle. Con 6 experimentos subidos:
#     Spearman  wape_test vs kaggle : -0.314   (negativa!)
#     Spearman  wape_val  vs kaggle : +0.029   (nula)
# lags_12, que tenia el MEJOR wape_test (0.367), quedo anteultimo en Kaggle (0.299).
#
# La unica relacion ordenada en los datos es la cantidad de clientes:
#     1 de cada 2 -> 0.267    1 de cada 4 -> 0.294    top 50 -> 0.331
# Monotona en tres puntos, y en direccion contraria a lo que sugeria el test interno:
# MAS DATOS, no menos.
#
# Por eso esta cola empuja hacia mas clientes. El problema es la memoria: 'todos' con
# 24 lags necesita 64 GB. Las dos formas de hacerlo entrar son bajar las columnas
# (lags_12 -> ~37 GB) o bajar las filas (solo_target -> ~40 GB).
#
# OJO al elegir el ganador de esta tanda: NO uses wape_test. Subi a Kaggle y decidi
# con ese numero, que es el unico que mide lo que se evalua.
COLA = [
    # El mejor de Kaggle hasta ahora, con la validacion nueva. Es la referencia.
    ("cli1de2",              {'clientes_n': 2}),

    # Empujar la direccion ganadora hasta donde la memoria permita.
    ("todos_lags12",         {'filtro_clientes': 'todos', 'max_lags': 12}),
    ("todos_solotarget",     {'filtro_clientes': 'todos', 'solo_target': True}),
    ("cli1de2_solotarget",   {'clientes_n': 2, 'solo_target': True}),

    # Reduccion de varianza sobre el mejor conocido. No cambia la busqueda, promedia
    # 3 modelos con distinta semilla. Es la mejora mas confiable que hay disponible.
    ("cli1de2_ensemble3",    {'clientes_n': 2,
                              'semillas_ensemble': [102191, 314159, 271828]}),

    # Mas capacidad, ahora que sabemos que simplificar no ayudaba.
    ("cli1de2_arb1000",      {'clientes_n': 2, 'techo_arboles': 1000}),

    # La regularizacion nunca llego a probarse (colisionaba de nombre con base).
    ("cli1de2_reg_fuerte",   {'clientes_n': 2, 'regularizacion': 'fuerte'}),

    # Nivel producto: quedo segundo y tercero en Kaggle con dos configuraciones
    # distintas, asi que la granularidad gruesa tiene algo. Y es baratisima de correr.
    ("producto",             {'agrupamiento': 'B'}),
    ("producto_ensemble3",   {'agrupamiento': 'B',
                              'semillas_ensemble': [102191, 314159, 271828]}),
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
