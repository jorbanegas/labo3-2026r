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
import atexit
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

    # ── EL MISMO SPLIT QUE LAS CORRIDAS QUE YA TIENEN PUNTAJE DE KAGGLE ──────
    # La version anterior de esta cola usaba validacion de 6 meses (train 201701-201901,
    # val 201903-201908). El argumento era que wape_val no predecia wape_test, asi que
    # 2 meses de val eran ruido y Optuna optimizaba nada.
    #
    # Esa cola nunca se corrio, y mientras tanto llegaron los puntajes de Kaggle, que
    # tumbaron el argumento: wape_test tampoco predice Kaggle (Spearman -0.13), asi que
    # "wape_val no predice wape_test" dejo de ser evidencia de nada. No hay ningun
    # motivo para creer que la validacion ancha da mejores puntajes; es una hipotesis
    # sin sustento, y cara: cambia el nombre del experimento, o sea que no reusa ningun
    # study ni checkpoint de Optuna ya corrido.
    #
    # Y sobre todo: esta tanda existe para medir UNA cosa, el efecto de regression_l1.
    # Si al mismo tiempo se mueve la validacion, un tgtFilter_l1 que salga 0.25 no dice
    # si gano el L1 o el split nuevo, y ya no queda tiempo para desambiguar.
    #
    # Estos son los defaults del notebook, con los que se corrieron tgtFilter (0.264),
    # cli1de2 (0.267) y todos los demas. Explicitos aca para que quede escrito que es
    # una eleccion y no un olvido.
    'meses_train': rango_meses(201701, 201905),
    'meses_val':   [201907, 201908],
    'meses_test':  [201910],
}

# ── POR QUE ESTA COLA ES SOLO regression_l1 ──────────────────────────────────
# La metrica de la competencia es WAPE: error ABSOLUTO. Los 14 experimentos de este
# pipeline se entrenaron con 'regression', que minimiza error CUADRATICO. Se estuvo
# optimizando una cosa distinta de la que evalua Kaggle, en todas las corridas.
#
# La evidencia de que importa es directa: en lgbm_producto.py, cambiar el objetivo y
# nada mas dio 0.275 -> 0.257, o sea 0.018. Es la mejora mas grande que produjo una
# sola palanca en toda la sesion, y ninguna otra movia nada -- los 14 experimentos
# quedaron amontonados entre 0.264 y 0.299 sin importar que se tocara. La hipotesis
# es que ese amontonamiento ERA el techo de la perdida equivocada.
#
# Cada linea de aca cambia UNA cosa respecto de una corrida que ya tiene puntaje de
# Kaggle, y esa cosa es el objetivo. Asi el resultado es una resta limpia.
#
# Baratas: el nombre del experimento incluye objective_lgbm pero NO afecta a los
# parquet de preprocesamiento ni de feature engineering, que son las dos etapas caras.
# Estas corridas los reusan tal cual y arrancan directo en Optuna.
#
# OJO al elegir el ganador: NO uses wape_test (Spearman -0.13 contra Kaggle, va al
# reves). Subi a Kaggle y decidi con ese numero, el unico que mide lo que se evalua.
COLA = [
    # ── LA APUESTA ──────────────────────────────────────────────────────────
    # El filtro de productos es la UNICA palanca que produjo una mejora grande en
    # algun lado: en linreg vale 0.04 (los 780 dan 0.271, los 182 magicos dan 0.231).
    # Nunca se transplanto al pipeline porque la palanca no existia -- solo_target
    # filtra a los 780, no a los 182. Ahora existe ('productos_train').
    #
    # Va con fillNA: a nivel producto un mes sin ventas es informacion ambigua, y
    # decirle al modelo "no se" en vez de "vendio 0" es mas honesto que a nivel
    # cliente. La combinacion nunca se corrio.
    ("producto_fillNA_magicos", {'agrupamiento': 'B',
                                 'completado': 'null',
                                 'productos_train': 'magicos'}),

    # ── LA QUE VA CONTRA LA EVIDENCIA ───────────────────────────────────────
    # Combinacion nueva, pero el hallazgo 5.7 apunta en contra: 'top' fue el peor
    # experimento del pipeline (0.331) y la direccion resulto monotona -- 1 de cada 2
    # da 0.267, 1 de cada 4 da 0.294, top 50 da 0.331. El top N cambia la POBLACION
    # (entrena solo con clientes grandes y se evalua contra todos), y fillNA no
    # deberia revertir una brecha de 0.064. Se corre igual: es barata y el pronostico
    # puede estar equivocado, que es exactamente lo que paso con lo del L1.
    #
    # n_trials mas bajo que la BASE a proposito: va segunda y quedan ~13 horas de
    # competencia. Vale mas que las dos terminen a que la primera use todo el tiempo.
    ("cliTop50_fillNA",         {'completado': 'null',
                                 'filtro_clientes': 'top',
                                 'clientes_n': 50,
                                 'n_trials': 25}),
]

# Tanda anterior (regression_l1), ya corrida y registrada: tgtFilter_l1 0.360,
# cli1de2_l1 0.360, producto_l1 0.290, contra 0.264 / 0.267 / 0.271 de sus gemelas
# L2. Quedan fuera de la lista para no confundir el --listar; el registro las
# saltearia igual.

# ── LO QUE NO ESTA ACA Y POR QUE ─────────────────────────────────────────────
# 'semillas_ensemble' NO entra en el nombre del experimento (ver la celda 3 del
# notebook). Un "tgtFilter_l1_ensemble3" caeria en la MISMA carpeta que tgtFilter_l1,
# encontraria su pred_infer.parquet ya escrito y se saltearia la etapa entera: daria
# por resultado una copia del anterior, sin ensemble y sin avisar. Es exactamente el
# bug que ya tuvo 'regularizacion'. Para probar el ensemble hay que correrlo a mano
# con 'forzar': {'final'}, sabiendo que pisa el resultado de la corrida simple.
#
# Tampoco esta la validacion de 6 meses, ni mas capacidad, ni mas clientes: todo eso
# es una palanca de segundo orden si la perdida esta equivocada. Primero se arregla
# la perdida, despues se vuelve a mirar el resto con los numeros nuevos.


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


def soltar_candado() -> None:
    """Borra el candado al terminar.

    Sin esto una cola que termina BIEN deja el archivo puesto, y la corrida siguiente
    arranca creyendo que hay otra viva. Paso dos veces, las dos con el candado de una
    VM que Google ya habia apagado, o sea que el aviso era siempre ruido.

    Se registra con atexit y no con un finally para no reindentar main(): cubre la
    salida normal y la excepcion no atrapada. Lo que NO cubre es que la spot muera de
    golpe -- y ahi el candado que queda SI es informacion util, porque avisa que hubo
    una corrida cortada a la mitad.
    """
    try:
        CANDADO.unlink(missing_ok=True)
    except OSError as e:
        print(f"[aviso] no se pudo borrar el candado: {e}")


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
    atexit.register(soltar_candado)
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
