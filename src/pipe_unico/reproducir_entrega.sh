#!/usr/bin/env bash
# Reproduce la entrega final: cuarteto_10-1-1-1, 0.229 en el leaderboard publico.
#
#     bash reproducir_entrega.sh              # regenera todo y verifica
#     bash reproducir_entrega.sh --verificar  # solo compara lo que ya existe
#
# La entrega es una media ponderada de cuatro modelos, con pesos 10:1:1:1:
#
#   linreg          OLS de 13 parametros a nivel producto, 12 lags, entrenado con una
#                   sola fila por producto del 201812.                        (0.231)
#   tgtFilter       LightGBM producto-cliente, ~600 features, solo los 780.   (0.264)
#   lgbmprod_l1     LightGBM a nivel producto, 12 lags + mes, perdida L1.     (0.257)
#   grpProducto     LightGBM a nivel producto prediciendo el DELTA.           (0.271)
#     y-delta
#
# Los cuatro salen de familias estructurales distintas, que es lo que hace que la
# mezcla valga: cada uno se equivoca en otro lado.
#
# ── LO QUE ESTE SCRIPT NO PUEDE GARANTIZAR ──────────────────────────────────────
# Los dos modelos de pipe_unico dependen de un study de Optuna que vive en el bucket
# (db/*.db), NO en el repo. Al reejecutarlos se rehace la busqueda: el TPESampler esta
# sembrado, asi que con las mismas versiones de las librerias deberia recorrer los
# mismos trials, pero no es una garantia formal. Si el study viejo sigue en el bucket,
# el notebook lo levanta y no busca nada -- ese es el camino reproducible de verdad.
#
# Y los argumentos exactos de lgbmprod_l1_chico NO quedaron registrados: se corrio con
# --nombre, asi que la carpeta no los codifica. Los de abajo estan reconstruidos de la
# bitacora (regression_l1, 8 hojas, min 50). Por eso el paso de verificacion: si el
# CSV regenerado coincide con el guardado, la reconstruccion era correcta.
set -euo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$AQUI"
EXP="${LABO3_BUCKET:-$HOME/buckets/b1}/exp"
SOLO_VERIFICAR="${1:-}"

# ── Los nombres de carpeta del pipeline codifican la configuracion ──────────────
# Se derivan con glob en vez de escribirse a mano: si algun default cambia, el script
# falla ruidosamente en vez de mezclar el experimento equivocado.
TGT=$(basename "$(ls -d "$EXP"/*tgtFilter*regression__val*__arb500 | grep -v pesoVol | head -1)")
DLT=$(basename "$(ls -d "$EXP"/grpProducto_fillNA*y-delta* | head -1)")
echo "socio producto-cliente : $TGT"
echo "socio y-delta          : $DLT"
echo

if [ "$SOLO_VERIFICAR" != "--verificar" ]; then
    echo "══ 1/5  linreg (la regresion de la catedra, con sus defaults) ══"
    python linreg.py

    echo "══ 2/5  lgbm_producto con perdida L1 ══"
    # OJO: receta RECONSTRUIDA de la bitacora, no registrada. El paso 5 la verifica.
    python lgbm_producto.py --objetivo regression_l1 --hojas 8 --min-hojas 50 \
                            --nombre lgbmprod_l1_chico

    echo "══ 3/5  tgtFilter (pipe_unico, producto-cliente, solo los 780) ══"
    python -m papermill pipe_unico.ipynb "$EXP/_corridas/repro_tgtFilter.ipynb" \
        -y "OVERRIDES: {solo_target: true}" --cwd "$AQUI" --no-progress-bar

    echo "══ 4/5  grpProducto y-delta (pipe_unico, nivel producto, target delta) ══"
    python -m papermill pipe_unico.ipynb "$EXP/_corridas/repro_delta.ipynb" \
        -y "OVERRIDES: {agrupamiento: B, completado: null, target: clase_tn_delta}" \
        --cwd "$AQUI" --no-progress-bar
fi

echo "══ 5/5  la mezcla ══"
python mezclar.py linreg_magicos_12lags_201812 "$TGT" lgbmprod_l1_chico "$DLT" \
                  --pesos 10,1,1,1 --nombre cuarteto_10-1-1-1.csv

# ── Verificacion ────────────────────────────────────────────────────────────────
# La entrega tiene que dar exactamente 780 filas y coincidir con la que se subio.
ENTREGA="$EXP/_mezclas/cuarteto_10-1-1-1.csv"
FILAS=$(($(wc -l < "$ENTREGA") - 1))
echo
echo "Filas en la entrega: $FILAS  (tienen que ser 780)"
[ "$FILAS" -eq 780 ] || { echo "FALLO: Kaggle rechazaria este archivo"; exit 1; }

REF="$EXP/_mezclas/cuarteto_10-1-1-1.ref.csv"
if [ -f "$REF" ]; then
    if diff -q "$REF" "$ENTREGA" >/dev/null; then
        echo "IDENTICA a la entrega original: la reproduccion es exacta."
    else
        echo "DISTINTA de la entrega original."
        echo "   Lo mas probable es que la receta de lgbmprod_l1_chico este mal"
        echo "   reconstruida, o que Optuna haya encontrado otros hiperparametros."
        exit 1
    fi
else
    echo "No hay copia de referencia para comparar. Para dejarla, ANTES de correr"
    echo "este script por primera vez:"
    echo "   cp $ENTREGA $REF"
fi
