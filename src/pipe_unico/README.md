# Pipe único

Un solo notebook, de los crudos al submit, con reanudación ante la muerte de spot VMs.

- `pipe_unico.ipynb` — el notebook. Todo se controla desde la celda de palancas.
- `labo3.py` — módulo con las rutas, el sistema de checkpoints y la lógica de
  preprocesamiento y feature engineering (tomada verbatim de `../pipe/01` y `../pipe/02`).

Los notebooks originales en `../pipe/` y `../pipe_dtw/` quedan intactos.

## Palancas

`agrupamiento` (A/B), `completado`, `densificacion`, `solo_target`, `max_lags`,
`metodo_norm`, `salto_deltas`, `target`, meses de train/val/test, `filtro_clientes`
(`todos`/`muestreo`/`top`), `n_trials`, `techo_arboles`, `objective_lgbm`,
`regularizacion`, `semillas_ensemble`, `submit`. Cada una entra en el nombre, así que
cambiarla crea un experimento nuevo y volver atrás reusa los checkpoints viejos.

## Correr varios experimentos sin supervisión

`cola.py` ejecuta una lista de configuraciones una tras otra con papermill, cambiando
solo las palancas indicadas en cada una. La lista se edita al principio del archivo.

### Antes de lanzarla

1. En el notebook, dejá `'forzar': set()` y guardá con **Ctrl+S**. Papermill lee el
   archivo del disco, así que lo que quede guardado es el default de todos los
   experimentos de la cola.
2. Cerrá el kernel del notebook con **Kernel → Shut Down Kernel**. No alcanza con
   cerrar la pestaña: el kernel sigue reteniendo el dataset entero.

### Lanzarla

```bash
cd ~/buckets/b1/labo3-2026r/src/pipe_unico && pip install -q papermill
```

```bash
cd ~/buckets/b1/labo3-2026r/src/pipe_unico && nohup python cola.py > ~/buckets/b1/exp/cola.log 2>&1 &
```

El `nohup ... &` la deja en segundo plano: sobrevive a que cierres el browser o se
caiga la conexión. Devuelve el prompt enseguida con el número de proceso.

### Seguirla

```bash
tail -f ~/buckets/b1/exp/cola.log
```

Se sale del `tail` con **Ctrl+C**; eso corta la vista, no la cola.

```bash
cd ~/buckets/b1/labo3-2026r/src/pipe_unico && python cola.py --listar
```

Muestra qué experimentos están hechos, cuáles faltan y el ranking por WAPE. Se puede
correr mientras la cola trabaja.

### Pararla

```bash
pkill -f cola.py
```

```bash
rm -f ~/buckets/b1/exp/_corridas/cola.lock
```

El candado hay que borrarlo tanto si la matás vos como si la mata Google. Después se
relanza con el mismo comando de arriba y retoma donde iba.

### Subir el ganador

La cola **no sube nada a Kaggle**: su `BASE` fija `'submit': False`. Cada experimento
deja su CSV en su carpeta de `exp/`. Cuando termine, mirás el ranking con `--listar` y
subís a mano solo el que ganó:

```bash
kaggle competitions submit -c labo-iii-2026-rosario -f ~/buckets/b1/exp/NOMBRE_DEL_GANADOR/submission_202002.csv -m "ganador de la cola"
```

### Notas

- Si Google mata la spot, se relanza el mismo comando: los experimentos terminados se
  saltean y el que estaba a medias retoma desde su último checkpoint.
- Un experimento que falla queda registrado y **no frena la cola**; se reintenta en la
  próxima corrida.

### Una cosa por vez

`cola.py` toma un candado en `exp/_corridas/cola.lock` y avisa si hay poca memoria
libre. **No corras el notebook a mano y la cola al mismo tiempo**: dos procesos con el
dataset entero no entran en 64 GB, y como la VM tiene swap no muere ninguno — los dos
siguen vivos avanzando a paso de tortuga, que se nota recién horas después.

Si el candado quedó huérfano se borra con `rm ~/buckets/b1/exp/_corridas/cola.lock`,
o se ignora con `python cola.py --forzar-candado`.
