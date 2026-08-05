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

```bash
pip install -q papermill
nohup python cola.py > ~/buckets/b1/exp/cola.log 2>&1 &
tail -f ~/buckets/b1/exp/cola.log
```

- `python cola.py --listar` muestra qué hay hecho, qué falta y el ranking por WAPE.
- Si Google mata la spot, se relanza el mismo comando: los experimentos terminados se
  saltean y el que estaba a medias retoma desde su último checkpoint.
- Un experimento que falla queda registrado y **no frena la cola**; se reintenta en la
  próxima corrida.
- El `nohup ... &` hace que siga corriendo aunque cierres el browser o se caiga la
  conexión. Lo único que la corta es que Google apague la máquina.
