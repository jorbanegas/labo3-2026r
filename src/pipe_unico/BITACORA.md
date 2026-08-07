# Bitácora — Labo III 2026 Rosario

Estado del trabajo, hallazgos y herramientas. Pensado para retomar el contexto en una
sesión nueva sin tener que reconstruir nada.

---

## 1. Contexto

- **Competencia Kaggle**: `labo-iii-2026-rosario`. Predecir toneladas de **202002**
  para **exactamente 780 productos** (`datasets/product_id_apredecir201912.txt`).
- **Métrica**: WAPE agregado por producto = `suma|error| / suma|real|`. **Menor es mejor.**
  Es una métrica de **error absoluto**, y los productos de mayor volumen dominan el
  denominador.
- **Datos**: `sell-in.txt.gz` (producto-cliente-mes, 2017-01 a 2019-12), `tb_productos.txt`,
  `tb_stocks.txt`. Horizonte de predicción: **t+2** (desde 201912 se predice 202002).
- **Repo**: `github.com/jorbanegas/labo3-2026r`
  - Laptop: `C:\Users\jorba\repos\labo3-2026r`, rama `main`.
  - VM: `~/buckets/b1/labo3-2026r`, rama `desktop-jr`.
- **Infra**: VM spot de Google Cloud, 64 GB / 8 vCPU, Toronto. Bucket montado con
  gcsfuse en `~/buckets/b1`. **El free trial limita a 8 vCPU simultáneas en total**, y
  la RAM va atada (máx 8 GB por vCPU), así que 64 GB *son* las 8 vCPU: no se puede
  paralelizar en varias máquinas grandes.
- **Referencia del profesor**: dijo haber alcanzado **0.17**. No se sabe cómo.

---

## 2. Resultado actual

**Mejor entrega: 0.229** — mezcla ponderada de `linreg` (peso 3 a 8) con `tgtFilter`
(peso 1). Archivos `mezcla_3a1.csv` … `mezcla_8a1.csv`.

Es una **meseta, no un punto**: cuatro pesos distintos dan el mismo 0.229, lo que hace
más probable que se sostenga en el puntaje privado.

---

## 3. Herramientas — `src/pipe_unico/`

| Archivo | Qué hace |
|---|---|
| `labo3.py` | Rutas, sistema de checkpoints (escritura atómica + validación `PAR1`), etapas de preprocesamiento y feature engineering tomadas verbatim de `../pipe/01` y `../pipe/02`. |
| `pipe_unico.ipynb` | Pipeline completo: crudos → FE → Optuna → modelo final → CSV → Kaggle. Todo se controla desde la celda de palancas (celda 6). Reanudable. |
| `cola.py` | Corre una lista de configuraciones en serie con papermill. Reanudable, con candado y guarda de memoria. |
| `mezclar.py` | Promedia predicciones de varios experimentos. Media o mediana, con pesos. |
| `subir.py` | Sube a Kaggle de mejor a peor, con registro y detección del límite diario. |
| `linreg.py` | La regresión lineal de la cátedra (`../Estadistica/z403`), portable y con palancas. |
| `lgbm_producto.py` | LightGBM a nivel producto, pocas features, con el mes del año como variable. |

Todos escriben en `exp/<nombre>/submission_202002.csv`, que es lo que `mezclar.py` y
`subir.py` consumen.

---

## 4. Resultados en Kaggle

### Modelos individuales

| Modelo | Kaggle |
|---|---|
| `linreg` magicos, 12 lags, origen 201812 | **0.231** |
| `linreg` magicos, dos diciembres (201812+201712) | 0.255 |
| `lgbmprod` regression_l1, 8 hojas, min 50 | **0.257** |
| LightGBM `tgtFilter` (`solo_target: True`) | 0.264 |
| `lgbmprod` tweedie, 8 hojas | 0.265 |
| LightGBM `clientes_1de2` | 0.267 |
| `linreg` todos los 780 productos | 0.271 |
| `lgbmprod` regression_l1, 16 hojas | 0.271 |
| producto fillNA y-delta (corrida vieja) | 0.271 |
| LightGBM `target_nivel` | 0.276 |
| `lgbmprod` regression (L2), 31 hojas | 0.275 |
| LightGBM `base` | 0.282 |
| `linreg` estables 182 | 0.288 |
| `lgbmprod` regression_l1, 4 hojas | 0.287 |
| LightGBM `lags_12` | 0.299 |
| `lgbmprod` solo diciembre | 0.306 |
| `linreg` topvol 300 / 182 / 100 / 50 | 0.312 / 0.400 / 0.584 / 0.757 |
| LightGBM `clientes_top50` | 0.331 |
| LightGBM `norm_zscore` | 4.6e9 (roto) |

### Mezclas

| Mezcla | Kaggle |
|---|---|
| **linreg + tgtFilter, pesos 3:1 a 8:1** | **0.229** |
| linreg + tgtFilter 12:1 / 16:1 | 0.230 |
| linreg + lgbmprod_l1, 4:1 y 6:1 | 0.230 |
| linreg + lgbmprod_l1, 3:1 | 0.231 |
| linreg + tgtFilter 2:1 | 0.231 |
| linreg + lgbmprod_l1, 2:1 | 0.232 |
| linreg + tgtFilter (50/50) | 0.237 |
| linreg + tgtFilter + cli1de2 (50/50/50) | 0.244 |
| 3 LightGBM (tgtFilter + cli1de2 + producto) | 0.256 |
| 4 LightGBM | 0.258 |
| 2 y 5 LightGBM | 0.263 |
| 6 LightGBM | 0.268 |
| mediana de 14 | 0.286 |
| media de 14 (con zscore adentro) | 1.6e10 |

### Barrido de lags en `linreg` (magicos, 201812)

| Lags | 6 | 9 | **12** | 15 | 18 | 24 |
|---|---|---|---|---|---|---|
| Kaggle | 0.347 | 0.270 | **0.231** | 0.267 | 0.278 | 0.303 |

---

## 5. Hallazgos

### 5.1 Las métricas internas NO predicen Kaggle

Con 11 experimentos: `wape_test` ↔ Kaggle da **Spearman −0.127, Pearson −0.016**.
`wape_val` ↔ Kaggle: **+0.029**. Ninguna informa nada.

Peor: `lags_12` tenía el **mejor** `wape_test` (0.367) y quedó anteúltimo en Kaggle
(0.299). **No elijas por `wape_test`.** El único criterio válido es el puntaje de Kaggle.

La causa estructural: el modelo final entrena con *todos* los meses y predice 202002,
mientras que el test mide un modelo entrenado con menos datos prediciendo 201910. Son
regímenes distintos.

### 5.2 Menos capacidad gana

Es la única dirección con señal consistente en toda la sesión.

- OLS de 13 parámetros sobre 182 filas (0.231) le gana a LightGBM con 600 features
  (0.264) y a LightGBM sobre **los mismos datos** (0.306 con `--meses diciembre`).
- Reducir hojas de 31 a 8 en `lgbm_producto`: 0.275 → 0.257.
- Agregar features (`--extras`) empeoró: 0.275 → 0.283.
- Pero tiene un piso: 4 hojas subajusta (0.287). El óptimo estuvo en 8.

### 5.3 `regression_l1` en vez de `regression`

WAPE es error **absoluto**; `regression` minimiza error **cuadrático**. Cambiar a
`regression_l1` dio **0.018** en `lgbm_producto` (0.275 → 0.257).

**Todos los experimentos del pipeline grande usaron `regression`.** Es la hipótesis
pendiente más prometedora: podría mover hacia abajo el rango 0.264–0.299 completo en el
que se amontonaron todos los experimentos, y explicaría por qué ninguna otra palanca
movía nada.

### 5.4 Mezclar: pocos, buenos y sobre todo DIVERSOS

- Con LightGBM parecidos entre sí, el óptimo fue **3 modelos** (curva en U clara:
  0.263 → 0.256 → 0.258 → 0.263 → 0.268 para 2, 3, 4, 5, 6).
- Mezclar 14 (mediana) dio 0.286: los modelos flojos ensucian.
- Con modelos de calidad dispar hay que **ponderar**: `linreg` (0.231) al 50/50 con
  `tgtFilter` (0.264) da 0.237, pero 4:1 da 0.229.
- **La diversidad importa más que la calidad del compañero**: `lgbmprod_l1` (0.257)
  mezcla peor que `tgtFilter` (0.264), porque el primero es estructuralmente casi igual
  a `linreg` (nivel producto, 12 lags) y el segundo es producto-cliente con 600 features.

### 5.5 Doce lags = un año

La U del barrido es nítida y simétrica. Doce lags cubren el ciclo estacional completo:
con menos se pierden meses del ciclo, con más se mete un segundo año que aporta ruido y
consume grados de libertad.

### 5.6 Los "productos mágicos" no responden a ningún criterio de los datos

Los 182 elegidos a mano por la cátedra dan 0.231. Alternativas probadas:

- los 780 completos → 0.271
- `topvol` (mayor volumen): **empeora monótonamente** cuanto más chica la selección —
  300 → 0.312, 182 → 0.400, 100 → 0.584, 50 → 0.757
- `estables` (menor coeficiente de variación) 182 → 0.288
- dos diciembres en vez de uno → 0.255 (más datos, peor)

Ni volumen, ni regularidad, ni cantidad lo explican. **Probablemente la lista fue
ajustada contra el leaderboard público**, lo que implica riesgo de que ese 0.231 se
degrade en el privado. Argumento a favor de entregar una mezcla y no `linreg` sola.

### 5.7 Top 50 clientes: cambio de población, no muestreo

`filtro_clientes: 'top'` fue el peor experimento del pipeline (0.331). El muestreo por
hash toma un subconjunto **representativo**; el top N **cambia la población**: entrena
solo con clientes grandes pero se evalúa contra todos, incluidos los chicos que nunca
vio. La dirección de clientes fue monótona: 1 de cada 2 → 0.267, 1 de cada 4 → 0.294,
top 50 → 0.331.

### 5.8 Normalización: solo `recta`

`zscore`, `minmax` y `media` **dividen** por una escala calculada sobre los lags. Con
demanda intermitente, series casi constantes dan desvíos de ~1e-9 y el target se va a
~1e9, envenenando el entrenamiento entero. `zscore` dio WAPE de 4.6e9. `recta` es
sustractiva y por eso es la única segura.

---

## 6. Trampas operativas

**JupyterLab pisa los merges.** Tiene el notebook en memoria y lo autoguarda; si hacés
`git merge` con el notebook abierto, el autosave revierte el archivo. Siempre:
**File → Close and Shut Down Notebook** antes de mergear. Para traer un archivo puntual
sin pelear con la rama:

```bash
cd ~/buckets/b1/labo3-2026r && git fetch origin && git checkout origin/main -- src/pipe_unico/
```

**gcsfuse y los permisos.** El bucket no tiene permisos Unix reales; git veía todos los
archivos como modificados. Resuelto con `git config core.fileMode false`, que queda
guardado en `.git/config` dentro del bucket y sobrevive a las VMs nuevas.

**Parquets truncados.** Si la spot muere durante una escritura, queda un archivo que
parece completo y revienta horas después. `labo3.py` escribe a `.tmp` y renombra, y
valida los bytes `PAR1` al leer.

**OOM.** `filtro_clientes: 'todos'` con 24 lags necesita ~64 GB y mata el proceso
entero, incluida la cola (SIGKILL no se puede capturar). El notebook ahora estima el
pico antes de materializar y corta con `MemoryError`, que la cola sí atrapa.

**Nunca dos procesos con el dataset a la vez.** Como la VM tiene 60 GB de swap, no
muere ninguno: los dos siguen vivos avanzando a paso de tortuga. `cola.py` toma un
candado en `exp/_corridas/cola.lock`.

**El nombre del experimento no incluye `semillas_ensemble`.** `EXPERIMENTO` arrastra
granularidad, target, objetivo, val/test, clientes, árboles y regularización — pero no
las semillas. Una corrida "con ensemble" cae en la **misma carpeta** que la simple,
encuentra su `pred_infer.parquet` ya escrito y se saltea la etapa: devuelve una copia
de la corrida sin ensemble, sin avisar. Es el mismo bug que tuvo `regularizacion`.
Para probar el ensemble hay que forzar (`'forzar': {'final'}`) asumiendo que pisa el
resultado anterior, o agregarle un tag al nombre.

**El log de la cola no muestra progreso de tqdm.** papermill captura la salida de a
bloques. Por eso Optuna imprime una línea por trial en vez de una barra.

**`tail -f` muestra solo las últimas 10 líneas.** Usar `tail -n 100 -f`.

**papermill se instala en el disco local**, no en el bucket: hay que reinstalarlo en
cada VM nueva (`pip install -q papermill`).

**La entrega debe tener exactamente 780 filas.** Tanto el notebook como `mezclar.py`
reindexan contra `product_id_apredecir201912.txt`.

---

## 7. Pendiente

1. **Correr la cola de `regression_l1`.** `cola.py` quedó reescrita con tres
   experimentos, cada uno una réplica de una corrida con puntaje conocido cambiando
   solo la pérdida: `tgtFilter_l1` (contra 0.264), `cli1de2_l1` (contra 0.267) y
   `producto_l1`. Reusan los parquets de preprocesamiento y FE, así que arrancan
   directo en Optuna. **Es lo más prometedor que queda.**

   La `BASE` volvió al split de los defaults del notebook (val `201907-201908`). La
   validación de 6 meses se sacó: se justificaba en que `wape_val` no predecía
   `wape_test`, pero el hallazgo 5.1 mostró que `wape_test` tampoco predice Kaggle,
   así que ese argumento se quedó sin evidencia — y moverla al mismo tiempo que la
   pérdida haría imposible atribuir la mejora.
2. `mezcla_trio_l1.csv` (linreg 6 + lgbmprod_l1 2 + tgtFilter 1) quedó generada sin subir.
3. Si `tgtFilter_l1` baja de 0.264, **rehacer la mezcla ponderada** contra `linreg`
   barriendo pesos 3:1 a 8:1, que es la meseta donde vive el 0.229.
4. **Marcar la entrega final** con el checkbox *Select* de Kaggle antes del cierre.
   Recomendación: `mezcla_4a1` (o cualquiera de la meseta 3:1–8:1), no `linreg` sola,
   por el riesgo del hallazgo 5.6.

Nota sobre `subir.py`: ordena los envíos por `wape_test`, que según el hallazgo 5.1
es levemente peor que al azar. Mientras el límite diario no corte, el orden no cambia
nada; si alguna vez corta, conviene subir a mano lo que importa primero.

---

## 8. Ideas no exploradas

- Ridge o Lasso en vez de OLS en `linreg`: con 13 parámetros y 182 filas, regularizar
  podría ganar algo.
- Transformación logarítmica de `tn` antes de ajustar (las ventas son lognormales).
- Ponderar el entrenamiento por volumen del producto, para alinear la pérdida con el
  denominador del WAPE.
- Un modelo específico para los productos que hoy caen en el respaldo por promedio
  (los que no tienen 12 meses de historia).
