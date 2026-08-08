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

Sigue en pie después de tres tandas que apuntaban a superarlo: `regression_l1` en el
pipeline grande (§5.3), regularización sobre `linreg` (§5.9), y el transplante del
filtro de productos mágicos al LightGBM más top 50 clientes con `fillNA` (§5.6, §5.7).
Las tres fallaron, y las dos primeras de forma monótona y sin óptimo interior. No
quedó ninguna dirección prometedora a medio explorar.

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
| `grpProducto` **regression_l1** | 0.290 |
| `tgtFilter` **regression_l1** | 0.360 |
| `cli1de2` **regression_l1** | 0.360 |
| `grpProducto` fillNA + **productos_train: magicos** | 0.306 |
| `clienteProducto` fillNA + **top 50 clientes** | 0.331 |

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
| linreg + tgtFilter_l1, 8:1 / 6:1 / 5:1 / 4:1 / 3:1 | 0.239 / 0.241 / 0.243 / 0.246 / 0.252 |
| trío linreg 6 + lgbmprod_l1 2 + tgtFilter 1 (L2) | 0.230 |
| trío linreg 6 + lgbmprod_l1 2 + tgtFilter_l1 1 | 0.239 |
| linreg + tgtFilter + cli1de2 (50/50/50) | 0.244 |
| 3 LightGBM (tgtFilter + cli1de2 + producto) | 0.256 |
| 4 LightGBM | 0.258 |
| 2 y 5 LightGBM | 0.263 |
| 6 LightGBM | 0.268 |
| mediana de 14 | 0.286 |
| media de 14 (con zscore adentro) | 1.6e10 |

### Barrido de regularización en `linreg` (magicos, 12 lags, 201812)

| Ridge α | 0 (OLS) | 3 | 10 | 30 | 100 | 300 | 1000 |
|---|---|---|---|---|---|---|---|
| Kaggle | **0.231** | 0.242 | 0.249 | 0.258 | 0.273 | 0.310 | 0.463 |

| Lasso α | 0 (OLS) | 0.01 | 0.03 | 0.1 | 0.3 |
|---|---|---|---|---|---|
| Kaggle | **0.231** | 0.243 | 0.248 | 0.274 | 0.408 |

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

### 5.2 Menos capacidad gana, pero solo cambiando de clase de modelo

- OLS de 13 parámetros sobre 182 filas (0.231) le gana a LightGBM con 600 features
  (0.264) y a LightGBM sobre **los mismos datos** (0.306 con `--meses diciembre`).
- Reducir hojas de 31 a 8 en `lgbm_producto`: 0.275 → 0.257.
- Agregar features (`--extras`) empeoró: 0.275 → 0.283.
- Pero tiene un piso: 4 hojas subajusta (0.287). El óptimo estuvo en 8.

**Ojo con generalizarlo**: encoger los coeficientes *dentro* de la clase de modelo no
es lo mismo que elegir una clase más chica, y el hallazgo 5.9 muestra que lo primero
solo destruye. Lo que gana es menos features y mejor alineación estacional, no
restringir el ajuste.

### 5.3 `regression_l1` ayuda a nivel producto y destruye a nivel producto-cliente

WAPE es error **absoluto** y `regression` minimiza error **cuadrático**, así que
cambiar a `regression_l1` parecía la corrección obvia. Dio **0.018** de mejora en
`lgbm_producto` (0.275 → 0.257) y se probó en el pipeline grande. **Fracasó, y el
daño crece con la granularidad:**

| Experimento | Granularidad | L2 | L1 | Δ |
|---|---|---|---|---|
| `tgtFilter` | producto-cliente | 0.264 | 0.360 | **+0.096** |
| `cli1de2` | producto-cliente | 0.267 | 0.360 | **+0.093** |
| `grpProducto` | producto | 0.271 | 0.290 | +0.019 |
| `lgbm_producto` | producto | 0.275 | 0.257 | −0.018 |

El barrido de mezclas lo confirma: es monótono (3:1 → 0.252 … 8:1 → 0.239),
convergiendo hacia `linreg` sola desde arriba. El socio L1 resta con cualquier peso.

**Primera explicación, descartada:** que L1 ajusta la mediana condicional, que con
demanda intermitente es 0 en la mayoría de las celdas producto-cliente, y que al sumar
sobre clientes eso subestimaría el total. Se midió y es falso — el L1 predice **más**
tonelaje que el L2 (30.441 contra 28.308).

**Explicación que queda en pie, sin confirmar:** WAPE se mide sobre los **totales por
producto**, así que lo dominan los productos de mayor volumen. L1 le da el mismo peso
marginal a cada fila, y a nivel producto-cliente hay millones de celdas chicas que
ahogan a las pocas grandes; L2, al castigar el error al cuadrado, atiende justo a las
filas grandes. A nivel producto ese desbalance casi no existe, y ahí L1 empata o gana.
Se testearía ponderando el entrenamiento por volumen del producto (ver §8).

**La lección transferible**: alinear la pérdida con la métrica solo vale si el modelo
predice en la misma unidad en que se evalúa. Si entre la predicción y la métrica hay
una agregación, la pérdida por fila y la métrica agregada son cosas distintas.

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

**Y no transfiere a otro modelo.** Se agregó la palanca `productos_train: 'magicos'`
al pipeline para entrenar el LightGBM con esos mismos 182 productos: dio **0.306**,
peor que entrenar con todos. El efecto es específico del modelo, no una propiedad de
los productos. En un OLS de 13 parámetros esas 182 filas *son* el dataset y sacar las
series erráticas evita que los outliers corran los coeficientes; en LightGBM,
restringir a 182 productos tira el grueso de la señal de entrenamiento de un modelo
que necesita datos.

### 5.7 Top 50 clientes: cambio de población, no muestreo

`filtro_clientes: 'top'` fue el peor experimento del pipeline (0.331). El muestreo por
hash toma un subconjunto **representativo**; el top N **cambia la población**: entrena
solo con clientes grandes pero se evalúa contra todos, incluidos los chicos que nunca
vio. La dirección de clientes fue monótona: 1 de cada 2 → 0.267, 1 de cada 4 → 0.294,
top 50 → 0.331.

Confirmado después: top 50 con `fillNA` en vez de `fill0` dio **0.331 otra vez**,
idéntico hasta el tercer decimal. Cuando la población de entrenamiento no es la que se
evalúa, ese error domina y ninguna otra palanca se nota.

### 5.8 Normalización: solo `recta`

`zscore`, `minmax` y `media` **dividen** por una escala calculada sobre los lags. Con
demanda intermitente, series casi constantes dan desvíos de ~1e-9 y el target se va a
~1e9, envenenando el entrenamiento entero. `zscore` dio WAPE de 4.6e9. `recta` es
sustractiva y por eso es la única segura.

### 5.9 Regularizar `linreg` solo empeora — no había sobreajuste que corregir

Diez alphas, dos familias de penalización, **monótono en ambas y sin óptimo interior**:
Ridge 3 → 0.242 hasta 1000 → 0.463; Lasso 0.01 → 0.243 hasta 0.3 → 0.408. Las dos
curvas extrapolan a 0.231 (OLS) cuando α→0.

Era la apuesta razonable — 13 parámetros sobre 182 filas, con 12 features que son lags
correlacionados de la misma serie — y la respuesta fue que **el ajuste no estaba
limitado por varianza**. Con la clase estandarizada, α grande colapsa las predicciones
hacia la media del target de entrenamiento (los 182 mágicos en 201812); predecir esa
media para todos es pésimo en una métrica dominada por los productos de mayor volumen,
y de ahí el 0.463.

Junto con el hallazgo 5.3, cierra las dos hipótesis que quedaban sobre `linreg` y el
pipeline grande. Todo lo probado después del 0.229 lo empeoró.

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

**Marcar la entrega final** con el checkbox *Select* de Kaggle antes del cierre. Es lo
único que falta: las dos hipótesis que quedaban (5.3 y 5.9) se probaron y fallaron, y
todo lo enviado después del 0.229 quedó por encima.

Recomendación: **`mezcla_5a1`**. La meseta de 0.229 va de 3:1 a 8:1 y se degrada en
2:1 (0.231) y en 12:1 (0.230), así que el centro está cerca de 5:1 — el punto más
lejano de los dos bordes, que es lo que hay que maximizar cuando el puntaje privado se
calcula sobre otra muestra.

Si Kaggle permite marcar dos, la segunda es **`mezcla_trio_l1`** (0.230): reparte entre
tres modelos en vez de dos, así que es el hedge más diversificado disponible.

No marcar `linreg` sola aunque sea el mejor modelo individual: por el hallazgo 5.6 su
lista de 182 productos huele a ajuste contra el leaderboard público.

Nota sobre `subir.py`: ordena los envíos por `wape_test`, que según el hallazgo 5.1
es levemente peor que al azar. Mientras el límite diario no corte, el orden no cambia
nada; si alguna vez corta, conviene subir a mano lo que importa primero.

---

## 8. Ideas no exploradas

- Transformación logarítmica de `tn` antes de ajustar (las ventas son lognormales).
- **Ponderar el entrenamiento por volumen del producto**, para alinear la pérdida con
  el denominador del WAPE. Subió de prioridad: es el test de la explicación que quedó
  viva en el hallazgo 5.3. Requiere tocar el notebook (`entrenar()` ya acepta
  `sample_weight`, hoy solo lo usa `decay_recencia`) y una corrida completa de Optuna.
- Un modelo específico para los productos que hoy caen en el respaldo por promedio
  (los que no tienen 12 meses de historia).
