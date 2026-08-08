# Predicción de demanda a dos meses — competencia `labo-iii-2026-rosario`

**Laboratorio de Implementación III — Universidad Austral**
Jorgelina Banegas · agosto de 2026

Repositorio: `github.com/jorbanegas/labo3-2026r`
Bitácora técnica completa: [`src/pipe_unico/BITACORA.md`](src/pipe_unico/BITACORA.md)

---

## 1. Resumen

Se predijo el volumen de ventas (toneladas) de **780 productos** para **febrero de 2020**,
a partir de historia mensual de sell-in entre enero de 2017 y diciembre de 2019. El
horizonte es de **dos meses**: se predice 202002 parado en 201912.

**Resultado: WAPE 0.229** en el leaderboard público, con una media ponderada de cuatro
modelos de familias estructurales distintas.

El hallazgo central del trabajo no es el modelo ganador sino algo que condicionó todo
lo demás: **las métricas de validación interna no predicen el desempeño real** en esta
competencia (correlación de Spearman de −0.13 contra el puntaje de Kaggle, es decir,
levemente peor que elegir al azar). Eso invalidó el ciclo de trabajo habitual —probar,
medir internamente, quedarse con lo mejor— y obligó a usar el leaderboard como único
criterio de selección, con las restricciones de presupuesto de envíos que eso impone.

El segundo hallazgo, y el más incómodo, es que **un modelo lineal de 13 parámetros le
gana a un LightGBM con 600 features** sobre los mismos datos (0.231 contra 0.264), y
que ninguna de las piezas que explicarían esa ventaja transfiere al modelo grande
cuando se las aísla y se las transplanta.

---

## 2. El problema

| | |
|---|---|
| **Objetivo** | Toneladas vendidas de cada producto en 202002 |
| **Universo** | 780 productos (`product_id_apredecir201912.txt`) |
| **Métrica** | WAPE agregado por producto: `Σ\|error\| / Σ\|real\|` — menor es mejor |
| **Horizonte** | t+2 (desde diciembre de 2019) |
| **Datos** | `sell-in.txt.gz` (producto-cliente-mes, 2017-01 a 2019-12), `tb_productos`, `tb_stocks` |

Dos propiedades de la métrica gobiernan todas las decisiones que siguen:

1. **Es error absoluto**, no cuadrático. Esto motivó una de las hipótesis centrales del
   trabajo (§5.1), que resultó falsa.
2. **Se mide sobre los totales por producto**, después de agregar sobre los clientes.
   Los productos de mayor volumen dominan el denominador, así que un error del 10% en
   un producto grande pesa más que un error del 100% en uno chico.

La demanda es **intermitente**: la mayoría de las celdas producto-cliente-mes son cero
o están vacías. Esto hace que varias transformaciones estándar sean peligrosas (§5.5).

---

## 3. Infraestructura y forma de trabajo

El cómputo corrió en una **VM spot de Google Cloud** (64 GB de RAM, 8 vCPU), que el
proveedor apaga cada 8-10 horas sin aviso. El free trial limita a 8 vCPU en total, así
que no fue posible paralelizar en varias máquinas: los experimentos corrieron en serie.

Esa restricción determinó la arquitectura del código más que cualquier consideración de
modelado. Todo el pipeline es **reanudable**:

- Cada etapa cara escribe su resultado a un bucket montado con `gcsfuse` y **se saltea
  sola si ya está hecha**. Al morir la máquina, se levanta otra y se reejecuta el mismo
  comando: retoma donde quedó.
- **Escritura atómica**: todo se escribe a un `.tmp` y se renombra al terminar. Un corte
  a mitad de camino nunca deja un archivo con el nombre definitivo.
- **Validación al leer**: un parquet válido termina con los bytes `PAR1`. Un checkpoint
  truncado se detecta y se recalcula, en vez de explotar tres etapas más adelante.
- El study de Optuna vive en SQLite dentro del bucket y se respalda cada N trials.
- Una **guarda de memoria** estima el pico antes de materializar el dataset y corta con
  `MemoryError`. Sin esto, el OOM killer manda `SIGKILL`, que no se puede capturar y se
  lleva puesta la cola entera de experimentos.

El diseño clave es que **la configuración se codifica en el nombre de los archivos**.
Cambiar una palanca crea un experimento nuevo sin pisar el anterior, y volver a la
configuración previa reusa sus checkpoints. Esto tuvo un costo: tres bugs de la misma
familia, en los que una palanca que cambiaba el entrenamiento *no* entraba en el nombre,
de modo que la corrida nueva encontraba el resultado de la vieja y se salteaba entera
devolviendo un resultado ajeno. Se detectaron y corrigieron para `regularizacion`,
`decay_recencia` y `semillas_ensemble`.

---

## 4. Modelos construidos

| Herramienta | Qué es | Mejor puntaje |
|---|---|---|
| `linreg.py` | OLS de 13 parámetros a nivel producto, 12 lags, entrenado con **una sola fila por producto** del período 201812 | **0.231** |
| `pipe_unico.ipynb` | Pipeline completo producto-cliente: ~600 features, normalización, Optuna, ensemble | 0.264 |
| `lgbm_producto.py` | LightGBM a nivel producto, 12 lags + mes del año, pérdida L1 | 0.257 |
| `mezclar.py` | Media o mediana ponderada de predicciones | **0.229** |
| `cola.py` | Runner de experimentos en serie con papermill, reanudable y con candado | — |
| `subir.py` | Envío a Kaggle con registro y detección del límite diario | — |

El pipeline grande explora un espacio amplio: granularidad producto-cliente o producto,
tratamiento de faltantes (cero o nulo), hasta 24 lags, cuatro métodos de normalización,
tres variables respuesta (nivel, normalizada, delta), población de clientes, y búsqueda
bayesiana de hiperparámetros con Optuna.

---

## 5. Hallazgos

### 5.1 Las métricas internas no predicen el desempeño real

Con 11 experimentos con puntaje de Kaggle conocido:

| Correlación con Kaggle | Spearman | Pearson |
|---|---|---|
| `wape_test` (holdout interno) | **−0.13** | −0.02 |
| `wape_val` (validación) | +0.03 | — |

El caso más claro: el experimento `lags_12` tenía el **mejor** `wape_test` de todos
(0.367) y quedó anteúltimo en Kaggle (0.299).

La causa es estructural, no un error de implementación. El modelo final entrena con
*todos* los meses disponibles y predice 202002, mientras que el test interno mide un
modelo entrenado con menos datos prediciendo 201910. Son regímenes distintos, y el
ordenamiento de configuraciones en uno no se traslada al otro. El test tampoco se puede
agrandar: 201911 y 201912 son los meses de inferencia y no pueden solaparse.

**Consecuencia metodológica.** Este resultado invalidó el ciclo de trabajo estándar. A
partir de acá cada experimento tuvo que subirse a Kaggle para saber si servía, con lo
que el diseño experimental pasó a estar limitado por el presupuesto de envíos y no por
el cómputo. También explica en retrospectiva por qué la búsqueda de hiperparámetros con
Optuna aportó tan poco: estaba optimizando un criterio no correlacionado con el objetivo.

### 5.2 Menos capacidad gana — pero solo cambiando de clase de modelo

Es la única dirección con señal consistente:

- OLS de 13 parámetros sobre 182 filas (**0.231**) le gana a LightGBM con 600 features
  (0.264), y también a LightGBM entrenado sobre **los mismos datos** (0.306).
- Reducir las hojas de 31 a 8 en `lgbm_producto`: 0.275 → **0.257**.
- Agregar features derivadas empeoró: 0.275 → 0.283.
- Pero tiene un piso: con 4 hojas subajusta (0.287). El óptimo estuvo en 8.

**El matiz importa y casi lo pasamos por alto.** Reducir capacidad *dentro* de una clase
de modelo no es lo mismo que elegir una clase más chica, y solo lo segundo funciona
(§5.4). Lo que gana es menos features y mejor alineación estacional, no restringir el
ajuste.

### 5.3 Doce lags son exactamente un año

| Lags | 6 | 9 | **12** | 15 | 18 | 24 |
|---|---|---|---|---|---|---|
| Kaggle | 0.347 | 0.270 | **0.231** | 0.267 | 0.278 | 0.303 |

La U es nítida y simétrica. Doce lags cubren el ciclo estacional completo: con menos se
pierden meses del ciclo, con más se incorpora un segundo año que aporta ruido y consume
grados de libertad. Es el resultado más limpio de todo el trabajo y el único donde el
óptimo interior aparece con claridad.

### 5.4 Cinco hipótesis razonables, cinco fracasos

Cada una tenía un argumento respetable detrás. Todas empeoraron el resultado.

| Hipótesis | Razonamiento | Resultado |
|---|---|---|
| **`regression_l1`** | WAPE es error absoluto; entrenar con error cuadrático optimiza otra cosa | 0.264 → **0.360** |
| **Ridge / Lasso** | 13 parámetros sobre 182 filas con features colineales: régimen clásico de varianza alta | monótono, 0.231 → 0.463 |
| **Filtro de productos** | La lista de 182 productos vale 0.04 en `linreg`; transplantarla al LightGBM | **0.306** |
| **Peso por volumen** | WAPE lo dominan los productos grandes; reponderar las filas para alinear la pérdida | monótono, 0.264 → 0.282 |
| **Mes del año** | El pipeline no sabe en qué mes está parado; `linreg` sí, y gana | 0.264 → **0.279** |

Tres de las cinco fallaron **monótonamente y sin óptimo interior** — Ridge fue de 0.242
con α=3 a 0.463 con α=1000; el peso por volumen, de 0.272 con raíz a 0.282 con lineal.
Esa forma es informativa: descarta que faltara afinar un parámetro y dice que la
dirección entera está equivocada.

El caso de `regression_l1` merece detalle porque el fracaso fue instructivo. La misma
corrección **funciona** a nivel producto (0.275 → 0.257 en `lgbm_producto`) y **destruye**
a nivel producto-cliente (+0.096). Se propusieron dos explicaciones y se midieron las dos:

1. *«L1 ajusta la mediana condicional, que con demanda intermitente es cero en la
   mayoría de las celdas, y al sumar sobre clientes subestima el total.»* Se midió el
   tonelaje total predicho: el modelo L1 predice **más**, no menos (30.441 contra
   28.308). Falsa.
2. *«L1 le da el mismo peso marginal a cada fila mientras WAPE solo mira los totales
   grandes.»* Se implementó la ponderación por volumen para corregirlo exactamente:
   **0.361 contra 0.360**. No recuperó nada. Falsa.

Queda anotado como hecho sin explicación. Es preferible a inventar una tercera historia
que no se pueda testear.

### 5.5 Normalización: solo la sustractiva es segura

Los métodos `zscore`, `minmax` y `media` **dividen** por una escala calculada sobre los
lags. Con demanda intermitente, las series casi constantes producen desvíos del orden de
1e-9, el target normalizado se va a ~1e9 y envenena el entrenamiento completo: el
experimento con `zscore` dio un WAPE de 4.6e9.

El método `recta` (restar la recta ajustada sobre los lags) es sustractivo y por eso es
el único seguro. Esto obligó además a blindar `mezclar.py`: **un solo modelo roto
destruye una media**, y la media de 14 modelos dio 1.6e10 por culpa de ese único
experimento. La versión final excluye automáticamente los WAPE absurdos.

### 5.6 Cambiar la población de entrenamiento es peor que muestrearla

| Clientes en entrenamiento | Kaggle |
|---|---|
| 1 de cada 2 (muestreo por hash) | 0.267 |
| 1 de cada 4 | 0.294 |
| Los 50 de mayor volumen | **0.331** |

El muestreo por hash toma un subconjunto **representativo**; el top N **cambia la
población**: entrena solo con clientes grandes pero se evalúa contra todos, incluidos los
chicos que el modelo nunca vio. Correr la misma configuración cambiando el tratamiento de
faltantes dio **0.331 otra vez**, idéntico hasta el tercer decimal: cuando la población de
entrenamiento no es la que se evalúa, ese error domina y ninguna otra palanca se nota.

### 5.7 Lo que hace andar al modelo chico no se pudo aislar

`linreg` (0.231) le gana a todo el pipeline (0.264), y sus dos diferencias estructurales
más claras son identificables: entrena solo con 182 productos elegidos a mano, y entrena
con orígenes de diciembre —o sea que aprende «cómo es febrero visto desde diciembre»—
mientras el pipeline mezcla todos los meses y ni siquiera tiene una variable de calendario.

Se aislaron las dos y se transplantaron. **Ninguna transfiere**: los productos dieron
0.306 y el mes del año 0.279, ambos peores que la base.

La hipótesis que sobrevive, sin poder testearse en el tiempo disponible, es que lo que
gana no es una pieza sino el conjunto: 13 parámetros sobre 182 filas es un régimen tan
distinto de 600 features sobre millones de filas que las piezas no significan lo mismo a
los dos lados.

### 5.8 Mezclar: pocos, buenos y sobre todo diversos

Es lo único que produjo una mejora sobre el mejor modelo individual.

- Con modelos parecidos entre sí, el óptimo fue **tres**, con una U clara: 0.263, 0.256,
  0.258, 0.263, 0.268 para 2, 3, 4, 5 y 6 modelos. Mezclar 14 dio 0.286 — los modelos
  flojos ensucian.
- Con modelos de calidad dispar hay que **ponderar**: `linreg` (0.231) al 50/50 con
  `tgtFilter` (0.264) da 0.237, pero a 4:1 da **0.229**.
- **La diversidad importa más que la calidad del socio.** `lgbmprod_l1` (0.257) mezcla
  peor que `tgtFilter` (0.264), porque el primero es estructuralmente casi igual a
  `linreg` —nivel producto, 12 lags— y el segundo es producto-cliente con 600 features.
- Pero la diversidad **de target no sirve**: los socios que predicen el nivel o el delta,
  mezclados con `linreg`, dan 0.232–0.234, peor que `linreg` sola.

El 0.229 resultó ser un **piso ancho**: 33 mezclas distintas alrededor del óptimo y
ninguna por debajo.

---

## 6. La entrega final

**`cuarteto_10-1-1-1`** — media ponderada con pesos 10:1:1:1 de:

| Modelo | Familia estructural | Individual |
|---|---|---|
| `linreg` | OLS, nivel producto, 12 lags, origen diciembre | 0.231 |
| `tgtFilter` | LightGBM, producto-cliente, ~600 features | 0.264 |
| `lgbmprod_l1` | LightGBM, nivel producto, pérdida L1, con mes | 0.257 |
| `grpProducto y-delta` | LightGBM, nivel producto, predice el cambio | 0.271 |

Se eligió entre **varias mezclas empatadas en 0.229** aplicando dos criterios que no
aparecen en el leaderboard público:

1. **Máxima diversificación.** Cuatro familias estructurales en vez de dos. Por §5.8, la
   diversidad es lo que sostiene una mezcla, y el puntaje privado se calcula sobre otra
   muestra.
2. **Posición interior en la meseta.** Los pesos 8:1:1:1 y 12:1:1:1 también dan 0.229,
   uno de cada lado. En cambio 8:1:1:1 —que era el candidato inicial— está en el borde:
   inmediatamente debajo, 6:1:1:1 cae a 0.230.

**Se descartó entregar `linreg` sola** pese a ser el mejor modelo individual. Su lista de
182 productos no responde a ningún criterio verificable de los datos: no es volumen (la
selección explícita por volumen empeora monótonamente, hasta 0.757 con 50 productos), no
es regularidad (0.288), no es cantidad (0.271 con los 780). Que funcione tan bien sin una
razón identificable es compatible con que haya sido ajustada contra el leaderboard
público, lo que implica riesgo concreto de degradación en el privado.

---

## 7. Reproducibilidad

`src/pipe_unico/reproducir_entrega.sh` regenera los cuatro modelos y la mezcla, y
verifica que el resultado coincida con el archivo entregado.

Tres limitaciones se documentan explícitamente en vez de disimularse:

- **Los studies de Optuna viven en el bucket, no en el repo.** Si el study existe, el
  notebook lo levanta y no vuelve a buscar: ese es el camino reproducible de verdad. Si
  no existe, rehace la búsqueda; el sampler está sembrado, pero eso no es una garantía
  formal frente a cambios de versión de las librerías.
- **Los argumentos exactos de `lgbmprod_l1_chico` no quedaron registrados.** Se corrió
  con un `--nombre` explícito, así que la carpeta no los codifica. Están reconstruidos de
  la bitácora y el script los verifica comparando el CSV regenerado contra el guardado.
- **Los datos crudos no están versionados** en el repositorio; se descargan al bucket.

La lección para un próximo trabajo es que la convención de codificar la configuración en
el nombre del archivo —que funcionó muy bien para todo lo demás— se rompe en cuanto se
permite un `--nombre` manual. Conviene que el nombre manual sea un *sufijo* del nombre
derivado, no un reemplazo.

---

## 8. Conclusiones

**Sobre el problema.** La demanda intermitente a nivel producto-cliente, con horizonte de
dos meses y evaluación agregada por producto, resultó ser un régimen donde la capacidad
del modelo no paga. Un OLS de 13 parámetros con doce lags —un ciclo estacional exacto— es
competitivo con un pipeline de 600 features, y la combinación de ambos es mejor que
cualquiera de los dos.

**Sobre el método.** El resultado más transferible es §5.1: en un problema con un cambio
de régimen entre validación y producción, la métrica interna puede ser no solo poco
informativa sino levemente anticorrelacionada. Detectarlo temprano —midiendo la
correlación entre la métrica interna y la real, en vez de asumirla— cambió por completo
la forma de trabajo y evitó seguir optimizando en la dirección equivocada.

**Sobre el resultado.** El puntaje quedó en 0.229 contra el 0.17 mencionado por la
cátedra como referencia alcanzable. La brecha es sustancial y no se cerró. Las cinco
hipótesis que se probaron para cerrarla fallaron, tres de ellas monótonamente, lo que
sugiere que la diferencia no está en las palancas exploradas sino en algo estructural que
no se identificó: probablemente en el planteo del problema —cómo se define la unidad de
observación y el target— más que en el modelado.

**Líneas abiertas.** Por qué `regression_l1` destruye a granularidad producto-cliente y
no a nivel producto quedó sin explicación después de descartar dos hipótesis con
mediciones. Y la transformación logarítmica del target, natural para ventas lognormales,
quedó sin probar por falta de tiempo.
