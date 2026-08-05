# -*- coding: utf-8 -*-
"""labo3 — pipeline unico para el laboratorio de implementacion III.

Contiene la infraestructura de rutas y checkpoints, mas la logica de
preprocesamiento y feature engineering. Las funciones de negocio estan tomadas
VERBATIM de los notebooks 01 y 02 ya validados: si algo falla en el modelado, no
es porque estas cuentas hayan cambiado.

Uso tipico desde el notebook:

    import labo3
    cfg  = labo3.PreprocessingConfig(group_mode="A", ...)
    path = labo3.etapa_preprocesar(cfg)         # saltea si ya existe
    path = labo3.etapa_fe(path, max_lags=24)    # idem
"""
import json
import logging
import os
import shutil
import sys
from time import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("labo3")


# ══════════════════════════════════════════════════════════════════════════
# RUTAS
# ══════════════════════════════════════════════════════════════════════════
def resolver_bucket() -> Path:
    """VM de Google Cloud -> ~/buckets/b1 | Colab -> /content/buckets/b1.
    LABO3_BUCKET pisa todo, para correr fuera de la nube."""
    env = os.environ.get("LABO3_BUCKET")
    if env:
        p = Path(env).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        return p
    for cand in (Path.home() / "buckets" / "b1", "/content/buckets/b1", "/home/ds/buckets/b1"):
        if Path(cand).is_dir():
            return Path(cand)
    raise RuntimeError(
        "No encontre el bucket. En la VM tiene que estar montado en ~/buckets/b1. "
        "Fuera de la nube: os.environ['LABO3_BUCKET'] = '/ruta/al/bucket'")


BUCKET   = resolver_bucket()
DIR_RAW  = BUCKET / "datasets"
DIR_PREP = BUCKET / "datasets" / "preprocesado"
RUTA_FE  = BUCKET / "datasets_fe"
RUTA_EXP = BUCKET / "exp"
DIR_DB   = RUTA_EXP / "optuna_db"
for _d in (DIR_RAW, DIR_PREP, RUTA_FE, RUTA_EXP, DIR_DB):
    _d.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════════════════
# Todo lo caro se escribe al bucket y se puede retomar. Dos reglas:
#
#   1. ESCRITURA ATOMICA. Se escribe a un .tmp y recien al terminar se renombra.
#      Si Google mata la spot en el medio, el nombre final nunca llega a existir
#      con contenido parcial. (Sin esto quedan parquet truncados que parecen
#      completos y revientan horas despues, en otra etapa.)
#
#   2. VALIDACION AL LEER. Un parquet valido termina con los bytes "PAR1". Antes
#      de dar por bueno un checkpoint se verifica; si esta roto, se recalcula.
# ══════════════════════════════════════════════════════════════════════════
def parquet_valido(path: Path) -> bool:
    """True si el archivo existe y termina con el magic 'PAR1' de parquet."""
    try:
        if not path.exists() or path.stat().st_size < 12:
            return False
        with open(path, "rb") as f:
            f.seek(-4, os.SEEK_END)
            return f.read(4) == b"PAR1"
    except OSError:
        return False


def escribir_parquet(df: pl.DataFrame, path: Path) -> Path:
    """Escritura atomica: .tmp -> rename. Nunca deja un parquet a medias."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp)
    tmp.replace(path)
    return path


def escribir_json(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=2,
              ensure_ascii=False, default=float)
    tmp.replace(path)
    return path


def limpiar_tmp(raiz: Path = None) -> int:
    """Borra los .tmp que hayan quedado de una corrida interrumpida."""
    n = 0
    for d in ([raiz] if raiz else [DIR_PREP, RUTA_FE, RUTA_EXP]):
        for p in Path(d).rglob("*.tmp"):
            p.unlink(missing_ok=True)
            n += 1
    return n


def etapa(nombre: str, salida: Path, fn, forzar: bool = False):
    """Corre `fn()` solo si `salida` no existe o no es valida.

    Devuelve (path, se_recalculo). Es el corazon del sistema de reanudacion:
    despues de que Google mate la spot, se vuelve a correr TODO el notebook y
    cada etapa ya hecha se saltea sola.
    """
    ok = parquet_valido(salida) if salida.suffix == ".parquet" else salida.exists()
    if ok and not forzar:
        mb = salida.stat().st_size / 1024**2
        logger.info(f"[{nombre}] ya existe, se saltea  ({salida.name}, {mb:,.1f} MB)")
        return salida, False
    if salida.exists() and not ok:
        logger.warning(f"[{nombre}] el checkpoint existe pero esta CORRUPTO -> se recalcula")
        salida.unlink(missing_ok=True)
    logger.info(f"[{nombre}] calculando -> {salida.name}")
    t0 = time()
    fn()
    logger.info(f"[{nombre}] listo en {time()-t0:,.0f}s")
    return salida, True


def descargar_datasets(url_origen: str = "https://storage.googleapis.com/open-courses/austral2026-5da5/labo3/",
                       archivos=("sell-in.txt.gz", "tb_productos.txt", "tb_stocks.txt",
                                 "product_id_apredecir201912.txt")) -> None:
    """Baja los crudos que falten. Atomica, por si muere la maquina en el medio."""
    import urllib.request
    for a in archivos:
        dst = DIR_RAW / a
        if dst.exists() and dst.stat().st_size > 0:
            logger.info(f"[datos] ya estaba: {a}")
            continue
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        logger.info(f"[datos] bajando {a} ...")
        urllib.request.urlretrieve(url_origen + a, tmp)
        tmp.replace(dst)
        logger.info(f"[datos] {a}: {dst.stat().st_size/1024**2:,.1f} MB")


def instalar_kaggle() -> bool:
    """Deja ~/.kaggle/kaggle.json con permisos 600. True si quedo listo."""
    dst = Path.home() / ".kaggle" / "kaggle.json"
    if dst.exists():
        dst.chmod(0o600)
        return True
    for cand in (BUCKET / "kaggle" / "kaggle.json", BUCKET / "kaggle.json"):
        if cand.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(cand, dst)
            dst.chmod(0o600)
            logger.info(f"[kaggle] credenciales instaladas desde {cand}")
            return True
    logger.warning("[kaggle] no encontre kaggle.json en el bucket (solo hace falta para submitear)")
    return False



# ══════════════════════════════════════════════════════════════════════════
# ETAPA 1 — PREPROCESAMIENTO   (verbatim de 01_Preprocesamiento)
# ══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PreprocessingConfig:
    """Clase inmutable para la parametrización absoluta de experimentos."""

    # Rutas de Ingesta (Compatibles con Google Colab / Local)
    base_dir: Path = DIR_RAW
    sell_in_path: Path = DIR_RAW / "sell-in.txt.gz"
    productos_path: Path = DIR_RAW / "tb_productos.txt"
    apredecir_path: Path = DIR_RAW / "product_id_apredecir201912.txt"
    output_dir: Path = DIR_PREP

    # [Experimento 1] Modo de Agrupamiento:
    # 'A': por Cliente-Producto-Mes | 'B': por Producto-Mes
    group_mode: Literal["A", "B"] = "A"
    default_customer_id: int = 0

    # [Experimento 2] Modo de completar faltantes (Post-Densificación)
    # 'zero': Completa con 0.0 e imita el comportamiento físico de no-venta
    # 'null': Preserva valores NA nativos
    missing_strategy: Literal["zero", "null"] = "zero"

    # [Experimento 3] Modo de Densificación Temporal
    # 'full': Malla cruzada total desde el inicio al fin del tiempo general
    # 'lifecycle': Respeta la vida comercial de cada producto individual
    densify_strategy: Literal["full", "lifecycle"] = "lifecycle"

    # Límites cronológicos del Dataset Histórico Observado
    timeline_start: str = "2017-01-01"
    timeline_end: str = "2019-12-01"

    # Filtro opcional: Mantener solo productos requeridos para el Target final de Negocio (los 780 que hay que predecir)
    filter_target_products_only: bool = True

    # [Muestreo] Quedarse solo con los N productos de mayor volumen (tn total).
    # None = todos. Sirve para validar el pipe entero en minutos y con poca RAM:
    # el subconjunto queda anotado en el nombre del archivo (_smplN), asi que NO
    # pisa al dataset completo y 02/03 lo distinguen solos.
    # La seleccion es deterministica (top-N por tn), no aleatoria -> reproducible.
    sample_n_products: Optional[int] = None

    def get_output_filename(self) -> str:
        """Genera de forma determinista el nombre del archivo Parquet resultante."""
        grp = "grpClienteProducto" if self.group_mode == "A" else "grpProducto"
        missing = "fill0" if self.missing_strategy == "zero" else "fillNA"
        dense = "denseFull" if self.densify_strategy == "full" else "denseLife"
        tgt = "_tgtFilter" if self.filter_target_products_only else ""
        smpl = f"_smpl{self.sample_n_products}" if self.sample_n_products else ""
        return f"preprocesado_{grp}_{missing}_{dense}{tgt}{smpl}.parquet"


class PreprocessingValidationError(Exception):
    """Excepción de control para detener el pipeline ante fallos críticos de sanidad."""
    pass


def validate_input_integrity(df_sell_in: pl.DataFrame, df_prod: pl.DataFrame) -> None:
    """Valida la integridad referencial y estructural de los insumos."""
    # Validación de duplicados en catálogo
    if df_prod.select("product_id").is_duplicated().any():
        raise PreprocessingValidationError("Existen llaves 'product_id' duplicadas en el maestro de productos.")

    # Identificar huérfanos sin detener, enviando advertencia controlada al log
    orphans = df_sell_in.select("product_id").unique().join(
        df_prod.select("product_id").unique(), on="product_id", how="left_anti"
    )
    if not orphans.is_empty():
        logger.warning(f"Calidad Alerta: Existen {orphans.height} product_ids transaccionados que faltan en el catálogo.")


def read_and_clean_sources(config: PreprocessingConfig) -> Tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
    """Carga los archivos en modo diferido (Lazy), unificando los tipos de datos temporales."""
    logger.info("Iniciando fase de lectura de fuentes transaccionales...")

    # Lectura diferida con delimitador correcto
    lf_sell_in = pl.scan_csv(config.sell_in_path, separator="\t")
    lf_productos = pl.scan_csv(config.productos_path, separator="\t")
    lf_apredecir = pl.scan_csv(config.apredecir_path, separator="\t")

    # Estandarización del eje temporal de Periodo (int YYYYMM -> pl.Date)
    lf_sell_in = lf_sell_in.with_columns(
        pl.format("{}-01", pl.col("periodo"))
        .str.to_date("%Y%m-01")
        .alias("periodo")
    )

    return lf_sell_in, lf_productos, lf_apredecir


def apply_aggregation_mode(lf: pl.LazyFrame, config: PreprocessingConfig) -> pl.LazyFrame:
    """[Experimento 1]: Agrupación de Demanda y Generación Consistente del Agrupacion_ID."""
    logger.info(f"Ejecutando agregación espacial en Modo: {config.group_mode}")

    # Lista base de métricas numéricas agregables
    metrics = ["tn", "cust_request_qty", "cust_request_tn", "plan_precios_cuidados"]
    agg_exprs = [pl.col(m).sum().alias(m) for m in metrics]

    if config.group_mode == "A":
        # Alternativa A: Granularidad Fina de Cliente
        lf_agg = lf.group_by(["periodo", "customer_id", "product_id"]).agg(agg_exprs)
    elif config.group_mode == "B":
        # Alternativa B: Desestimar clientes, seteando la convención de Customer_ID = 0
        lf_agg = lf.group_by(["periodo", "product_id"]).agg(agg_exprs).with_columns(
            pl.lit(config.default_customer_id).alias("customer_id")
        )
    else:
        raise ValueError(f"Modo de grupo inválido: {config.group_mode}")

    # Firma obligatoria para consistencia de Series Temporales aguas abajo
    lf_agg = lf_agg.with_columns(
        (pl.col("customer_id").cast(pl.Int64) * 100000 + pl.col("product_id").cast(pl.Int64)).alias("Agrupacion_ID")
    )
    return lf_agg


def apply_densification_strategy(lf: pl.LazyFrame, config: PreprocessingConfig) -> pl.LazyFrame:
    """[Experimento 3]: Algoritmo de densificación temporal avanzada."""
    logger.info(f"Construyendo rejilla temporal bajo estrategia: {config.densify_strategy}")

    start_dt = pl.lit(config.timeline_start).str.to_date()
    end_dt = pl.lit(config.timeline_end).str.to_date()

    # Universo único observado de agrupaciones combinatorias espaciales
    unique_series = lf.select(["Agrupacion_ID", "customer_id", "product_id"]).unique()

    if config.densify_strategy == "full":
        # Estrategia A: Producto Cartesiano Completo
        # Crear un LazyFrame a partir de la serie de fechas
        period_series_lf = pl.LazyFrame({"periodo": pl.date_range(start_dt, end_dt, interval="1mo", eager=True)}).lazy()
        grid = unique_series.join(
            period_series_lf,
            how="cross"
        )
    elif config.densify_strategy == "lifecycle":
        # Estrategia B: Delimitación paramétrica del ciclo de vida del SKU
        lifecycle_bounds = lf.group_by("product_id").agg([
            pl.col("periodo").min().alias("birth_date"),
            pl.col("periodo").max().alias("death_date")
        ])

        # Restricciones estrictas de frontera: No asumir muerte/nacimiento en los extremos del dataset
        lifecycle_bounds = lifecycle_bounds.with_columns([
            pl.when(pl.col("birth_date") == start_dt).then(start_dt).otherwise(pl.col("birth_date")).alias("birth_date"),
            pl.when(pl.col("death_date") == end_dt).then(end_dt).otherwise(pl.col("death_date")).alias("death_date")
        ])

        # Explotar la ventana de tiempo válida exclusiva de cada producto
        grid = unique_series.join(lifecycle_bounds, on="product_id", how="inner")
        grid = grid.with_columns(
            pl.date_ranges(pl.col("birth_date"), pl.col("death_date"), interval="1mo").alias("periodo")
        ).explode("periodo").drop(["birth_date", "death_date"])

    # Unión izquierda para rellenar los valores observados en la rejilla generada
    dense_lf = grid.join(lf, on=["Agrupacion_ID", "customer_id", "product_id", "periodo"], how="left")
    return dense_lf


def apply_imputation_strategy(lf: pl.LazyFrame, config: PreprocessingConfig) -> pl.LazyFrame:
    """[Experimento 2]: Tratamiento paramétrico de celdas vacías post-densificación."""
    logger.info(f"Tratando nulos resultantes con estrategia: {config.missing_strategy}")

    if config.missing_strategy == "zero":
        lf = lf.with_columns([
            pl.col("tn").fill_null(0.0),
            pl.col("cust_request_qty").fill_null(0),
            pl.col("cust_request_tn").fill_null(0.0),
            pl.col("plan_precios_cuidados").fill_null(0)
        ])
    elif config.missing_strategy == "null":
        # Mantener nulos nativos para procesamiento avanzado posterior o imputación interna de LightGBM
        pass
    return lf


def run_pipeline(config: PreprocessingConfig) -> Path:
    """Orquestador maestro optimizado para cómputo por flujos (Streaming Execution)."""
    t_start = time()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingesta diferida
    lf_sell_in, lf_productos, lf_apredecir = read_and_clean_sources(config)

    # Filtro opcional preventivo para acotar el experimento a las demandas objetivo (los 780 productos que hay que predecir)
    if config.filter_target_products_only:
        target_ids = lf_apredecir.select("product_id").unique()
        lf_sell_in = lf_sell_in.join(target_ids, on="product_id", how="inner")

    # Muestreo opcional: los N productos de mayor volumen. Se aplica DESPUES del
    # filtro de target, asi el subconjunto sale de los productos que ya quedaron.
    if config.sample_n_products is not None:
        top_ids = (
            lf_sell_in.group_by("product_id")
            .agg(pl.col("tn").sum().alias("_tn_total"))
            .sort("_tn_total", descending=True)
            .head(config.sample_n_products)
            .select("product_id")
        )
        lf_sell_in = lf_sell_in.join(top_ids, on="product_id", how="inner")
        logger.info(f"MUESTREO ACTIVO: solo los {config.sample_n_products} productos de mayor tn")

    # 2. Encadenamiento del grafo de ejecución lógica (Lazy Execution)
    processed_graph = (
        lf_sell_in
        .pipe(apply_aggregation_mode, config=config)
        .pipe(apply_densification_strategy, config=config)
        .pipe(apply_imputation_strategy, config=config)
    )

    # Pegar catálogo de productos antes de la salida definitiva
    processed_graph = processed_graph.join(lf_productos, on="product_id", how="left")

    # Mostrar dimensiones antes de guardar
    logger.info(f"Dimensiones del dataset procesado (filas, columnas) antes de guardar: {processed_graph.collect().shape}")

    # 3. Compilación Física de Resultados usando Motor Streaming de Polars
    out_file = config.output_dir / config.get_output_filename()
    logger.info(f"Compilando grafo y volcando datos en formato Parquet hacia: {out_file}")

    # .sink_parquet procesa de forma eficiente sin cargar todo el dataset simultáneamente en RAM
    processed_graph.sink_parquet(out_file)

    # 4. Validaciones Finales Ansiosas (Eager Verification)
    df_final = pl.read_parquet(out_file)

    # Validación de unicidad de llave primaria de series de tiempo
    is_duplicated = df_final.select(["Agrupacion_ID", "periodo"]).is_duplicated().any()
    if is_duplicated:
        raise PreprocessingValidationError("Error de Consistencia: Se generaron llaves duplicadas (Agrupacion_ID, periodo).")

    # 5. Métricas de Control Informativo (Logging)
    duration = time() - t_start
    logger.info("==================================================")
    logger.info(f" PROCESAMIENTO COMPLETADO EXITOSAMENTE")
    logger.info(f" Archivo generado: {config.get_output_filename()}")
    logger.info(f" Filas finales registradas: {df_final.height}")
    logger.info(f" Series temporales únicas: {df_final['Agrupacion_ID'].n_unique()}")
    logger.info(f" Tiempo consumido por el Pipeline: {duration:.2f} segundos")
    logger.info("==================================================")

    return out_file # Retornar la ruta del archivo generado



# ══════════════════════════════════════════════════════════════════════════
# ETAPA 2 — FEATURE ENGINEERING   (verbatim de 02_FE)
# ══════════════════════════════════════════════════════════════════════════
def lagear_dataset(df, granularidad=None, periodo_final=201912, n_periodos=42,
                   max_lags=24, horizonte=2, keys=None):
    """
    Arma la tabla supervisada (panel): una fila por (clave, periodo actual).

    Para CADA periodo t del rango [periodo_final - (n_periodos - 1) .. periodo_final]
    (los ultimos `n_periodos` meses hasta `periodo_final`) genera, por clave:
      - tn0..tn{max_lags}: tn del propio t y de los meses anteriores (t, t-1, ..., t-max_lags).
      - clase_tn        : tn `horizonte` meses adelante -> tn(t + horizonte). Default 2.
      - ppc0            : plan_precios_cuidados en el propio t (si viene la columna).
      - periodo         : el periodo actual t (clave temporal de cada fila).

    A diferencia del achatado viejo (un unico t0 fijo), aca t recorre un rango: cada
    mes es su propio "periodo actual". Solo se genera fila para los (clave, t) con
    observacion en t; los lags que caen fuera de la serie quedan NULL. Los 0 dentro de
    la vida vienen del completado del modulo 1.

    Los lags se calculan con shift sobre la serie ordenada por mes: asume que la serie
    es densa mensual dentro de la vida de cada clave (garantizado por el completado del
    modulo 1: denseLife / denseFull), de modo que un shift de k filas == k meses.

    Params
    ------
    df           : long con periodo (AAAAMM entero), tn y las columnas de `keys`.
    granularidad : "p" (producto) o "pc" (producto-cliente). Atajo de `keys`.
    periodo_final: ultimo periodo actual (AAAAMM). Default 201910.
    n_periodos   : cuantos periodos actuales generar hacia atras. Default 42.
    max_lags     : cantidad de lags -> tn0..tn{max_lags}.
    horizonte    : cuantos meses adelante es la clase. Default 2.
    keys         : columnas de agrupacion; si viene, gana sobre `granularidad`.
    """
    if keys is None:
        if granularidad == "pc":
            keys = ["product_id", "customer_id"]
        elif granularidad == "p":
            keys = ["product_id"]
        else:
            raise ValueError(
                f"granularidad no soportada: {granularidad!r}. Usa 'p' / 'pc', o pasa keys=[...]"
            )

    m_final = (periodo_final // 100) * 12 + (periodo_final % 100)
    m_ini = m_final - (n_periodos - 1)
    has_ppc = "plan_precios_cuidados" in df.columns

    # Serie por (clave, periodo), ordenada cronologicamente dentro de cada clave.
    # En "p" agrega tn sobre los clientes; en "pc" es 1:1.
    aggs = [pl.col("tn").sum().alias("tn")]
    if has_ppc:
        aggs.append(pl.col("plan_precios_cuidados").max().alias("ppc"))
    serie = (
        df.group_by(keys + ["periodo"]).agg(aggs)
          .with_columns(((pl.col("periodo") // 100) * 12 + (pl.col("periodo") % 100)).alias("m"))
          .sort(keys + ["m"])
    )

    # Lags por shift (serie densa -> shift de k filas == k meses). shift(k) mira k meses
    # atras; shift(-horizonte) mira horizonte meses adelante (la clase). En los bordes
    # (previo al first_sell / posterior al ultimo mes) shift devuelve NULL. over(keys)
    # evita que el shift cruce de una clave a otra.
    exprs = [pl.col("tn").shift(k).over(keys).alias(f"tn{k}") for k in range(max_lags + 1)]
    exprs.append(pl.col("tn").shift(-horizonte).over(keys).alias("clase_tn"))
    serie = serie.with_columns(exprs)

    # Panel: periodos "actuales" (rango) con observacion propia (tn0 no nulo, descarta
    # meses fuera de la vida en datasets densos tipo denseFull; en denseLife es no-op).
    panel = serie.filter(
        (pl.col("m") >= m_ini) & (pl.col("m") <= m_final) & pl.col("tn0").is_not_null()
    )

    extra = []
    if has_ppc:
        panel = panel.with_columns(pl.col("ppc").alias("ppc0"))
        extra = ["ppc0"]

    # Metadata 1:1 con la clave (atributos del producto): la arrastramos.
    meta_cols = [c for c in ["Agrupacion_ID", "cat1", "cat2", "cat3", "brand"]
                 if c in df.columns and c not in keys]
    id_cols = []
    if meta_cols:
        meta = df.select(keys + meta_cols).unique(subset=keys)
        panel = panel.join(meta, on=keys, how="left")
        id_cols = meta_cols

    lag_cols = [f"tn{k}" for k in range(max_lags + 1)]
    return (
        panel
        .select(id_cols + keys + ["periodo"] + lag_cols + extra + ["clase_tn"])
        .sort(keys + ["periodo"])
    )


def normalizar_achatado(df_achatado, metodo="recta", norm_lags=None,
                        prefix="", clase_col="clase_tn", b0_name="B0", b1_name="B1"):
    """
    Normaliza fila a fila las columnas {prefix}tn* (y opcionalmente {clase_col})
    de df_achatado.

    Los parametros de normalizacion (B0, B1) se calculan SOLO con la ventana
    {prefix}tn0..{prefix}tn{norm_lags} (nunca con la clase -> evita leakage) y
    luego se aplican a todos los {prefix}tn* y a la clase, generando
    {prefix}tn0_norm..{prefix}tnN_norm y {clase_col}_norm.

    Params
    ------
    metodo:
      "recta"   -> ajuste lineal por minimos cuadrados sobre la ventana.
                   B0 = ordenada al origen, B1 = pendiente.
                   norm = valor - (B0 + B1 * lag)   (la clase esta en lag = -2)
      "zscore"  -> B0 = media, B1 = desvio.   norm = (valor - B0) / B1
      "minmax"  -> B0 = min,   B1 = max-min.  norm = (valor - B0) / B1
      "media"   -> B0 = 0,     B1 = media.    norm = valor / B1
    norm_lags:
      hasta que lag usar para calcular B0/B1 (ventana {prefix}tn0..{prefix}tn{norm_lags}).
      None = usa todos los lags disponibles.
    prefix:
      prefijo de las columnas de lags a normalizar (ej. "" para pc, "p_" para producto).
    clase_col:
      nombre de la columna de clase a normalizar; None si no hay clase (ej. features
      de producto, donde no se normaliza ninguna clase).
    b0_name, b1_name:
      nombres con los que se guardan los parametros de normalizacion (ej. "B0"/"B1"
      para pc, "p_B0"/"p_B1" para producto).

    Devuelve df_achatado + columnas b0_name, b1_name, {prefix}tn*_norm y (si aplica)
    {clase_col}_norm.
    """
    p = len(prefix)
    lag_cols = sorted(
        [c for c in df_achatado.columns if c.startswith(prefix + "tn") and c[p + 2:].isdigit()],
        key=lambda c: int(c[p + 2:]),
    )
    n_max = int(lag_cols[-1][p + 2:])
    if norm_lags is None:
        norm_lags = n_max
    fit_cols = [f"{prefix}tn{k}" for k in range(norm_lags + 1)]

    if metodo == "recta":
        # Minimos cuadrados sobre la ventana, ignorando los nulls
        n   = pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Float64) for c in fit_cols])
        Sy  = pl.sum_horizontal([pl.col(c) for c in fit_cols])
        Sx  = pl.sum_horizontal([pl.when(pl.col(c).is_not_null()).then(float(k)).otherwise(0.0)
                                 for k, c in enumerate(fit_cols)])
        Sxx = pl.sum_horizontal([pl.when(pl.col(c).is_not_null()).then(float(k * k)).otherwise(0.0)
                                 for k, c in enumerate(fit_cols)])
        Sxy = pl.sum_horizontal([pl.when(pl.col(c).is_not_null()).then(pl.col(c) * float(k)).otherwise(0.0)
                                 for k, c in enumerate(fit_cols)])
        denom = n * Sxx - Sx * Sx
        slope = pl.when(denom != 0).then((n * Sxy - Sx * Sy) / denom).otherwise(0.0)
        inter = pl.when(n > 0).then((Sy - slope * Sx) / n).otherwise(None)

        df = df_achatado.with_columns(inter.alias(b0_name), slope.alias(b1_name))

        # norm = valor - (B0 + B1 * lag);  la clase esta en lag = -2 (t0 + 2)
        exprs = [
            (pl.col(c) - (pl.col(b0_name) + pl.col(b1_name) * float(int(c[p + 2:])))).alias(f"{c}_norm")
            for c in lag_cols
        ]
        if clase_col is not None:
            exprs.append((pl.col(clase_col) - (pl.col(b0_name) + pl.col(b1_name) * (-2.0))).alias(f"{clase_col}_norm"))
        return df.with_columns(exprs)

    # --- metodos afines: norm = (valor - B0) / B1 ---
    if metodo == "zscore":
        B0 = pl.mean_horizontal(fit_cols)
        B1 = pl.concat_list(fit_cols).list.std()
    elif metodo == "minmax":
        B0 = pl.min_horizontal(fit_cols)
        B1 = pl.max_horizontal(fit_cols) - pl.min_horizontal(fit_cols)
    elif metodo == "media":
        B0 = pl.lit(0.0)
        B1 = pl.mean_horizontal(fit_cols)
    else:
        raise ValueError(f"metodo no soportado: {metodo}")

    df = df_achatado.with_columns(B0.alias(b0_name), B1.alias(b1_name))
    # B1 seguro para no dividir por 0 ni por null (series constantes / sin datos)
    b1_safe = pl.when((pl.col(b1_name) == 0) | pl.col(b1_name).is_null()).then(1.0).otherwise(pl.col(b1_name))
    exprs = [((pl.col(c) - pl.col(b0_name)) / b1_safe).alias(f"{c}_norm") for c in lag_cols]
    if clase_col is not None:
        exprs.append(((pl.col(clase_col) - pl.col(b0_name)) / b1_safe).alias(f"{clase_col}_norm"))
    return df.with_columns(exprs)


def desnormalizar(df, col_norm, metodo, lag=-2, alias="pred_tn"):
    """
    Recupera la escala original (toneladas) de una columna normalizada,
    usando los parametros B0/B1 guardados por normalizar_achatado.
    Es el inverso exacto de normalizar_achatado.

    Params
    ------
    df       : DataFrame que tiene las columnas B0, B1 y col_norm.
    col_norm : columna normalizada a desnormalizar (ej. la prediccion del modelo).
    metodo   : el MISMO metodo usado al normalizar ("recta"|"zscore"|"minmax"|"media").
    lag      : posicion temporal de la columna respecto de t0.
               clase_tn (y su prediccion) esta en lag = -2 (t0 + 2); tn{k} esta en lag = k.
    alias    : nombre de la columna de salida.

    Devuelve df + columna 'alias' en escala original.
    """
    if metodo == "recta":
        # inverso de: norm = valor - (B0 + B1 * lag)
        valor = pl.col(col_norm) + (pl.col("B0") + pl.col("B1") * float(lag))
    elif metodo in ("zscore", "minmax", "media"):
        # inverso de: norm = (valor - B0) / B1  (con el mismo B1 seguro del forward)
        b1_safe = pl.when((pl.col("B1") == 0) | pl.col("B1").is_null()).then(1.0).otherwise(pl.col("B1"))
        valor = pl.col(col_norm) * b1_safe + pl.col("B0")
    else:
        raise ValueError(f"metodo no soportado: {metodo}")
    return df.with_columns(valor.alias(alias))


def agregar_deltas(df_norm, salto=2, prefix="", clase_col="clase_tn"):
    """
    Agrega columnas de delta (saltos de 'salto' periodos) sobre las columnas
    NORMALIZADAS. No elimina ninguna columna, solo agrega.

      {prefix}tn{k}_delta = {prefix}tn{k}_norm - {prefix}tn{k+salto}_norm  (k = 0 .. maxlag-salto)
      {clase_col}_delta   = {clase_col}_norm - {prefix}tn0_norm            (cambio a 2 periodos -> target)

    tn0_delta es el delta t0 - t2 (el cambio en el ti mas reciente).
    Para este problema salto=2 porque se predice a 2 periodos.

    Params
    ------
    prefix:
      prefijo de las columnas normalizadas a deltear (ej. "" para pc, "p_" para producto).
    clase_col:
      nombre base de la clase a deltear; None si no hay clase (ej. nivel producto).
    """
    p = len(prefix)
    norm_lags = sorted(
        int(c[p + 2:-5]) for c in df_norm.columns
        if c.startswith(prefix + "tn") and c.endswith("_norm") and c[p + 2:-5].isdigit()
    )
    max_k = max(norm_lags)
    exprs = [
        (pl.col(f"{prefix}tn{k}_norm") - pl.col(f"{prefix}tn{k + salto}_norm")).alias(f"{prefix}tn{k}_delta")
        for k in range(0, max_k - salto + 1)
    ]
    # Delta de la clase: valor de la clase normalizada menos t0 normalizado
    if clase_col is not None:
        exprs.append((pl.col(f"{clase_col}_norm") - pl.col(f"{prefix}tn0_norm")).alias(f"{clase_col}_delta"))
    return df_norm.with_columns(exprs)


def agregar_deltas2(df_norm, prefix=""):
    """
    Agrega columnas de delta calculadas como la diferencia entre tn0_norm 
    y cada tnk_norm (k >= 1) sobre las columnas NORMALIZADAS. 
    No elimina ninguna columna, solo agrega.

      {prefix}tn_delta{k} = {prefix}tn0_norm - {prefix}tn{k}_norm  (k = 1 .. maxlag)

    Params
    ------
    prefix:
      prefijo de las columnas normalizadas a deltear (ej. "" para pc, "p_" para producto).
    """
    p = len(prefix)
    norm_lags = sorted(
        int(c[p + 2:-5]) for c in df_norm.columns
        if c.startswith(prefix + "tn") and c.endswith("_norm") and c[p + 2:-5].isdigit()
    )
    max_k = max(norm_lags)
    exprs = [
        (pl.col(f"{prefix}tn0_norm") - pl.col(f"{prefix}tn{k}_norm")).alias(f"{prefix}tn_delta{k}")
        for k in range(1, max_k + 1)
    ]
    
    return df_norm.with_columns(exprs)


def agregar_racha(df, prefix="", alias=None, suffix="_delta"):
    """
    Agrega una columna de RACHA: hace cuantos periodos seguidos viene subiendo (+)
    o bajando (-) la serie. Se lee sobre el signo de las columnas
    {prefix}tn*{suffix} que ya dejo agregar_deltas. No elimina nada, solo agrega.

      {prefix}tn_racha = direccion * largo

    Arranca en {prefix}tn0{suffix} (el cambio mas reciente) y avanza hacia los lags
    mas viejos mientras el signo no se de vuelta:
      - el signo del primer delta != 0 fija la direccion de la racha
      - un delta de signo contrario la corta
      - un delta = 0 (empate) NO la corta: la racha sigue viva y el empate suma
      - un delta nulo (el par todavia no existia) la corta

    Casos borde:
      - todos los deltas en 0 (serie plana, tipico de un producto sin ventas)
        -> racha = 0: no hay direccion.
      - {prefix}tn0{suffix} nulo (el par no llega a salto+1 periodos de historia)
        -> racha = null, para no confundir "sin historia" con "plano".
      - como el empate no corta, una serie que sube una vez y despues queda plana
        acumula racha alta (ej. deltas [1, 0, 0, 0] -> racha 4).

    Ojo: la unidad de la racha es el salto de agregar_deltas, no el mes. Con
    salto=2, racha=+3 son 3 deltas seguidos subiendo, o sea 6 meses de ventana.

    Params
    ------
    prefix : prefijo de las columnas a leer (ej. "" para pc, "p_" para producto,
             "c1_"/"c2_"/"c3_" para categoria).
    alias  : nombre de la columna de salida. Default: "{prefix}tn_racha".
    suffix : sufijo de las columnas a leer. Default "_delta".
    """
    p, s = len(prefix), len(suffix)
    ks = sorted(
        int(c[p + 2:-s]) for c in df.columns
        if c.startswith(f"{prefix}tn") and c.endswith(suffix) and c[p + 2:-s].isdigit()
    )
    if not ks:
        raise ValueError(f"no hay columnas {prefix}tn*{suffix} en el df")
    if alias is None:
        alias = f"{prefix}tn_racha"

    sgn = {k: pl.col(f"{prefix}tn{k}{suffix}").sign() for k in ks}

    # Direccion: signo del primer delta distinto de 0. Los nulls son de cola (los
    # meses previos al first_sell del par caen en los lags mas viejos), asi que el
    # primer delta != 0 siempre cae dentro del tramo con datos.
    # Sin direccion (todo empates, o todo nulo) -> 0, y la racha termina en 0.
    d = pl.coalesce([pl.when(sgn[k] != 0).then(sgn[k]) for k in ks]).fill_null(0).cast(pl.Int32)

    # Corta en el primer lag nulo o de signo contrario. sgn*d < 0 da falso para los
    # empates (producto 0) y para d = 0, que es justo lo que queremos.
    corta = {k: sgn[k].is_null() | ((sgn[k] * d) < 0) for k in ks}

    # Largo = indice del primer corte; si nunca corta, entra toda la ventana.
    largo = pl.min_horizontal([
        pl.when(corta[k]).then(pl.lit(k, dtype=pl.Int32)).otherwise(pl.lit(len(ks), dtype=pl.Int32))
        for k in ks
    ])

    return df.with_columns(
        pl.when(pl.col(f"{prefix}tn0{suffix}").is_null())
        .then(None)
        .otherwise(d * largo)
        .cast(pl.Int32)
        .alias(alias)
    )




# ══════════════════════════════════════════════════════════════════════════
# NOMBRES DE ARCHIVO
# ══════════════════════════════════════════════════════════════════════════
# El nombre arrastra TODAS las palancas. Es lo que permite que convivan varios
# experimentos en el mismo bucket sin pisarse, y que un checkpoint solo se reuse
# cuando de verdad corresponde a la misma configuracion.
def nombre_preprocesado(cfg: "PreprocessingConfig") -> str:
    return cfg.get_output_filename()


def nombre_fe(nombre_prep: str, max_lags: int, metodo: str, salto: int,
              sufijo: str = "") -> str:
    return f"{nombre_prep[:-8]}_{max_lags}lags_{metodo}_{salto}deltas{sufijo}.parquet"


def granularidad_de(nombre: str) -> str:
    """'pc' si el dataset es producto-cliente, 'p' si ya viene agregado por producto."""
    return "pc" if "grpClienteProducto" in nombre else "p"


# ══════════════════════════════════════════════════════════════════════════
# ETAPAS CON CHECKPOINT
# ══════════════════════════════════════════════════════════════════════════
def etapa_preprocesar(cfg: "PreprocessingConfig", forzar: bool = False) -> Path:
    """Etapa 1. Devuelve el path del parquet preprocesado, calculandolo solo si falta."""
    salida = DIR_PREP / cfg.get_output_filename()

    def _calc():
        run_pipeline(cfg)
        generado = cfg.output_dir / cfg.get_output_filename()
        if generado != salida:
            shutil.move(str(generado), str(salida))

    etapa("01 preprocesamiento", salida, _calc, forzar)
    return salida


def etapa_fe(path_prep: Path, max_lags: int = 24, metodo: str = "recta",
             salto: int = 2, periodo_final: int = 201912, n_periodos: int = 42,
             horizonte: int = 2, sufijo: str = "", forzar: bool = False) -> Path:
    """Etapa 2. Feature engineering completo sobre el parquet preprocesado.

    El cuerpo es el de 02_FE celda por celda, sin cambios de logica: lags ->
    normalizacion -> deltas -> complemento a nivel producto -> agregados por
    categoria y marca -> rachas.

    Devuelve el path del parquet con features, calculandolo solo si falta.
    """
    salida = RUTA_FE / nombre_fe(path_prep.name, max_lags, metodo, salto, sufijo)

    def _calc():
        # Nombres que esperan las celdas de 02_FE
        dataset  = path_prep.name
        MAX_LAGS = max_lags
        METODO   = metodo
        SALTO    = salto
        PERIODO_FINAL = periodo_final
        N_PERIODOS    = n_periodos
        HORIZONTE     = horizonte
        granularidad  = granularidad_de(dataset)
        logger.info(f"[02 fe] granularidad={granularidad} lags={MAX_LAGS} "
                    f"metodo={METODO} salto={SALTO}")

        df_final = pl.read_parquet(path_prep)
        # 01 guarda periodo como Date; el FE trabaja con enteros AAAAMM.
        # Es lo que hace la celda 3 de 02_FE antes de lagear.
        if df_final.schema["periodo"] == pl.Date:
            df_final = df_final.with_columns(
                (pl.col("periodo").dt.year() * 100 + pl.col("periodo").dt.month()).alias("periodo"))
        # df_aux: relectura del MISMO parquet, sin filtrar. De aca sale el total por
        # producto (suma sobre todos los clientes), que es senial nueva respecto de
        # la serie del par. Es lo que hacia la celda 30 de 02_FE.
        df_aux = pl.read_parquet(path_prep)

        # ── [02_FE celda 21] ──────────────────────────────────────────────────
        # --- Uso ---
        # Definir granularidad en funcion del nombre del dataset
        granularidad = "pc" if "grpClienteProducto" in dataset else "p"

        # Parametros del panel supervisado (compartidos con producto y categoria mas abajo):
        #   PERIODO_FINAL : ultimo "periodo actual" (mes desde el que se predice).
        #   N_PERIODOS    : cuantos periodos actuales generar hacia atras (rango).
        #   MAX_LAGS      : lags tn0..tn{MAX_LAGS} por periodo.
        #   HORIZONTE     : la clase es tn(periodo + HORIZONTE).
        PERIODO_FINAL = 201912 # dejar en 201912, la division de train/val/test se hace en el modulo 3
        N_PERIODOS    = 42
        # MAX_LAGS    = 24  (Definido arriba)
        HORIZONTE     = 2

        # Una fila por (clave, periodo) para los ultimos N_PERIODOS meses hasta PERIODO_FINAL.
        df_lags = lagear_dataset(
            df_final, granularidad,
            periodo_final=PERIODO_FINAL, n_periodos=N_PERIODOS,
            max_lags=MAX_LAGS, horizonte=HORIZONTE,
        )
        print(df_lags)

        # ── [02_FE celda 23] ──────────────────────────────────────────────────
        # --- Uso ---
        # metodo: "recta" | "zscore" | "minmax" | "media"
        # norm_lags: ventana tn0..tn{norm_lags} para calcular B0/B1 (None = todos los lags)
        df_norm = normalizar_achatado(df_lags, metodo=METODO, norm_lags=None)
        print(df_norm)

        # ── [02_FE celda 25] ──────────────────────────────────────────────────

        # --- Uso ---
        df_norm = agregar_deltas(df_norm, salto=SALTO)
        print(df_norm)

        # ── [02_FE celda 28] ──────────────────────────────────────────────────
        # --- Uso ---
        df_norm = agregar_deltas2(df_norm)
        print(df_norm)

        # ── [02_FE celda 31] ──────────────────────────────────────────────────
        if df_aux.schema["periodo"] == pl.Date:
            df_aux = df_aux.with_columns(
                (pl.col("periodo").dt.year() * 100 + pl.col("periodo").dt.month()).alias("periodo")
            )

        df_aux = df_aux.with_columns(
                pl.col("periodo").min().over("product_id").alias("first_sell_in_period")
            )

        print(df_aux)

        # ── [02_FE celda 32] ──────────────────────────────────────────────────
        # Colapsamos la dimension cliente: agregamos por periodo y producto.
        #   - cust_request_qty, cust_request_tn, tn -> suma (acumulado sobre los clientes)
        #   - first_sell_in_period                 -> minimo del grupo
        # plan_precios_cuidados NO se agrega a nivel producto: nos quedamos solo con la
        # columna original (nivel producto-cliente), que es la mas realista.
        df_aux = (
            df_aux
            .group_by(["periodo", "product_id"])
            .agg(
                pl.col("cust_request_qty").sum().alias("cust_request_qty"),
                pl.col("cust_request_tn").sum().alias("cust_request_tn"),
                pl.col("tn").sum().alias("tn"),
                pl.col("first_sell_in_period").min().alias("first_sell_in_period"),
                pl.col("cat1").first().alias("cat1"),
                pl.col("cat2").first().alias("cat2"),
                pl.col("cat3").first().alias("cat3"),
                pl.col("brand").first().alias("brand"),
            )
            .sort(["product_id", "periodo"])
        )
        print(df_aux)

        # ── [02_FE celda 33] ──────────────────────────────────────────────────
        # --- Features a nivel PRODUCTO (complemento "p" de la granularidad pc) ---
        # Solo tiene sentido con granularidad "pc": ahi cada fila es un par producto-cliente
        # y el total del producto (tn sumado sobre TODOS sus clientes) es una senial NUEVA
        # que complementa la serie del par. Con granularidad "p" el dataset YA viene agregado
        # por producto -> tn* == p_tn*, asi que este complemento seria redundante y lo salteamos.
        if granularidad == "pc":
            # Mismo panel (mismos PERIODO_FINAL / N_PERIODOS) pero a nivel producto, sumando tn
            # sobre TODOS los clientes. Sale de df_aux (mercado completo), NO de df_final
            # (filtrado) -> ahi el total por producto seria parcial. Se pega por
            # product_id + periodo (cada periodo actual del panel).
            prod_feats = lagear_dataset(
                df_aux, granularidad="p",
                periodo_final=PERIODO_FINAL, n_periodos=N_PERIODOS,
                max_lags=MAX_LAGS, horizonte=HORIZONTE,
            )

            lag_cols_p = [c for c in prod_feats.columns if c.startswith("tn") and c[2:].isdigit()]
            prod_feats = (
                prod_feats
                # brand tambien se dropea: df_norm ya la trae del panel base y el join la
                # duplicaria como 'brand_right' (columna de texto que despues rompe LightGBM).
                .drop(["clase_tn", "cat1", "cat2", "cat3", "brand"])
                # tn{k} -> p_tn{k} para no pisar los tn{k} de nivel producto-cliente
                .rename({c: f"p_{c}" for c in lag_cols_p})
                # mes sin ventas del producto -> 0 tn (a nivel producto un mes ausente = 0)
                .with_columns([pl.col(f"p_{c}").fill_null(0.0) for c in lag_cols_p])
            )

            # Pegamos las features de producto a cada fila (producto-cliente) de df_norm
            df_norm = df_norm.join(prod_feats, on=["product_id", "periodo"], how="left")

            print(df_norm.select(
                ["product_id", "customer_id", "periodo"] + [f"p_tn{k}" for k in range(MAX_LAGS + 1)]
            ))
        else:
            print(f"granularidad={granularidad!r}: sin complemento a nivel producto (tn* ya es producto).")

        # ── [02_FE celda 35] ──────────────────────────────────────────────────
        # Normalizamos los lags de PRODUCTO con la misma funcion/metodo que los pc,
        # pero referenciados al producto: parametros propios p_B0 / p_B1 y columnas
        # p_tn*_norm. Sin clase (clase_col=None): a nivel producto no normalizamos ninguna.
        # metodo debe ser el mismo que en la normalizacion pc de arriba ("media").
        # Solo con granularidad "pc": las columnas p_tn* solo existen en ese caso.
        if granularidad == "pc":
            df_norm = normalizar_achatado(
                df_norm, metodo=METODO, norm_lags=None,
                prefix="p_", clase_col=None, b0_name="p_B0", b1_name="p_B1",
            )

            print(df_norm.select(
                ["product_id", "customer_id", "periodo", "p_B0", "p_B1"]
                + [f"p_tn{k}_norm" for k in range(MAX_LAGS + 1)]
            ))

        # ── [02_FE celda 36] ──────────────────────────────────────────────────
        # Deltas de los lags de PRODUCTO: misma idea que los pc pero sobre p_tn*_norm.
        #   p_tn{k}_delta = p_tn{k}_norm - p_tn{k+2}_norm
        # Sin clase (clase_col=None): la clase vive solo a nivel producto-cliente.
        # Solo con granularidad "pc": las columnas p_tn*_norm solo existen en ese caso.
        if granularidad == "pc":
            df_norm = agregar_deltas(df_norm, salto=SALTO, prefix="p_", clase_col=None)

            print(df_norm.select(
                ["product_id", "customer_id", "periodo"]
                + [f"p_tn{k}_norm" for k in range(MAX_LAGS + 1)]
                + [f"p_tn{k}_delta" for k in range(MAX_LAGS - 1)]
            ))

        # ── [02_FE celda 37] ──────────────────────────────────────────────────
        # Deltas de los lags de PRODUCTO respecto al momento 0
        if granularidad == "pc":
            df_norm = agregar_deltas2(df_norm, prefix="p_")

            print(df_norm.select(
                ["product_id", "customer_id", "periodo"]
                + [f"p_tn{k}_norm" for k in range(MAX_LAGS + 1)]
                + [f"p_tn_delta{k}" for k in range(1,MAX_LAGS - 1)]
            ))

        # ── [02_FE celda 40] ──────────────────────────────────────────────────
        # --- Toneladas de la jerarquia (cat1, cat2, cat3) y por brand por periodo ---
        # Tamanio de mercado CONTEMPORANEO: para cada periodo actual t, el tn total de la
        # categoria en t (incluye al propio producto). Una feature por nivel.
        #
        # Se calcula sobre df_aux (mercado completo, tn ya sumado sobre los clientes), NO
        # sobre df_final (filtrado) -> ahi el total de la categoria seria parcial.

        CATS = ["cat1", "cat2", "cat3", "brand"]

        # cat1/cat2/cat3 ya vienen en df_norm desde el panel base (lagear_dataset las arrastra
        # como metadata 1:1 del producto), asi que no hace falta re-pegarlas aca.

        # tn total de cada categoria en CADA periodo; lo pegamos por (categoria, periodo).
        for c in CATS:
            tn_cat = df_aux.group_by([c, "periodo"]).agg(pl.col("tn").sum().alias(f"tn_total_{c}"))
            df_norm = df_norm.join(tn_cat, on=[c, "periodo"], how="left")

        # Una categoria sin ventas en ese periodo vendio 0 tn; no es un dato faltante.
        df_norm = df_norm.with_columns([pl.col(f"tn_total_{c}").fill_null(0.0) for c in CATS])

        # customer_id solo existe con granularidad "pc" (en "p" el panel es por producto).
        _id_cols = [c for c in ["product_id", "customer_id", "periodo"] if c in df_norm.columns]
        print(df_norm.select(
            _id_cols + CATS + [f"tn_total_{c}" for c in CATS]
        ))

        # ── [02_FE celda 41] ──────────────────────────────────────────────────
        # --- Panel por CATEGORIA (cat1, cat2, cat3) ---
        # Mismo pipeline que el nivel producto pero con keys=[cat]: reusamos las tres
        # funciones tal cual (lagear -> normalizar -> deltas), solo cambia el prefijo.
        #
        # Los tres niveles aportan: cat3 agrupa 5 productos en mediana (max 134), asi que su
        # serie no es la del producto suelto. Solo 12 de los 1233 productos (1%) son unicos en
        # su cat3; para esos c3_tn* si termina siendo igual a p_tn*.

        NIVELES = [("cat1", "c1_"), ("cat2", "c2_"), ("cat3", "c3_"), ("brand", "b_")]

        for cat, pref in NIVELES:
            # Serie mensual de la categoria: tn de todo el mercado sumado sobre sus productos.
            # Sale de df_aux (mercado completo), no de df_final (filtrado).
            df_cat = (
                df_aux
                .group_by([cat, "periodo"])
                .agg(pl.col("tn").sum().alias("tn"))
            )

            # Lags de la categoria por periodo. Sin clase: la clase vive a nivel producto-cliente.
            cat_feats = lagear_dataset(
                df_cat, keys=[cat],
                periodo_final=PERIODO_FINAL, n_periodos=N_PERIODOS,
                max_lags=MAX_LAGS, horizonte=HORIZONTE,
            ).drop("clase_tn")
            lag_cols_c = [c for c in cat_feats.columns if c.startswith("tn") and c[2:].isdigit()]
            # tn{k} -> {pref}tn{k} para no pisar los tn{k} de nivel producto-cliente
            cat_feats = cat_feats.rename({c: f"{pref}{c}" for c in lag_cols_c})

            df_norm = df_norm.join(cat_feats, on=[cat, "periodo"], how="left")

            # Dos fuentes de null, y las dos significan 0 tn (mismo criterio que p_tn*):
            #   - mes sin ventas de la categoria -> el lag queda null
            #   - categoria sin fila en ese periodo del panel -> el left join no la encuentra
            # Rellenamos despues del join y antes de normalizar para que B0/B1 no salgan nulos.
            df_norm = df_norm.with_columns(
                [pl.col(f"{pref}tn{k}").fill_null(0.0) for k in range(MAX_LAGS + 1)]
            )

            df_norm = normalizar_achatado(
                df_norm, metodo=METODO, norm_lags=None,
                prefix=pref, clase_col=None, b0_name=f"{pref}B0", b1_name=f"{pref}B1",
            )
            df_norm = agregar_deltas(df_norm, salto=SALTO, prefix=pref, clase_col=None)

            df_norm = agregar_deltas2(df_norm, prefix=pref)

        # customer_id solo existe con granularidad "pc" (en "p" el panel es por producto).
        _id_cols = [c for c in ["product_id", "customer_id", "periodo", "cat1"] if c in df_norm.columns]
        print(df_norm.select(
            _id_cols
            + [f"c1_tn{k}_norm" for k in range(3)]
            + [f"c1_tn{k}_delta" for k in range(2)]
            + [f"c1_tn_delta{k}" for k in range(1, 3)]
            + [f"c2_tn{k}_norm" for k in range(3)]
        ))

        # ── [02_FE celda 43] ──────────────────────────────────────────────────

        # --- Uso: una racha por nivel (producto-cliente, producto y las 3 categorias) ---
        # El nivel "p_" (producto) solo existe con granularidad "pc"; ver el complemento a
        # nivel producto de mas arriba. Con granularidad "p" no se genera.
        RACHA_PREFIXES = ["", "c1_", "c2_", "c3_", "b_"]
        if granularidad == "pc":
            RACHA_PREFIXES.insert(1, "p_")

        for pref in RACHA_PREFIXES:
            df_norm = agregar_racha(df_norm, prefix=pref)

        # customer_id solo existe con granularidad "pc" (en "p" el achatado es por producto).
        _id_cols = [c for c in ["product_id", "customer_id", "periodo"] if c in df_norm.columns]
        print(df_norm.select(
            _id_cols
            + [f"tn{k}_delta" for k in range(3)]
            + [f"{pref}tn_racha" for pref in RACHA_PREFIXES]
        ))


        escribir_parquet(df_norm, salida)
        logger.info(f"[02 fe] {df_norm.height:,} filas x {df_norm.width} columnas")

    etapa("02 feature engineering", salida, _calc, forzar)
    return salida
