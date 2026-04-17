"""
app.py
Controlador central del menú CLI del proyecto Wine Classifier.

Orquesta las distintas fases del proyecto a través de un menú interactivo,
gestionando el estado de la sesión, el orden de ejecución y los avisos
cuando el usuario intenta saltar pasos con dependencias pendientes.

Ejecución:
    python app.py
"""

import sys

import config
import ui_helpers as ui
from data_loader import DataLoader
from analyzer import Analyzer
from evaluator import Evaluator
from logger_config import get_logger
from model_manager import ModelManager
from preprocessor import Preprocessor
from state_manager import SessionState
from trainer import Trainer, MODEL_CATALOG

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  ACCIONES DE CADA SECCIÓN DEL MENÚ
# ══════════════════════════════════════════════════════════════════════════════

def seccion_inicio(estado: SessionState) -> None:
    """Pantalla de bienvenida e información del proyecto."""
    ui.cabecera(config.PROJECT_NAME, f"v{config.PROJECT_VERSION}  ·  {config.PROJECT_AUTHOR}")
    ui.seccion("Información del proyecto")

    filas = [
        ("Nombre",    config.PROJECT_NAME),
        ("Versión",   config.PROJECT_VERSION),
        ("Autor",     config.PROJECT_AUTHOR),
        ("Dataset",   config.DATASET_NAME),
        ("Descripción", config.DATASET_DESC),
        ("Modelos",   ", ".join(MODEL_CATALOG.keys())),
        ("Split",     f"{int(config.TRAIN_SIZE * 100)}% train / {int((1 - config.TRAIN_SIZE) * 100)}% test"),
    ]
    for clave, valor in filas:
        print(ui.C_DIM + f"  {clave:<14}: " + ui.C_RESET + valor)

    ui.salto()
    ui.seccion("Flujo de ejecución recomendado")
    pasos = [
        "Carga y Preparación de Datos",
        "Análisis Exploratorio",
        "Modelado (División + Entrenamiento)",
        "Evaluación de Modelos",
        "Gestión de Modelos (Guardar / Cargar)",
    ]
    for i, paso in enumerate(pasos, 1):
        print(ui.C_OPCION + f"  {i}. " + ui.C_RESET + paso)

    ui.salto()
    logger.info("Pantalla de inicio mostrada.")


def seccion_datos(estado: SessionState) -> None:
    """Carga el dataset y aplica el preprocesamiento. Actualiza el estado."""
    ui.cabecera(config.PROJECT_NAME, "Carga y Preparación de Datos")

    # ── Carga ─────────────────────────────────────────────────────────────────
    ui.seccion("1 / 2  ·  Carga del dataset")
    try:
        loader = DataLoader()
        df = loader.load()
        loader.summary()

        estado.df            = df
        estado.feature_names = loader.get_feature_names()
        estado.target_names  = loader.get_target_names()

        ui.ok(f"Dataset cargado: {len(df)} muestras  ·  {len(estado.feature_names)} características")
        logger.info("Dataset cargado correctamente desde el menú.")

    except Exception as e:
        ui.error(f"Error al cargar el dataset: {e}")
        logger.error(f"Fallo en carga de datos (menú): {e}", exc_info=True)
        return

    # ── Preprocesamiento ──────────────────────────────────────────────────────
    ui.seccion("2 / 2  ·  Preprocesamiento (StandardScaler)")
    try:
        preprocessor = Preprocessor()
        X_scaled, y  = preprocessor.fit_transform(df, estado.feature_names)

        estado.preprocessor = preprocessor
        estado.X_scaled     = X_scaled
        estado.y            = y
        estado.datos_listos = True

        ui.ok(f"Escalado completado  ·  X_scaled: {X_scaled.shape}")
        logger.info("Preprocesamiento completado correctamente desde el menú.")

    except Exception as e:
        ui.error(f"Error en el preprocesamiento: {e}")
        logger.error(f"Fallo en preprocesamiento (menú): {e}", exc_info=True)


def seccion_analisis(estado: SessionState) -> None:
    """Ejecuta el análisis exploratorio sobre los datos ya cargados."""
    ui.cabecera(config.PROJECT_NAME, "Análisis Exploratorio de Datos")

    try:
        analyzer = Analyzer(estado.df, estado.feature_names, estado.target_names)

        ui.seccion("Elige el tipo de análisis")
        print(ui.C_OPCION + "  [1]" + ui.C_RESET + "  Análisis completo (todas las secciones)")
        print(ui.C_OPCION + "  [2]" + ui.C_RESET + "  Resumen general")
        print(ui.C_OPCION + "  [3]" + ui.C_RESET + "  Distribución de clases")
        print(ui.C_OPCION + "  [4]" + ui.C_RESET + "  Estadísticas descriptivas")
        print(ui.C_OPCION + "  [5]" + ui.C_RESET + "  Top correlaciones con la clase objetivo")
        ui.salto()

        opcion = ui.pedir_opcion()

        acciones = {
            "1": analyzer.ejecutar_completo,
            "2": analyzer.resumen_general,
            "3": analyzer.distribucion_clases,
            "4": analyzer.estadisticas_descriptivas,
            "5": analyzer.top_correlaciones,
        }

        if opcion in acciones:
            acciones[opcion]()
            ui.salto()
            ui.ok("Análisis completado.")
            logger.info(f"Análisis opción '{opcion}' ejecutado correctamente.")
        else:
            ui.warn("Opción no reconocida. Volviendo al menú.")

    except Exception as e:
        ui.error(f"Error durante el análisis: {e}")
        logger.error(f"Fallo en análisis (menú): {e}", exc_info=True)


def seccion_modelado(estado: SessionState) -> None:
    """Divide los datos y entrena los modelos seleccionados por el usuario."""
    ui.cabecera(config.PROJECT_NAME, "Modelado: División + Entrenamiento")

    # ── Selección de modelos ───────────────────────────────────────────────────
    ui.seccion("Modelos disponibles")
    catalogo = list(MODEL_CATALOG.keys())
    for i, nombre in enumerate(catalogo, 1):
        print(ui.C_OPCION + f"  [{i}]" + ui.C_RESET + f"  {nombre}")
    print(ui.C_OPCION + "  [*]" + ui.C_RESET + "  Todos los modelos")
    ui.salto()

    raw = ui.pedir_opcion("  Elige modelo(s) separados por coma [* = todos]: ")

    if raw == "*" or raw == "":
        model_keys = None          # Trainer interpreta None como "todos"
        ui.info("Se entrenarán todos los modelos del catálogo.")
    else:
        indices = [r.strip() for r in raw.split(",")]
        model_keys = []
        for idx in indices:
            if idx.isdigit() and 1 <= int(idx) <= len(catalogo):
                model_keys.append(catalogo[int(idx) - 1])
            else:
                ui.warn(f"  Índice '{idx}' ignorado.")
        if not model_keys:
            ui.error("No se seleccionó ningún modelo válido.")
            return

    # ── División train/test ───────────────────────────────────────────────────
    ui.seccion("División train / test")
    try:
        trainer = Trainer(train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE)
        X_train, X_test, y_train, y_test = trainer.split(estado.X_scaled, estado.y)

        estado.trainer = trainer
        ui.ok(f"División OK  ·  Train: {len(X_train)}  |  Test: {len(X_test)}")
        logger.info("División train/test completada desde el menú.")

    except Exception as e:
        ui.error(f"Error en la división de datos: {e}")
        logger.error(f"Fallo en división (menú): {e}", exc_info=True)
        return

    # ── Entrenamiento ─────────────────────────────────────────────────────────
    ui.seccion("Entrenamiento de modelos")
    try:
        ui.info("Entrenando... (puede tardar unos segundos)")
        trained_models = trainer.train(
            model_keys=model_keys,
            model_params=config.MODEL_PARAMS,
        )

        estado.trained_models    = trained_models
        estado.modelos_entrenados = True
        # Resetear flags de pasos posteriores si se re-entrena
        estado.modelos_evaluados = False
        estado.modelos_guardados = False

        for nombre in trained_models:
            ui.ok(f"Modelo entrenado: {nombre}")
        logger.info(f"Entrenamiento completado. Modelos: {list(trained_models.keys())}")

    except Exception as e:
        ui.error(f"Error durante el entrenamiento: {e}")
        logger.error(f"Fallo en entrenamiento (menú): {e}", exc_info=True)


def seccion_evaluacion(estado: SessionState) -> None:
    """Evalúa los modelos entrenados y muestra el informe de métricas."""
    ui.cabecera(config.PROJECT_NAME, "Evaluación de Modelos")

    try:
        trainer   = estado.trainer
        evaluator = Evaluator(
            trainer.X_test,
            trainer.y_test,
            target_names=estado.target_names,
        )
        results = evaluator.evaluate(estado.trained_models)
        evaluator.print_summary()

        estado.evaluator          = evaluator
        estado.evaluation_results = results
        estado.modelos_evaluados  = True

        ui.ok("Evaluación completada correctamente.")
        logger.info("Evaluación completada desde el menú.")

    except Exception as e:
        ui.error(f"Error durante la evaluación: {e}")
        logger.error(f"Fallo en evaluación (menú): {e}", exc_info=True)


def seccion_gestion(estado: SessionState) -> None:
    """Sub-menú para guardar o cargar modelos desde disco."""
    ui.cabecera(config.PROJECT_NAME, "Gestión de Modelos")

    manager = ModelManager(model_dir=config.MODELS_DIR)
    estado.model_manager = manager

    ui.seccion("Opciones disponibles")
    print(ui.C_OPCION + "  [1]" + ui.C_RESET + "  Guardar modelos actuales en disco")
    print(ui.C_OPCION + "  [2]" + ui.C_RESET + "  Cargar modelos desde disco")
    print(ui.C_OPCION + "  [3]" + ui.C_RESET + "  Listar modelos guardados")
    ui.salto()

    opcion = ui.pedir_opcion()

    if opcion == "1":
        _guardar_modelos(estado, manager)

    elif opcion == "2":
        _cargar_modelos(estado, manager)

    elif opcion == "3":
        _listar_modelos(manager)

    else:
        ui.warn("Opción no reconocida.")


def _guardar_modelos(estado: SessionState, manager: ModelManager) -> None:
    ui.seccion("Guardando modelos")
    try:
        paths = manager.save(estado.trained_models)
        estado.saved_model_paths = paths
        estado.modelos_guardados  = True
        for nombre, ruta in paths.items():
            ui.ok(f"{nombre}  →  {ruta}")
        logger.info(f"Modelos guardados: {list(paths.keys())}")
    except Exception as e:
        ui.error(f"Error al guardar: {e}")
        logger.error(f"Fallo al guardar modelos (menú): {e}", exc_info=True)


def _cargar_modelos(estado: SessionState, manager: ModelManager) -> None:
    ui.seccion("Cargando modelos desde disco")
    try:
        disponibles = manager.list_saved()
        if not disponibles:
            ui.warn("No hay modelos guardados en disco.")
            return
        ui.info(f"Modelos disponibles: {', '.join(disponibles)}")
        reloaded = manager.load()
        estado.trained_models     = reloaded
        estado.modelos_entrenados = True
        estado.modelos_evaluados  = False
        for nombre in reloaded:
            ui.ok(f"Cargado: {nombre}")
        logger.info(f"Modelos cargados desde disco: {list(reloaded.keys())}")
    except Exception as e:
        ui.error(f"Error al cargar: {e}")
        logger.error(f"Fallo al cargar modelos (menú): {e}", exc_info=True)


def _listar_modelos(manager: ModelManager) -> None:
    ui.seccion("Modelos guardados en disco")
    modelos = manager.list_saved()
    if modelos:
        for m in modelos:
            ui.info(f"  • {m}.joblib")
    else:
        ui.warn("No hay modelos guardados todavía.")


def seccion_pipeline(estado: SessionState) -> None:
    """Ejecuta el pipeline completo y actualiza el estado de la sesión."""
    ui.cabecera(config.PROJECT_NAME, "Pipeline Completo")

    ui.warn("Se ejecutará el pipeline completo. El estado actual se reiniciará.")
    ui.salto()
    confirma = ui.pedir_opcion("  ¿Continuar? [s/n]: ")
    if confirma not in ("s", "si", "sí", "y", "yes"):
        ui.info("Pipeline cancelado.")
        return

    estado.reset()

    # Fase 1 y 2: Datos
    seccion_datos(estado)
    if not estado.datos_listos:
        ui.error("Pipeline abortado en la fase de datos.")
        return

    # Fase 3: Modelado (todos los modelos, parámetros del config)
    seccion_modelado_auto(estado)
    if not estado.modelos_entrenados:
        ui.error("Pipeline abortado en la fase de modelado.")
        return

    # Fase 4: Evaluación
    seccion_evaluacion(estado)

    # Fase 5: Guardar
    manager = ModelManager(model_dir=config.MODELS_DIR)
    estado.model_manager = manager
    _guardar_modelos(estado, manager)

    ui.salto()
    ui.ok("Pipeline completo finalizado con éxito.")
    logger.info("Pipeline completo ejecutado desde el menú.")


def seccion_modelado_auto(estado: SessionState) -> None:
    """Versión silenciosa de modelado para el pipeline completo (sin interacción)."""
    try:
        trainer = Trainer(train_size=config.TRAIN_SIZE, random_state=config.RANDOM_STATE)
        trainer.split(estado.X_scaled, estado.y)
        trained_models = trainer.train(model_keys=None, model_params=config.MODEL_PARAMS)

        estado.trainer            = trainer
        estado.trained_models     = trained_models
        estado.modelos_entrenados = True
        estado.modelos_evaluados  = False
        estado.modelos_guardados  = False

        for nombre in trained_models:
            ui.ok(f"Modelo entrenado: {nombre}")
        logger.info("Modelado automático completado (pipeline completo).")
    except Exception as e:
        ui.error(f"Error en el modelado automático: {e}")
        logger.error(f"Fallo en modelado automático: {e}", exc_info=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DESPACHADOR DEL MENÚ
# ══════════════════════════════════════════════════════════════════════════════

ACCIONES_MENU: dict = {
    "1": seccion_inicio,
    "2": seccion_datos,
    "3": seccion_analisis,
    "4": seccion_modelado,
    "5": seccion_evaluacion,
    "6": seccion_gestion,
    "7": seccion_pipeline,
}

DEPENDENCIAS_MENU: dict = {
    "3": ["datos_listos"],
    "4": ["datos_listos"],
    "5": ["datos_listos", "modelos_entrenados"],
    "6": ["modelos_entrenados"],
}


def ejecutar_opcion(opcion: str, estado: SessionState) -> None:
    """
    Comprueba dependencias y ejecuta la acción correspondiente a la opción.

    Args:
        opcion: Tecla pulsada por el usuario.
        estado: Estado de la sesión actual.
    """
    requires = DEPENDENCIAS_MENU.get(opcion, [])
    ok_flag, faltantes = estado.check_requires(requires)

    if not ok_flag:
        mensajes_faltantes = {
            "datos_listos":        "Carga y Preparación de Datos  [opción 2]",
            "modelos_entrenados":  "Modelado  [opción 4]",
            "modelos_evaluados":   "Evaluación  [opción 5]",
        }
        ui.salto()
        ui.warn("No puedes ejecutar este paso todavía.")
        ui.info("Primero debes completar:")
        for f in faltantes:
            ui.info(f"  → {mensajes_faltantes.get(f, f)}")
        logger.warning(f"Opción '{opcion}' bloqueada. Flags pendientes: {faltantes}")
        return

    accion = ACCIONES_MENU.get(opcion)
    if accion:
        accion(estado)


# ══════════════════════════════════════════════════════════════════════════════
#  BUCLE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto de entrada principal. Inicia el bucle del menú interactivo."""
    logger.info("Aplicación Wine Classifier iniciada (menú interactivo).")
    estado = SessionState()

    while True:
        ui.cabecera(
            config.PROJECT_NAME,
            f"v{config.PROJECT_VERSION}  ·  {config.PROJECT_AUTHOR}",
        )
        ui.mostrar_menu(config.MENU_SECTIONS, estado)

        opcion = ui.pedir_opcion()

        if opcion == "0":
            ui.cabecera(config.PROJECT_NAME)
            ui.info("Cerrando aplicación. ¡Hasta pronto!")
            ui.salto()
            logger.info("Aplicación cerrada por el usuario.")
            sys.exit(0)

        elif opcion == "s":
            ui.cabecera(config.PROJECT_NAME, "Estado de la sesión")
            ui.mostrar_estado(estado)
            ui.pausar()

        elif opcion in ACCIONES_MENU:
            ejecutar_opcion(opcion, estado)
            ui.pausar()

        else:
            ui.salto()
            ui.warn(f"Opción '{opcion}' no válida. Elige un número del menú.")
            ui.pausar("Pulsa ENTER para continuar...")


if __name__ == "__main__":
    main()
