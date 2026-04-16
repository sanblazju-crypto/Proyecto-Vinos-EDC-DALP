"""
main.py
Punto de entrada del proyecto Wine Classifier.
Orquesta el pipeline completo: carga → preprocesamiento → entrenamiento → evaluación → persistencia.
"""

import sys

from data_loader import DataLoader
from evaluator import Evaluator
from logger_config import get_logger
from model_manager import ModelManager
from preprocessor import Preprocessor
from trainer import Trainer

logger = get_logger(__name__)


def run_pipeline(
    train_size: float = 0.7,
    random_state: int = 25,
    model_keys: list | None = None,
    model_params: dict | None = None,
    save_models: bool = True,
    reload_and_evaluate: bool = True,
) -> dict:
    """
    Ejecuta el pipeline completo de clasificación de vinos.

    Args:
        train_size: Proporción de datos para entrenamiento.
        random_state: Semilla para reproducibilidad.
        model_keys: Modelos a entrenar. None = todos los del catálogo.
        model_params: Hiperparámetros personalizados por modelo.
        save_models: Si True, guarda los modelos en disco tras el entrenamiento.
        reload_and_evaluate: Si True, recarga los modelos guardados y vuelve a evaluar.

    Returns:
        dict: Resultados de evaluación del pipeline.
    """
    logger.info("=" * 60)
    logger.info("INICIO DEL PIPELINE WINE CLASSIFIER")
    logger.info("=" * 60)

    # ── 1. CARGA DEL DATASET ──────────────────────────────────────
    logger.info("[ FASE 1 ] Carga del dataset")
    try:
        loader = DataLoader()
        df = loader.load()
        loader.summary()
        feature_names = loader.get_feature_names()
        target_names = loader.get_target_names()
    except RuntimeError as e:
        logger.critical(f"Fallo crítico en la carga de datos. Abortando pipeline. Error: {e}")
        sys.exit(1)

    # ── 2. PREPROCESAMIENTO ───────────────────────────────────────
    logger.info("[ FASE 2 ] Preprocesamiento de datos")
    try:
        preprocessor = Preprocessor()
        X_scaled, y = preprocessor.fit_transform(df, feature_names)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Fallo crítico en el preprocesamiento. Abortando pipeline. Error: {e}")
        sys.exit(1)

    # ── 3. DIVISIÓN TRAIN / TEST ──────────────────────────────────
    logger.info("[ FASE 3 ] División de datos (train/test)")
    try:
        trainer = Trainer(train_size=train_size, random_state=random_state)
        X_train, X_test, y_train, y_test = trainer.split(X_scaled, y)
    except (ValueError, RuntimeError) as e:
        logger.critical(f"Fallo crítico en la división de datos. Abortando pipeline. Error: {e}")
        sys.exit(1)

    # ── 4. ENTRENAMIENTO ──────────────────────────────────────────
    logger.info("[ FASE 4 ] Entrenamiento de modelos")
    try:
        trained_models = trainer.train(model_keys=model_keys, model_params=model_params)
    except RuntimeError as e:
        logger.critical(f"Fallo crítico en el entrenamiento. Abortando pipeline. Error: {e}")
        sys.exit(1)

    # ── 5. EVALUACIÓN ─────────────────────────────────────────────
    logger.info("[ FASE 5 ] Evaluación de modelos entrenados")
    evaluator = Evaluator(X_test, y_test, target_names=target_names)
    results = evaluator.evaluate(trained_models)
    evaluator.print_summary()

    # ── 6. GUARDADO DE MODELOS ────────────────────────────────────
    model_manager = ModelManager()
    if save_models:
        logger.info("[ FASE 6 ] Guardado de modelos en disco")
        try:
            model_manager.save(trained_models)
        except ValueError as e:
            logger.error(f"No se pudieron guardar los modelos: {e}")

    # ── 7. RECARGA Y RE-EVALUACIÓN (DEMO) ─────────────────────────
    if reload_and_evaluate and save_models:
        logger.info("[ FASE 7 ] Recarga de modelos desde disco y re-evaluación")
        try:
            reloaded_models = model_manager.load()
            re_evaluator = Evaluator(X_test, y_test, target_names=target_names)
            re_evaluator.evaluate(reloaded_models)
            logger.info("Re-evaluación de modelos recargados completada.")
        except (FileNotFoundError, RuntimeError) as e:
            logger.error(f"Error durante la recarga de modelos: {e}", exc_info=True)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETADO EXITOSAMENTE")
    logger.info("=" * 60)
    return results


if __name__ == "__main__":
    run_pipeline(
        train_size=0.7,
        random_state=25,
        model_keys=None,       # None → entrena todos los modelos del catálogo
        model_params={
            "svm": {"kernel": "rbf", "C": 1.0},
            "logistic_regression": {"max_iter": 200},
        },
        save_models=True,
        reload_and_evaluate=True,
    )
