"""
evaluator.py
Módulo responsable de la evaluación de los modelos entrenados.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from logger_config import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Evalúa el rendimiento de los modelos entrenados sobre el conjunto de prueba."""

    def __init__(self, X_test: np.ndarray, y_test: pd.Series, target_names: list | None = None):
        """
        Args:
            X_test: Características del conjunto de prueba.
            y_test: Etiquetas reales del conjunto de prueba.
            target_names: Nombres de las clases (opcional, para mejorar los reportes).
        """
        self.X_test = X_test
        self.y_test = y_test
        self.target_names = target_names
        self.results: dict = {}

    def evaluate(self, trained_models: dict) -> dict:
        """
        Genera predicciones y métricas de evaluación para cada modelo.

        Args:
            trained_models: Diccionario {nombre_modelo: modelo_entrenado}.

        Returns:
            dict: Diccionario con resultados por modelo, incluyendo accuracy,
                  classification_report y confusion_matrix.

        Raises:
            ValueError: Si no se reciben modelos.
            RuntimeError: Si la evaluación falla de forma crítica.
        """
        if not trained_models:
            raise ValueError("No se han recibido modelos para evaluar.")

        logger.info(f"Evaluando {len(trained_models)} modelo(s)...")

        for name, model in trained_models.items():
            try:
                logger.debug(f"Generando predicciones para '{name}'...")
                preds = model.predict(self.X_test)

                accuracy = accuracy_score(self.y_test, preds)
                report = classification_report(
                    self.y_test, preds,
                    target_names=self.target_names,
                    zero_division=0
                )
                conf_matrix = confusion_matrix(self.y_test, preds)

                self.results[name] = {
                    "accuracy": accuracy,
                    "report": report,
                    "confusion_matrix": conf_matrix,
                    "predictions": preds,
                }

                logger.info(f"[{name}] Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
                logger.debug(f"[{name}] Reporte de clasificación:\n{report}")
                logger.debug(f"[{name}] Matriz de confusión:\n{conf_matrix}")

            except Exception as e:
                logger.error(f"Error al evaluar el modelo '{name}': {e}", exc_info=True)
                continue

        logger.info("Evaluación finalizada.")
        return self.results

    def print_summary(self) -> None:
        """Imprime en consola un resumen de los resultados de evaluación."""
        if not self.results:
            logger.warning("No hay resultados de evaluación disponibles.")
            return

        separator = "=" * 60
        print(f"\n{separator}")
        print("RESUMEN DE EVALUACIÓN DE MODELOS")
        print(separator)

        for name, metrics in self.results.items():
            print(f"\n🔹 {name.upper()}")
            print(f"   Accuracy: {metrics['accuracy'] * 100:.2f}%")
            print(f"\n{metrics['report']}")

        # Determinar el mejor modelo por accuracy
        best_model = max(self.results, key=lambda k: self.results[k]["accuracy"])
        best_acc = self.results[best_model]["accuracy"]
        print(f"\n✅ Mejor modelo: '{best_model}' con accuracy {best_acc * 100:.2f}%")
        print(separator)
        logger.info(f"Mejor modelo: '{best_model}' | Accuracy: {best_acc:.4f}")


# ── Ejecución independiente ────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import DataLoader
    from preprocessor import Preprocessor
    from trainer import Trainer
    print("Ejecutando Evaluator de forma independiente...")
    loader = DataLoader()
    df = loader.load()
    prep = Preprocessor()
    X_scaled, y = prep.fit_transform(df, loader.get_feature_names())
    trainer = Trainer(train_size=0.7, random_state=25)
    trainer.split(X_scaled, y)
    modelos = trainer.train()
    evaluator = Evaluator(trainer.X_test, trainer.y_test, loader.get_target_names())
    evaluator.evaluate(modelos)
    evaluator.print_summary()
