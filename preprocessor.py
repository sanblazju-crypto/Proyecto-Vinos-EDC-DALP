"""
preprocessor.py
Módulo responsable del preprocesamiento de datos: separación de características/etiquetas y escalado.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from logger_config import get_logger

logger = get_logger(__name__)


class Preprocessor:
    """Separa características y etiquetas, y aplica escalado estándar."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names: list = []
        self.X_scaled: np.ndarray | None = None
        self.y: pd.Series | None = None

    def fit_transform(self, df: pd.DataFrame, feature_names: list) -> tuple[np.ndarray, pd.Series]:
        """
        Separa X e y del DataFrame, ajusta el scaler y transforma las características.

        Args:
            df: DataFrame completo con características y columna 'target'.
            feature_names: Lista de columnas que representan las características.

        Returns:
            Tuple (X_scaled, y): Características escaladas y vector de etiquetas.

        Raises:
            ValueError: Si faltan columnas requeridas en el DataFrame.
            RuntimeError: Si el escalado falla por algún motivo inesperado.
        """
        logger.info("Iniciando preprocesamiento de datos...")
        try:
            missing = [col for col in feature_names + ["target"] if col not in df.columns]
            if missing:
                raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")

            self.feature_names = feature_names
            X = df[feature_names].copy()
            self.y = df["target"].copy()

            logger.debug(f"Características seleccionadas: {feature_names}")
            logger.debug(f"Distribución de etiquetas:\n{self.y.value_counts().to_string()}")

            self.scaler.fit(X)
            self.X_scaled = self.scaler.transform(X.values)

            logger.info(f"Escalado completado. Forma de X_scaled: {self.X_scaled.shape}")
            logger.debug(f"Media por característica (después del escalado): {self.X_scaled.mean(axis=0).round(4)}")

            return self.X_scaled, self.y

        except ValueError as ve:
            logger.error(f"Error de validación en preprocesamiento: {ve}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error inesperado durante el preprocesamiento: {e}", exc_info=True)
            raise RuntimeError(f"Fallo en el preprocesamiento: {e}") from e

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transforma nuevos datos usando el scaler ya ajustado.

        Args:
            X: Array de características sin escalar.

        Returns:
            np.ndarray: Características escaladas.

        Raises:
            RuntimeError: Si el scaler no ha sido ajustado previamente.
        """
        if self.X_scaled is None:
            raise RuntimeError("El Preprocessor no ha sido ajustado. Llama a fit_transform() primero.")
        try:
            logger.debug("Transformando nuevas muestras con el scaler existente.")
            return self.scaler.transform(X)
        except Exception as e:
            logger.error(f"Error al transformar datos nuevos: {e}", exc_info=True)
            raise
