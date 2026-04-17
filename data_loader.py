"""
data_loader.py
Módulo responsable de la carga del dataset Wine y su conversión a DataFrame.
"""

import pandas as pd
from sklearn.datasets import load_wine

from logger_config import get_logger

logger = get_logger(__name__)


class DataLoader:
    """Carga el dataset Wine de sklearn y lo expone como un DataFrame de pandas."""

    def __init__(self):
        self.wine_data = None
        self.df = None
        self.feature_names = None
        self.target_names = None

    def load(self) -> pd.DataFrame:
        """
        Carga el dataset Wine y lo convierte a DataFrame.

        Returns:
            pd.DataFrame: DataFrame con características y columna 'target'.

        Raises:
            RuntimeError: Si ocurre algún error durante la carga.
        """
        logger.info("Iniciando carga del dataset Wine...")
        try:
            self.wine_data = load_wine()
            self.feature_names = self.wine_data.feature_names
            self.target_names = self.wine_data.target_names

            self.df = pd.DataFrame(self.wine_data.data, columns=self.feature_names)
            self.df["target"] = self.wine_data.target

            n_samples, n_features = self.df.shape
            logger.info(f"Dataset cargado correctamente: {n_samples} muestras, {n_features - 1} características.")
            logger.debug(f"Clases objetivo: {list(self.target_names)}")
            logger.debug(f"Distribución de clases:\n{self.df['target'].value_counts().to_string()}")

            return self.df

        except Exception as e:
            logger.error(f"Error al cargar el dataset: {e}", exc_info=True)
            raise RuntimeError(f"No se pudo cargar el dataset: {e}") from e

    def get_feature_names(self) -> list:
        """Devuelve la lista de nombres de características."""
        if self.feature_names is None:
            raise RuntimeError("El dataset no ha sido cargado aún. Llama a load() primero.")
        return list(self.feature_names)

    def get_target_names(self) -> list:
        """Devuelve los nombres de las clases objetivo."""
        if self.target_names is None:
            raise RuntimeError("El dataset no ha sido cargado aún. Llama a load() primero.")
        return list(self.target_names)

    def summary(self) -> None:
        """Imprime un resumen estadístico del dataset cargado."""
        if self.df is None:
            logger.warning("No hay datos cargados para mostrar el resumen.")
            return
        logger.info("Resumen estadístico del dataset:")
        logger.debug(f"\n{self.df.describe().to_string()}")


# ── Ejecución independiente ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ejecutando DataLoader de forma independiente...")
    loader = DataLoader()
    df = loader.load()
    loader.summary()
    print(f"\nPrimeras filas:\n{df.head()}")
