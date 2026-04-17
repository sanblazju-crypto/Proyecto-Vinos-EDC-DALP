"""
model_manager.py
Módulo responsable de guardar y cargar modelos entrenados en disco.
"""

import os
import joblib

from logger_config import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL_DIR = "saved_models"


class ModelManager:
    """Gestiona la persistencia de modelos: guardado y carga desde disco."""

    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR):
        """
        Args:
            model_dir: Directorio donde se guardarán/cargarán los modelos.
        """
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        logger.debug(f"ModelManager inicializado. Directorio de modelos: '{self.model_dir}'")

    def save(self, trained_models: dict) -> dict:
        """
        Guarda todos los modelos del diccionario en disco usando joblib.

        Args:
            trained_models: Diccionario {nombre_modelo: modelo_entrenado}.

        Returns:
            dict: Rutas de los ficheros guardados {nombre_modelo: ruta}.

        Raises:
            ValueError: Si el diccionario de modelos está vacío.
        """
        if not trained_models:
            raise ValueError("No hay modelos para guardar.")

        logger.info(f"Guardando {len(trained_models)} modelo(s) en '{self.model_dir}'...")
        saved_paths = {}

        for name, model in trained_models.items():
            try:
                filename = f"{name}.joblib"
                filepath = os.path.join(self.model_dir, filename)
                joblib.dump(model, filepath)
                saved_paths[name] = filepath
                logger.info(f"✔ Modelo '{name}' guardado en: {filepath}")

            except Exception as e:
                logger.error(f"Error al guardar el modelo '{name}': {e}", exc_info=True)
                continue

        logger.info(f"Proceso de guardado completado. {len(saved_paths)}/{len(trained_models)} modelo(s) guardados.")
        return saved_paths

    def load(self, model_names: list | None = None) -> dict:
        """
        Carga modelos desde disco.

        Args:
            model_names: Lista de nombres de modelos a cargar (sin extensión).
                         Si es None, carga todos los modelos disponibles en el directorio.

        Returns:
            dict: Diccionario {nombre_modelo: modelo_cargado}.

        Raises:
            FileNotFoundError: Si el directorio no existe o no contiene modelos.
        """
        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(f"El directorio de modelos no existe: '{self.model_dir}'")

        available_files = [f for f in os.listdir(self.model_dir) if f.endswith(".joblib")]

        if not available_files:
            raise FileNotFoundError(f"No se encontraron modelos en '{self.model_dir}'.")

        # Filtrar por nombres solicitados si se especifican
        if model_names:
            target_files = [f"{name}.joblib" for name in model_names]
            files_to_load = [f for f in available_files if f in target_files]
            not_found = [f for f in target_files if f not in available_files]
            for missing in not_found:
                logger.warning(f"Modelo solicitado no encontrado en disco: '{missing}'")
        else:
            files_to_load = available_files

        logger.info(f"Cargando {len(files_to_load)} modelo(s) desde '{self.model_dir}'...")
        loaded_models = {}

        for filename in files_to_load:
            try:
                model_name = filename.replace(".joblib", "")
                filepath = os.path.join(self.model_dir, filename)
                model = joblib.load(filepath)
                loaded_models[model_name] = model
                logger.info(f"✔ Modelo '{model_name}' cargado desde: {filepath}")

            except Exception as e:
                logger.error(f"Error al cargar el modelo '{filename}': {e}", exc_info=True)
                continue

        logger.info(f"Carga completada. {len(loaded_models)}/{len(files_to_load)} modelo(s) disponibles.")
        return loaded_models

    def list_saved(self) -> list:
        """
        Lista los modelos guardados en el directorio.

        Returns:
            list: Nombres de los modelos disponibles (sin extensión).
        """
        if not os.path.isdir(self.model_dir):
            logger.warning(f"El directorio '{self.model_dir}' no existe.")
            return []

        models = [f.replace(".joblib", "") for f in os.listdir(self.model_dir) if f.endswith(".joblib")]
        logger.debug(f"Modelos guardados disponibles: {models}")
        return models


# ── Ejecución independiente ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Ejecutando ModelManager de forma independiente...")
    manager = ModelManager()
    guardados = manager.list_saved()
    if guardados:
        print(f"Modelos disponibles en disco: {guardados}")
        modelos = manager.load()
        print(f"Modelos cargados correctamente: {list(modelos.keys())}")
    else:
        print("No hay modelos guardados en disco todavía.")
        print("Ejecuta el pipeline completo primero: python app.py")
