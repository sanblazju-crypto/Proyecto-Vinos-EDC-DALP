"""
logger_config.py
Configuración centralizada del sistema de logging para el proyecto Wine Classifier.
"""

import logging
import os
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Devuelve un logger configurado con salida a consola y a fichero de log.

    Args:
        name: Nombre del módulo que solicita el logger (usa __name__).

    Returns:
        logging.Logger: Instancia del logger configurada.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(log_dir, f"wine_classifier_{datetime.now().strftime('%Y%m%d')}.log")

    logger = logging.getLogger(name)

    # Evitar handlers duplicados si el logger ya fue inicializado
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler para fichero (nivel DEBUG, captura todo)
    file_handler = logging.FileHandler(log_filename, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Handler para consola (nivel INFO, solo mensajes relevantes)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
