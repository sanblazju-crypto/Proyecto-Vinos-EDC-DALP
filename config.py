"""
config.py
Configuración centralizada del proyecto Wine Classifier.
Todos los parámetros ajustables y rutas residen aquí para facilitar el mantenimiento.
"""

import os

# ── RUTAS ─────────────────────────────────────────────────────────────────────
# Directorio raíz del proyecto (donde vive este fichero)
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))

# Directorios de salida (se crean automáticamente si no existen)
LOGS_DIR: str        = os.path.join(BASE_DIR, "logs")
MODELS_DIR: str      = os.path.join(BASE_DIR, "saved_models")
OUTPUTS_DIR: str     = os.path.join(BASE_DIR, "outputs")

# ── PARÁMETROS DE ENTRENAMIENTO ───────────────────────────────────────────────
TRAIN_SIZE: float    = 0.7          # Proporción train / total
RANDOM_STATE: int    = 25           # Semilla de reproducibilidad

# Hiperparámetros por modelo (se pasan a Trainer.train())
MODEL_PARAMS: dict = {
    "logistic_regression": {"max_iter": 200},
    "svm":                 {"kernel": "rbf", "C": 1.0},
    "decision_tree":       {},       # usa valores por defecto de sklearn
}

# ── INFORMACIÓN DEL PROYECTO ──────────────────────────────────────────────────
PROJECT_NAME: str    = "Wine Classifier"
PROJECT_VERSION: str = "2.0.0"
PROJECT_AUTHOR: str  = "EDC-DALP"
DATASET_NAME: str    = "UCI Wine Dataset"
DATASET_DESC: str    = (
    "178 muestras de vino clasificadas en 3 variedades, "
    "con 13 atributos fisicoquímicos cada una."
)

# ── MENÚ: ETIQUETAS Y DEPENDENCIAS ───────────────────────────────────────────
# Cada sección define su id, etiqueta visible y los flags de estado requeridos
MENU_SECTIONS: list[dict] = [
    {
        "id": "inicio",
        "label": "Inicio / Información del proyecto",
        "requires": [],
    },
    {
        "id": "datos",
        "label": "Carga y Preparación de Datos",
        "requires": [],
    },
    {
        "id": "analisis",
        "label": "Análisis Exploratorio",
        "requires": ["datos_listos"],
    },
    {
        "id": "modelado",
        "label": "Modelado (División + Entrenamiento)",
        "requires": ["datos_listos"],
    },
    {
        "id": "evaluacion",
        "label": "Evaluación de Modelos",
        "requires": ["datos_listos", "modelos_entrenados"],
    },
    {
        "id": "gestion",
        "label": "Gestión de Modelos (Guardar / Cargar)",
        "requires": ["modelos_entrenados"],
    },
    {
        "id": "pipeline",
        "label": "Ejecutar Pipeline Completo",
        "requires": [],
    },
]
