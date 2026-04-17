"""
state_manager.py
Gestión del estado de la sesión activa del proyecto Wine Classifier.

Centraliza todos los objetos y flags que se generan durante la ejecución
del menú, garantizando que cada paso dispone de los datos que necesita.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    # Importaciones solo para anotaciones de tipo (evita importaciones circulares)
    from evaluator import Evaluator
    from model_manager import ModelManager
    from preprocessor import Preprocessor
    from trainer import Trainer


@dataclass
class SessionState:
    """
    Contenedor único del estado compartido entre todas las fases del menú.

    Los objetos se rellenan secuencialmente a medida que el usuario ejecuta
    cada sección. Los flags booleanos actúan como semáforos que el menú
    consulta para bloquear o permitir cada opción.
    """

    # ── Datos crudos ──────────────────────────────────────────────────────────
    df: pd.DataFrame | None               = None
    feature_names: list[str]              = field(default_factory=list)
    target_names: list[str]               = field(default_factory=list)

    # ── Datos preprocesados ───────────────────────────────────────────────────
    X_scaled: np.ndarray | None           = None
    y: pd.Series | None                   = None

    # ── Objetos de las fases ─────────────────────────────────────────────────
    preprocessor: object | None           = None   # Preprocessor
    trainer: object | None                = None   # Trainer
    evaluator: object | None              = None   # Evaluator
    model_manager: object | None          = None   # ModelManager

    # ── Resultados ────────────────────────────────────────────────────────────
    trained_models: dict                  = field(default_factory=dict)
    evaluation_results: dict              = field(default_factory=dict)
    saved_model_paths: dict               = field(default_factory=dict)

    # ── FLAGS de estado (semáforos de dependencia) ────────────────────────────
    datos_listos: bool                    = False   # Carga + preprocesado OK
    modelos_entrenados: bool              = False   # train() ejecutado OK
    modelos_evaluados: bool               = False   # evaluate() ejecutado OK
    modelos_guardados: bool               = False   # save() ejecutado OK

    # ─────────────────────────────────────────────────────────────────────────

    def check_requires(self, requires: list[str]) -> tuple[bool, list[str]]:
        """
        Comprueba si todos los flags requeridos están activos.

        Args:
            requires: Lista de nombres de flags que deben ser True.

        Returns:
            (ok, faltantes): True si todo está listo; lista de flags en False.
        """
        faltantes = [r for r in requires if not getattr(self, r, False)]
        return (len(faltantes) == 0, faltantes)

    def resumen_estado(self) -> dict[str, str]:
        """Devuelve un diccionario con el estado visual de cada flag."""
        iconos = {True: "✔", False: "✘"}
        return {
            "Datos cargados y preprocesados": iconos[self.datos_listos],
            "Modelos entrenados":             iconos[self.modelos_entrenados],
            "Modelos evaluados":              iconos[self.modelos_evaluados],
            "Modelos guardados":              iconos[self.modelos_guardados],
        }

    def reset(self) -> None:
        """Reinicia el estado completo de la sesión (útil para re-ejecutar el pipeline)."""
        self.__init__()
