"""
analyzer.py
Módulo de Análisis Exploratorio de Datos (EDA) del dataset Wine.

Genera estadísticas descriptivas, distribución de clases y correlaciones.
Puede ejecutarse de forma independiente si los datos ya han sido cargados,
o integrarse en el flujo del menú principal mediante SessionState.
"""

from __future__ import annotations

import pandas as pd

from logger_config import get_logger

logger = get_logger(__name__)


class Analyzer:
    """
    Realiza análisis exploratorio sobre el DataFrame del dataset Wine.
    No modifica los datos: solo los lee y produce informes.
    """

    def __init__(self, df: pd.DataFrame, feature_names: list[str], target_names: list[str]):
        """
        Args:
            df: DataFrame completo con características + columna 'target'.
            feature_names: Nombres de las columnas de características.
            target_names: Nombres de las clases objetivo.
        """
        if df is None or df.empty:
            raise ValueError("El DataFrame está vacío o no ha sido inicializado.")
        self.df = df
        self.feature_names = feature_names
        self.target_names = target_names

    # ── Sección 1: Vista general ──────────────────────────────────────────────

    def resumen_general(self) -> None:
        """Muestra dimensiones, tipos de datos y primeras filas."""
        logger.info("Generando resumen general del dataset...")
        n_muestras, n_cols = self.df.shape
        n_caracteristicas = n_cols - 1  # excluye 'target'

        print("\n" + "─" * 55)
        print("  RESUMEN GENERAL DEL DATASET")
        print("─" * 55)
        print(f"  Muestras totales   : {n_muestras}")
        print(f"  Características    : {n_caracteristicas}")
        print(f"  Clases objetivo    : {len(self.target_names)}  → {list(self.target_names)}")
        print(f"  Valores nulos      : {self.df.isnull().sum().sum()}")
        print(f"  Tipos de datos     : {dict(self.df.dtypes.value_counts())}")
        print("─" * 55)
        logger.info(f"Dataset: {n_muestras} muestras | {n_caracteristicas} características | {len(self.target_names)} clases")

    # ── Sección 2: Distribución de clases ─────────────────────────────────────

    def distribucion_clases(self) -> None:
        """Muestra el número de muestras por clase."""
        logger.info("Calculando distribución de clases...")
        counts = self.df["target"].value_counts().sort_index()

        print("\n" + "─" * 55)
        print("  DISTRIBUCIÓN DE CLASES")
        print("─" * 55)
        print(f"  {'Clase':<6} {'Nombre':<20} {'Muestras':>8}  {'%':>6}")
        print("  " + "-" * 45)
        total = len(self.df)
        for idx, count in counts.items():
            nombre = self.target_names[idx] if idx < len(self.target_names) else "?"
            pct = count / total * 100
            barra = "█" * int(pct / 3)
            print(f"  {idx:<6} {nombre:<20} {count:>8}  {pct:>5.1f}%  {barra}")
        print("─" * 55)

    # ── Sección 3: Estadísticas descriptivas ──────────────────────────────────

    def estadisticas_descriptivas(self) -> None:
        """Muestra estadísticas descriptivas de cada característica."""
        logger.info("Calculando estadísticas descriptivas...")
        desc = self.df[self.feature_names].describe().T
        desc = desc[["mean", "std", "min", "max"]]
        desc.columns = ["Media", "Desv.Est.", "Mínimo", "Máximo"]

        print("\n" + "─" * 70)
        print("  ESTADÍSTICAS DESCRIPTIVAS (por característica)")
        print("─" * 70)
        print(f"  {'Característica':<30} {'Media':>9} {'Desv.Est.':>10} {'Mín':>9} {'Máx':>9}")
        print("  " + "-" * 65)
        for nombre, fila in desc.iterrows():
            print(f"  {nombre:<30} {fila['Media']:>9.3f} {fila['Desv.Est.']:>10.3f} "
                  f"{fila['Mínimo']:>9.3f} {fila['Máximo']:>9.3f}")
        print("─" * 70)

    # ── Sección 4: Top correlaciones ──────────────────────────────────────────

    def top_correlaciones(self, top_n: int = 8) -> None:
        """
        Muestra las características con mayor correlación con la clase objetivo.

        Args:
            top_n: Número de características a mostrar.
        """
        logger.info("Calculando correlaciones con la variable objetivo...")
        corr = self.df[self.feature_names + ["target"]].corr()["target"].drop("target")
        corr_abs = corr.abs().sort_values(ascending=False).head(top_n)

        print("\n" + "─" * 55)
        print(f"  TOP {top_n} CORRELACIONES CON LA CLASE OBJETIVO")
        print("─" * 55)
        print(f"  {'Característica':<30} {'Correlación':>12}")
        print("  " + "-" * 45)
        for nombre, val in corr_abs.items():
            signo = "+" if corr[nombre] >= 0 else "-"
            barra = "█" * int(abs(val) * 20)
            print(f"  {nombre:<30} {signo}{val:>10.4f}  {barra}")
        print("─" * 55)

    # ── Ejecutar análisis completo ─────────────────────────────────────────────

    def ejecutar_completo(self) -> None:
        """Ejecuta todas las secciones de análisis en orden."""
        logger.info("Iniciando análisis exploratorio completo...")
        try:
            self.resumen_general()
            self.distribucion_clases()
            self.estadisticas_descriptivas()
            self.top_correlaciones()
            logger.info("Análisis exploratorio completado.")
        except Exception as e:
            logger.error(f"Error durante el análisis: {e}", exc_info=True)
            raise


# ── Ejecución independiente ────────────────────────────────────────────────────

if __name__ == "__main__":
    from data_loader import DataLoader

    print("Ejecutando Analyzer de forma independiente...")
    loader = DataLoader()
    df = loader.load()
    analyzer = Analyzer(df, loader.get_feature_names(), loader.get_target_names())
    analyzer.ejecutar_completo()
