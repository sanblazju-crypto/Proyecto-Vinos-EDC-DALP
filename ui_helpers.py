"""
ui_helpers.py
Utilidades de presentación para el menú CLI del proyecto Wine Classifier.

Usa colorama (disponible en el entorno) para colores ANSI portables.
Todas las funciones son puras y no tienen efectos secundarios sobre el estado
de la aplicación: solo imprimen en pantalla.
"""

import os
import shutil
from colorama import Fore, Back, Style, init

# Inicializa colorama (strip=False mantiene los códigos ANSI en todos los SO)
init(autoreset=True)


# ── Constantes de estilo ──────────────────────────────────────────────────────

ANCHO: int = min(shutil.get_terminal_size(fallback=(72, 24)).columns, 72)

C_TITULO    = Fore.CYAN   + Style.BRIGHT
C_SUBTITULO = Fore.WHITE  + Style.BRIGHT
C_OK        = Fore.GREEN  + Style.BRIGHT
C_WARN      = Fore.YELLOW + Style.BRIGHT
C_ERROR     = Fore.RED    + Style.BRIGHT
C_DIM       = Fore.WHITE  + Style.DIM
C_ACENTO    = Fore.MAGENTA + Style.BRIGHT
C_OPCION    = Fore.CYAN
C_RESET     = Style.RESET_ALL


# ── Helpers de pantalla ───────────────────────────────────────────────────────

def limpiar() -> None:
    """Limpia la pantalla del terminal (compatible con Windows y Unix)."""
    os.system("cls" if os.name == "nt" else "clear")


def linea(car: str = "─", color: str = C_DIM) -> None:
    """Imprime una línea separadora del ancho del terminal."""
    print(color + car * ANCHO + C_RESET)


def salto() -> None:
    print()


# ── Bloques estructurales ─────────────────────────────────────────────────────

def cabecera(titulo: str, subtitulo: str = "") -> None:
    """Imprime la cabecera principal de pantalla."""
    limpiar()
    linea("═", C_TITULO)
    _centrar(f"🍷  {titulo}", C_TITULO)
    if subtitulo:
        _centrar(subtitulo, C_DIM)
    linea("═", C_TITULO)
    salto()


def seccion(titulo: str) -> None:
    """Imprime un encabezado de sección."""
    salto()
    linea("─", C_ACENTO)
    print(C_ACENTO + f"  ▶  {titulo.upper()}" + C_RESET)
    linea("─", C_ACENTO)


def _centrar(texto: str, color: str = "") -> None:
    """Imprime texto centrado en el ancho de la pantalla."""
    # Elimina códigos ANSI para calcular longitud real
    import re
    limpio = re.sub(r"\x1b\[[0-9;]*m", "", texto)
    pad = max((ANCHO - len(limpio)) // 2, 0)
    print(" " * pad + color + texto + C_RESET)


# ── Mensajes de estado ────────────────────────────────────────────────────────

def ok(msg: str) -> None:
    print(C_OK + f"  ✔  {msg}" + C_RESET)


def warn(msg: str) -> None:
    print(C_WARN + f"  ⚠  {msg}" + C_RESET)


def error(msg: str) -> None:
    print(C_ERROR + f"  ✘  {msg}" + C_RESET)


def info(msg: str) -> None:
    print(C_DIM + f"     {msg}" + C_RESET)


# ── Renderizado del menú ──────────────────────────────────────────────────────

def mostrar_menu(secciones: list[dict], estado) -> None:
    """
    Renderiza el menú principal con el estado de cada opción.

    Args:
        secciones: Lista de dicts del catálogo MENU_SECTIONS (config.py).
        estado: Instancia de SessionState para leer flags.
    """
    print(C_SUBTITULO + "  MENÚ PRINCIPAL" + C_RESET)
    linea()
    salto()

    for i, sec in enumerate(secciones, start=1):
        ok_flag, _ = estado.check_requires(sec["requires"])
        if ok_flag:
            prefijo = C_OPCION + f"  [{i}]" + C_RESET
            etiq    = C_SUBTITULO + f"  {sec['label']}" + C_RESET
        else:
            prefijo = C_DIM + f"  [{i}]" + C_RESET
            etiq    = C_DIM + f"  {sec['label']}  (requiere pasos previos)" + C_RESET
        print(prefijo + etiq)

    salto()
    linea()
    print(C_DIM + "  [s]  Estado de la sesión" + C_RESET)
    print(C_ERROR + "  [0]  Salir" + C_RESET)
    salto()


def mostrar_estado(estado) -> None:
    """Muestra el panel de estado de la sesión actual."""
    seccion("Estado de la sesión")
    for descripcion, icono in estado.resumen_estado().items():
        color = C_OK if icono == "✔" else C_DIM
        print(color + f"  {icono}  {descripcion}" + C_RESET)
    salto()

    if estado.modelos_entrenados:
        modelos = list(estado.trained_models.keys())
        print(C_DIM + f"  Modelos entrenados  : {', '.join(modelos)}" + C_RESET)
    if estado.modelos_evaluados:
        mejor = max(
            estado.evaluation_results,
            key=lambda k: estado.evaluation_results[k]["accuracy"]
        )
        acc = estado.evaluation_results[mejor]["accuracy"]
        print(C_DIM + f"  Mejor modelo        : {mejor}  ({acc * 100:.2f}%)" + C_RESET)
    salto()


# ── Input con validación ──────────────────────────────────────────────────────

def pedir_opcion(prompt: str = "  Selecciona una opción: ") -> str:
    """Solicita entrada al usuario con formato consistente."""
    try:
        return input(C_ACENTO + prompt + C_RESET).strip().lower()
    except (KeyboardInterrupt, EOFError):
        return "0"


def pausar(msg: str = "Pulsa ENTER para volver al menú...") -> None:
    """Pausa la ejecución esperando que el usuario pulse ENTER."""
    try:
        input(C_DIM + f"\n  {msg}" + C_RESET)
    except (KeyboardInterrupt, EOFError):
        pass
