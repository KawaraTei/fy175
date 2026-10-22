"""Platform paths and user-facing names shared by image and video mode."""
from pathlib import Path
import sys


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[1]


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS).resolve()
    return Path(__file__).resolve().parents[1]


def default_output_dir() -> Path:
    if sys.platform == "darwin":
        from PySide6.QtCore import QStandardPaths

        documents = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DocumentsLocation)
        root = Path(documents) if documents else Path.home() / "Documents"
        return root / "FY175AutoMosaic" / "output"
    return application_root() / "output"


def command_key() -> str:
    # Qt's ControlModifier / Ctrl key sequences already map to Command on macOS.
    return "Cmd" if sys.platform == "darwin" else "Ctrl"


def option_key() -> str:
    return "Option" if sys.platform == "darwin" else "Alt"


def file_manager_name() -> str:
    return "Finder" if sys.platform == "darwin" else "エクスプローラ"
