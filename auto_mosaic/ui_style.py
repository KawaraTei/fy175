"""Canonical image-mode styling, shared by both independent windows."""
from PySide6.QtWidgets import QLabel


APP_STYLE = """
    QMainWindow, QWidget { background: #17191f; color: #d9dde5; font-family: "Segoe UI"; font-size: 10pt; }
    QFrame#toolbar, QFrame#panel, QFrame#previewBar { background: #20232b; }
    QFrame#previewPanel { background: #0e1015; border: 1px solid #303540; }
    QLabel#preview { color: #6f7787; background: #0e1015; }
    QLabel#section { color: #f0f2f6; font-weight: 600; }
    QLabel#muted { color: #8991a1; }
    QLabel#value { color: #53b8ff; }
    QPushButton { background: #356b92; color: white; border: none; border-radius: 3px; padding: 7px 12px; }
    QPushButton:hover { background: #4f89b2; }
    QPushButton:disabled { background: #3a3d45; color: #858993; }
    QListWidget, QLineEdit, QComboBox { background: #111319; color: #e5e8ef; border: 1px solid #353a46; border-radius: 3px; padding: 6px; }
    QListWidget::item { padding: 5px; }
    QListWidget::item:selected { background: #356b92; color: white; }
    QComboBox QAbstractItemView { background: #111319; color: #e5e8ef; selection-background-color: #356b92; }
    QCheckBox { spacing: 8px; }
    QCheckBox::indicator { width: 16px; height: 16px; }
    QSlider::groove:horizontal { height: 5px; background: #3a404d; border-radius: 2px; }
    QSlider::sub-page:horizontal { background: #53b8ff; border-radius: 2px; }
    QSlider::handle:horizontal { width: 15px; margin: -5px 0; background: #e9edf4; border-radius: 7px; }
    QProgressBar { background: #2a2e38; border: none; height: 7px; border-radius: 3px; }
    QProgressBar::chunk { background: #53b8ff; border-radius: 3px; }
    QSplitter::handle:vertical { background: #303540; height: 6px; margin: 2px 0; }
    QSplitter::handle:vertical:hover { background: #53b8ff; }
"""


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("section")
    return label
