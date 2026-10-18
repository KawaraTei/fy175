"""Shared modeless settings window and automatic-mask smoothing control."""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget

from auto_mosaic.domain import PROCESSING_RANGES, ProcessingSettings


class AdvancedSettingsDialog(QDialog):
    def __init__(self, parent, context):
        super().__init__(parent)
        self.setWindowTitle(f"詳細設定 — {context}")
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.resize(360, 500)
        self.setMinimumSize(320, 340)
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        content = QWidget()
        self.settings_layout = QVBoxLayout(content)
        self.settings_layout.setSpacing(10)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        close = QPushButton("閉じる")
        close.clicked.connect(self.close)
        layout.addWidget(close)

    def show_settings(self):
        self.show()
        self.raise_()
        self.activateWindow()


def add_mask_smoothing(layout):
    row = QHBoxLayout()
    label = QLabel("マスクの滑らかさ")
    spin = QSpinBox()
    spin.setRange(*PROCESSING_RANGES["mask_smoothing"][:2])
    spin.setValue(ProcessingSettings.mask_smoothing)
    spin.setSuffix(" px")
    spin.setKeyboardTracking(False)
    spin.setAccessibleName(label.text())
    spin.setToolTip(
        "0で無効。大きいほど細かな凹凸を丸めます（元画像のピクセル単位）。\n"
        "輪郭が内外に動くため、再解析後にマスク範囲を確認してください。\n"
        "手動編集済みマスク・矩形補完には適用されません。"
    )
    label.setBuddy(spin)
    row.addWidget(label)
    row.addStretch(1)
    row.addWidget(spin)
    layout.addLayout(row)
    return spin
