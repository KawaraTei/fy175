"""Shared collapsible settings section and automatic-mask smoothing control."""
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget

from auto_mosaic.domain import PROCESSING_RANGES, ProcessingSettings


class AdvancedSettingsSection(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("advancedToggle")
        self.toggle_button.setAccessibleName("詳細設定")
        self.toggle_button.setCheckable(True)
        layout.addWidget(self.toggle_button)
        self.content = QWidget()
        self.settings_layout = QVBoxLayout(self.content)
        self.settings_layout.setContentsMargins(0, 10, 0, 10)
        self.settings_layout.setSpacing(7)
        layout.addWidget(self.content)
        self.toggle_button.toggled.connect(self._set_expanded)
        self.toggle_button.setChecked(True)

    def _set_expanded(self, expanded):
        self.content.setVisible(expanded)
        self.toggle_button.setText(("▼" if expanded else "▶") + " 詳細設定")
        self.toggle_button.setToolTip("詳細設定を折り畳む" if expanded else "詳細設定を展開する")


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
