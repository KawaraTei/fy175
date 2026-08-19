from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import numpy as np
from PIL import Image
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import ProcessingResult
from auto_mosaic.image_ops import load_image_bgr
from auto_mosaic.ui import AutoMosaicWindow


def test_drop_multiple_images_and_analyze_selected_automatically() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    assert window.windowTitle() == "FY175AutoMosaic"
    assert window.mode_combo.currentText() == "イラスト"
    window.show()
    app.processEvents()

    def fake_analyze(path, _settings):
        image = load_image_bgr(path)
        return ProcessingResult(
            source_path=path,
            image_bgr=image,
            mask=np.zeros(image.shape[:2], dtype=bool),
        )

    window.pipeline.analyze = fake_analyze  # type: ignore[method-assign]

    with TemporaryDirectory() as temporary:
        folder = Path(temporary)
        first = folder / "first.png"
        second = folder / "second.jpg"
        first_pixels = np.arange(12 * 8 * 3, dtype=np.uint8).reshape(8, 12, 3)
        Image.fromarray(first_pixels).save(first)
        Image.new("RGB", (10, 10), (20, 80, 160)).save(second)

        mime_data = QMimeData()
        mime_data.setUrls([QUrl.fromLocalFile(str(first)), QUrl.fromLocalFile(str(second))])
        event = QDropEvent(
            QPointF(10, 10),
            Qt.DropAction.CopyAction,
            mime_data,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.dropEvent(event)

        deadline = monotonic() + 2.0
        while window.current_result is None and monotonic() < deadline:
            app.processEvents()
            window._poll_events()
            sleep(0.01)

        assert event.isAccepted()
        assert window.image_paths == [first.resolve(), second.resolve()]
        assert window.current_result is not None
        assert window.current_result.source_path == first.resolve()
        assert window.preview_mode_combo.currentText() == "処理結果"
        assert window.suffix_edit.text() == "_mosaic"
        assert window.process_current_button.text() == "表示中の1枚を処理して保存"
        assert not window.remove_after_process_check.isChecked()
        assert window.list_splitter.orientation() == Qt.Orientation.Vertical
        assert window.list_splitter.count() == 2

        window.preview_mode_combo.setCurrentText("検出範囲")
        app.processEvents()
        assert window.mask_edit_button.isEnabled()
        window._toggle_mask_edit()
        assert window.mask_edit_active
        assert window.preview_label.mask_editing
        assert window.preview_mode_combo.isEnabled()
        assert not window.process_button.isEnabled()
        assert window.process_current_button.isEnabled()
        assert not window.brush_size_slider.isHidden()
        border_color = window.preview_label.pixmap().toImage().pixelColor(2, 2)
        assert border_color.red() > 200
        assert border_color.green() < 120

        window.brush_size_slider.setValue(4)
        center = window.preview_image_rect.center()
        QTest.mouseMove(window.preview_label, center.toPoint())
        assert window.preview_label.brush_position is not None
        assert window.preview_label.brush_diameter > 0
        QTest.mouseClick(
            window.preview_label,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            center.toPoint(),
        )
        assert window.mask_edit_dirty
        assert window.edited_mask is not None
        assert np.any(window.edited_mask)
        QTest.mouseClick(
            window.preview_label,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.AltModifier,
            center.toPoint(),
        )
        assert not window.edited_mask[
            window.edited_mask.shape[0] // 2,
            window.edited_mask.shape[1] // 2,
        ]
        retained_point = QPointF(
            center.x() + window.preview_image_rect.width() / 4,
            center.y(),
        )
        QTest.mouseClick(
            window.preview_label,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            retained_point.toPoint(),
        )
        assert np.any(window.edited_mask)

        edited_before_preview_switch = window.edited_mask.copy()
        window.preview_mode_combo.setCurrentText("処理結果")
        app.processEvents()
        assert window.mask_edit_active
        assert not window.preview_label.mask_editing
        assert window.preview_label.brush_position is None
        assert np.array_equal(window.edited_mask, edited_before_preview_switch)
        assert window.preview_rgb is not None
        assert np.any(window.preview_rgb != first_pixels)
        result_border = window.preview_label.pixmap().toImage().pixelColor(2, 2)
        assert result_border.red() > 200
        window.preview_mode_combo.setCurrentText("検出範囲")
        app.processEvents()
        assert window.mask_edit_active
        assert window.preview_label.mask_editing
        assert np.array_equal(window.edited_mask, edited_before_preview_switch)

        original_confirm = window._confirm_discard_mask_edit
        window._confirm_discard_mask_edit = lambda *_args, **_kwargs: False  # type: ignore[method-assign]
        window.file_list.setCurrentRow(1)
        app.processEvents()
        assert window.file_list.currentRow() == 0
        assert window.mask_edit_active
        window._confirm_discard_mask_edit = original_confirm  # type: ignore[method-assign]
        window._end_mask_edit(refresh_preview=True)

        window.file_list.clearSelection()
        window.file_list.setCurrentRow(1)
        window.file_list.item(1).setSelected(True)
        app.processEvents()
        window._remove_selected()
        assert window.image_paths == [first.resolve()]
        assert window.manual_review_paths == []

        window._add_image_paths([second])
        window.file_list.clearSelection()
        window.file_list.setCurrentRow(1)
        app.processEvents()
        window._remove_selected_to_manual()
        assert window.image_paths == [first.resolve()]
        assert window.manual_review_paths == [second.resolve()]
        assert window.manual_review_list.item(0).text() == "second.jpg"

        window._clear_manual_review()
        assert window.manual_review_paths == []
        assert window.manual_review_list.count() == 0

        window._add_image_paths([second])
        window._remove_processed_paths([first.resolve()])
        assert window.image_paths == [second.resolve()]
        assert window.manual_review_paths == []

    window.close()
