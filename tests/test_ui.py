from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, QRectF, Qt, QUrl
from PySide6.QtGui import QDropEvent, QMouseEvent, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import Detection, ProcessingResult, ProcessingSettings
from auto_mosaic.image_ops import load_image_bgr
from auto_mosaic.ui import AutoMosaicWindow



def _send_wheel(widget, position: QPointF, delta: int) -> None:
    event = QWheelEvent(
        position,
        position,
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def test_mask_settings_reach_analysis_and_saving_and_lock_during_editing() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    window.show()
    app.processEvents()
    received = []

    def fake_analyze(path, settings):
        received.append(settings)
        image = load_image_bgr(path)
        return ProcessingResult(path, image, np.zeros(image.shape[:2], dtype=bool))

    def wait_for_result():
        deadline = monotonic() + 3.0
        while monotonic() < deadline:
            app.processEvents()
            window._poll_events()
            if not window.busy and window.current_result is not None:
                return
            sleep(0.01)
        raise AssertionError("analysis or saving did not complete")

    window.pipeline.analyze = fake_analyze  # type: ignore[method-assign]
    try:
        assert window.mask_threshold_spin.value() == ProcessingSettings.mask_threshold
        assert window.mask_expansion_spin.value() == ProcessingSettings.mask_expansion
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "sample.png"
            Image.new("RGB", (64, 64), (100, 120, 140)).save(source)
            window.output_edit.setText(str(folder / "output"))
            window._add_image_paths([source])
            wait_for_result()
            original_result = window.current_result
            QTest.keyClick(window.mask_threshold_spin, Qt.Key.Key_Up)
            assert window.mask_threshold_spin.value() == 0.5
            window.mask_threshold_spin.setValue(1.5)
            window.mask_expansion_spin.setValue(0)
            assert window.current_result is original_result
            assert len(received) == 1
            QTest.mouseClick(window.analyze_button, Qt.MouseButton.LeftButton)
            assert not window.mask_threshold_spin.isEnabled()
            assert not window.mask_expansion_spin.isEnabled()
            wait_for_result()
            assert received[-1].mask_threshold == 1.5
            assert received[-1].mask_expansion == 0
            assert received[-1].confidence_threshold == 0.25
            window.set_preview_mode("検出範囲")
            QTest.mouseClick(window.mask_edit_button, Qt.MouseButton.LeftButton)
            assert window.mask_edit_active
            assert window.mask_threshold_spin.isEnabled()
            assert not window.mask_expansion_spin.isEnabled()
            window._end_mask_edit(refresh_preview=True)
            assert window.mask_threshold_spin.isEnabled()
            assert window.mask_expansion_spin.isEnabled()
            with patch("auto_mosaic.ui.QMessageBox.information"):
                QTest.mouseClick(window.process_current_button, Qt.MouseButton.LeftButton)
                wait_for_result()
                assert (folder / "output" / "sample_mosaic.png").exists()
                QTest.mouseClick(window.process_button, Qt.MouseButton.LeftButton)
                wait_for_result()
                assert (folder / "output" / "sample_mosaic_2.png").exists()
            assert len(received) == 4
            assert all(settings.mask_threshold == 1.5 for settings in received[1:])
            assert all(settings.mask_expansion == 0 for settings in received[1:])
    finally:
        window._end_mask_edit(refresh_preview=False)
        window.close()


def test_shift_click_region_add_erase_and_error_preserve_draft() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    window.show()
    app.processEvents()

    def wait_idle():
        deadline = monotonic() + 3
        while monotonic() < deadline:
            app.processEvents()
            window._poll_events()
            if not window.busy:
                return
            sleep(0.01)
        raise AssertionError("region selection did not finish")

    try:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.png"
            Image.new("RGB", (100, 80), (80, 120, 160)).save(source)
            window.pipeline.analyze = lambda path, settings: ProcessingResult(
                path, load_image_bgr(path), np.zeros((80, 100), dtype=bool)
            )
            window._add_image_paths([source])
            wait_idle()
            window.set_preview_mode("マスク範囲")
            window._toggle_mask_edit()
            window.mask_threshold_spin.setValue(1.5)
            window._zoom_preview(window.preview_image_rect.center(), 1)
            point = window.preview_image_rect.center().toPoint()
            expected_point = window._preview_point_to_mask(QPointF(point))
            region = np.zeros((80, 100), dtype=bool)
            region[25:55, 35:65] = True
            region[37:43, 47:53] = False  # A brush dab would incorrectly fill this hole.
            encoded = object()
            with patch.object(window.pipeline, "select_mask_region", return_value=(region, encoded)) as select:
                QTest.mouseClick(window.preview_label, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.ShiftModifier, point)
                assert not window.mask_threshold_spin.isEnabled()
                wait_idle()
                assert np.array_equal(window.edited_mask, region)
                assert window.mask_edit_dirty
                assert select.call_args.args[1:] == (expected_point, 1.5, None)
                window.edited_mask[0, 0] = True
                window.mask_threshold_spin.setValue(-0.5)
                QTest.mouseClick(window.preview_label, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier, point)
                wait_idle()
                assert window.edited_mask.sum() == 1 and window.edited_mask[0, 0]
                assert select.call_args.args[1:] == (expected_point, -0.5, encoded)
                assert window.mask_threshold_spin.isEnabled()
                window._zoom_preview(window.preview_image_rect.center(), -1)
                assert not window.preview_image_rect.contains(QPointF(1, 1))
                QTest.mouseClick(window.preview_label, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.ShiftModifier, QPoint(1, 1))
                assert select.call_count == 2
                window.activateWindow()
                QTest.qWait(20)
                QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
                assert np.array_equal(window.edited_mask[1:], region[1:])
                QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
                assert not window.edited_mask.any() and not window.mask_edit_dirty
            before = window.edited_mask.copy()
            with patch.object(window.pipeline, "select_mask_region", side_effect=RuntimeError("test")), patch.object(window, "_notice") as notice:
                QTest.mouseClick(window.preview_label, Qt.MouseButton.LeftButton,
                                 Qt.KeyboardModifier.ShiftModifier, point)
                wait_idle()
                assert np.array_equal(window.edited_mask, before)
                assert window.preview_label.mask_editing
                notice.assert_called_once()
            # Results from an ended edit session must never modify a new draft.
            old_draft = window.edited_mask
            window._end_mask_edit(False)
            window._toggle_mask_edit()
            window.events.put(("mask_region", (window.analysis_generation, old_draft, False, region, encoded)))
            window._poll_events()
            assert not window.edited_mask.any()
    finally:
        window._end_mask_edit(False)
        window.close()


def test_edit_from_any_preview_and_undo_whole_strokes() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    window.show()
    window.activateWindow()
    QTest.qWait(30)
    try:
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / "sample.png"
            Image.new("RGB", (160, 100), (80, 120, 160)).save(source)
            window.pipeline.analyze = lambda path, settings: ProcessingResult(
                path, load_image_bgr(path), np.zeros((100, 160), dtype=bool)
            )
            window._add_image_paths([source])
            deadline = monotonic() + 3
            while monotonic() < deadline:
                app.processEvents()
                window._poll_events()
                if not window.busy and window.current_result is not None:
                    break
                sleep(.01)
            for view in ("元画像", "処理結果", "検出範囲", "マスク範囲"):
                QTest.mouseClick(window.preview_mode_buttons[view], Qt.MouseButton.LeftButton)
                assert window.preview_mode() == view
                assert sum(button.isChecked() for button in window.preview_mode_buttons.values()) == 1
                assert window.mask_edit_button.isEnabled()
                window.mask_edit_button.click()
                assert window.mask_edit_active and window.preview_label.mask_editing
                assert window.preview_mode() == (view if view in {"検出範囲", "マスク範囲"} else "マスク範囲")
                window._end_mask_edit(True)
            window._toggle_mask_edit()
            window.brush_size_slider.setValue(8)
            rect = window.preview_image_rect
            positions = [QPointF(rect.left() + rect.width() * x, rect.center().y()) for x in (.2, .4, .6)]

            def drag(modifiers):
                QTest.mousePress(window.preview_label, Qt.MouseButton.LeftButton, modifiers, positions[0].toPoint())
                for pos in positions[1:]:
                    event = QMouseEvent(QEvent.Type.MouseMove, pos, pos,
                                        Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, modifiers)
                    QApplication.sendEvent(window.preview_label, event)
                QTest.mouseRelease(window.preview_label, Qt.MouseButton.LeftButton, modifiers, positions[-1].toPoint())

            def undo():
                QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
                app.processEvents()

            drag(Qt.KeyboardModifier.NoModifier)
            added = window.edited_mask.copy()
            assert added[50, 32] and added[50, 64] and added[50, 96]
            assert len(window.mask_undo_history) == 1
            drag(Qt.KeyboardModifier.AltModifier)
            assert not window.edited_mask.any() and len(window.mask_undo_history) == 2
            undo()
            assert np.array_equal(window.edited_mask, added)
            undo()
            assert not window.edited_mask.any() and not window.mask_edit_dirty
            assert not window.mask_undo_history
            drag(Qt.KeyboardModifier.AltModifier)
            assert not window.mask_undo_history
            drag(Qt.KeyboardModifier.NoModifier)
            window.set_preview_mode("処理結果")
            undo()
            assert not window.edited_mask.any()
            window._end_mask_edit(True)
            window._toggle_mask_edit()
            assert not window.mask_undo_history and not window.mask_undo_shortcut.isEnabled()
    finally:
        window._end_mask_edit(False)
        window.close()


def test_preview_wheel_zoom_uses_cursor_or_image_center_anchor() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    window.show()
    app.processEvents()

    window.preview_rgb = np.zeros((300, 500, 3), dtype=np.uint8)
    window._update_mask_editor_interaction()
    window._render_preview()
    initial_rect = QRectF(window.preview_image_rect)

    cursor = QPointF(
        initial_rect.left() + initial_rect.width() * 0.7,
        initial_rect.top() + initial_rect.height() * 0.4,
    )
    _send_wheel(window.preview_label, cursor, 120)
    zoomed_rect = QRectF(window.preview_image_rect)
    assert zoomed_rect.width() > initial_rect.width()
    assert abs((cursor.x() - zoomed_rect.left()) / zoomed_rect.width() - 0.7) < 0.01
    assert abs((cursor.y() - zoomed_rect.top()) / zoomed_rect.height() - 0.4) < 0.01

    outside = QPointF(1, 1)
    assert not zoomed_rect.contains(outside)
    center_before = zoomed_rect.center()
    _send_wheel(window.preview_label, outside, 120)
    center_after = window.preview_image_rect.center()
    assert abs(center_after.x() - center_before.x()) < 1.0
    assert abs(center_after.y() - center_before.y()) < 1.0

    while window.preview_zoom_index > 0:
        _send_wheel(window.preview_label, window.preview_image_rect.center(), -120)
    minimum_rect = QRectF(window.preview_image_rect)
    _send_wheel(window.preview_label, minimum_rect.center(), -120)
    assert window.preview_zoom_index == 0
    assert abs(window.preview_image_rect.width() - initial_rect.width()) < 1.0
    assert abs(window.preview_image_rect.height() - initial_rect.height()) < 1.0

    window.close()


def test_copy_current_preserves_original_without_processing() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    try:
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "sample.JPG"
            Image.new("RGB", (12, 8), (20, 80, 160)).save(source)
            original_bytes = source.read_bytes()
            window.image_paths = [source]
            window.file_list.blockSignals(True)
            window.file_list.addItem(source.name)
            window.file_list.setCurrentRow(0)
            window.file_list.blockSignals(False)
            window.output_edit.setText(str(folder / "output"))
            window.suffix_edit.setText("invalid/suffix")
            window.penis_check.setChecked(False)
            window.vagina_check.setChecked(False)
            window.remove_after_process_check.setChecked(True)
            window.current_result = ProcessingResult(
                source, load_image_bgr(source), np.ones((8, 12), dtype=bool)
            )
            window.set_preview_mode("検出範囲")
            window._toggle_mask_edit()
            window.mask_edit_dirty = True
            result = window.current_result
            edited_mask = window.edited_mask.copy()

            def wait_for_copy():
                deadline = monotonic() + 3.0
                while window.busy and monotonic() < deadline:
                    app.processEvents()
                    window._poll_events()
                    sleep(0.01)
                assert not window.busy

            with patch.object(window.pipeline, "analyze") as analyze, patch.object(
                window.pipeline, "process_with_mask"
            ) as process, patch.object(window.pipeline, "save") as save, patch.object(
                window, "_notice"
            ) as notice:
                window.copy_current_button.click()
                assert not window.copy_current_button.isEnabled()
                wait_for_copy()
                output = folder / "output" / source.name
                assert output.read_bytes() == original_bytes
                window.copy_current_button.click()
                wait_for_copy()
                assert (output.parent / "sample_2.JPG").read_bytes() == original_bytes
                assert output.read_bytes() == original_bytes
                # A source folder used as the output must not overwrite the source.
                window.output_edit.setText(str(folder))
                window.copy_current_button.click()
                wait_for_copy()
                assert (folder / "sample_2.JPG").read_bytes() == original_bytes
                assert source.read_bytes() == original_bytes
                assert window.current_result is result
                assert window.mask_edit_active and window.mask_edit_dirty
                assert np.array_equal(window.edited_mask, edited_mask)
                assert window.image_paths == [source]
                analyze.assert_not_called()
                process.assert_not_called()
                save.assert_not_called()
                assert notice.call_args.args[0] == "information"
                # A failed copy reports an error and re-enables the action.
                window.output_edit.setText(str(source))
                window.copy_current_button.click()
                wait_for_copy()
                assert notice.call_args.args[0] == "critical"
                assert window.copy_current_button.isEnabled()
    finally:
        window._end_mask_edit(refresh_preview=False)
        window.close()


def test_open_output_folder_button_creates_and_reveals_folder() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    with TemporaryDirectory() as temporary:
        output_dir = Path(temporary) / "new-output"
        window.output_edit.setText(str(output_dir))

        with patch(
            "auto_mosaic.ui.QDesktopServices.openUrl", return_value=True
        ) as open_url:
            window.open_output_button.click()

        assert window.open_output_button.text() == "エクスプローラで表示"
        assert output_dir.is_dir()
        assert Path(open_url.call_args.args[0].toLocalFile()) == output_dir.resolve()
    window.close()


def test_remove_selected_moves_to_next_image_or_clears_tail_selection() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
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
        paths = [folder / f"image-{index}.png" for index in range(3)]
        colors = [(20, 40, 60), (80, 100, 120), (140, 160, 180)]
        for path, color in zip(paths, colors):
            Image.new("RGB", (12, 8), color).save(path)

        window._add_image_paths(paths)
        window.file_list.clearSelection()
        window.file_list.setCurrentRow(1)
        window.file_list.item(1).setSelected(True)
        app.processEvents()
        window._remove_selected()

        assert window.image_paths == [paths[0].resolve(), paths[2].resolve()]
        assert window.file_list.currentRow() == 1
        assert window._selected_path() == paths[2].resolve()
        assert window.preview_rgb is not None
        assert tuple(window.preview_rgb[0, 0]) == colors[2]

        window._remove_selected()

        assert window.image_paths == [paths[0].resolve()]
        assert window.file_list.currentRow() == -1
        assert window.file_list.selectedItems() == []
        assert window.preview_rgb is None
        assert window.preview_label.pixmap().isNull()
        assert window.preview_label.text() == "プレビュー"

    window.close()


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
            detections=[Detection("penis", 0.9, (1, 1, 10, 7))],
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
        assert window.preview_mode() == "処理結果"
        assert window.brush_size_slider.value() == 20
        assert window.brush_size_slider.maximum() == 100
        assert window.brush_size_value.text() == "20 px"
        assert window.suffix_edit.text() == "_mosaic"
        assert window.process_current_button.text() == "表示中の1枚を処理して保存"
        assert not window.remove_after_process_check.isChecked()
        assert window.list_splitter.orientation() == Qt.Orientation.Vertical
        assert window.list_splitter.count() == 2

        for preview_mode in ("元画像", "検出範囲", "マスク範囲", "処理結果"):
            QTest.mouseClick(window.preview_mode_buttons[preview_mode], Qt.MouseButton.LeftButton)
            assert window.preview_mode() == preview_mode
            assert sum(button.isChecked() for button in window.preview_mode_buttons.values()) == 1
            app.processEvents()
            initial_width = window.preview_image_rect.width()
            _send_wheel(
                window.preview_label,
                window.preview_image_rect.center(),
                120,
            )
            assert window.preview_zoom_index == 1
            assert window.preview_image_rect.width() > initial_width
            _send_wheel(
                window.preview_label,
                window.preview_image_rect.center(),
                -120,
            )
            assert window.preview_zoom_index == 0

        window.set_preview_mode("検出範囲")
        app.processEvents()
        assert window.preview_rgb is not None
        assert np.any(window.preview_rgb != first_pixels)
        assert window.preview_label.zoom_enabled
        assert window.mask_edit_button.isEnabled()
        window._toggle_mask_edit()
        assert window.mask_edit_active
        assert window.preview_rgb is not None
        assert np.array_equal(window.preview_rgb, first_pixels)
        assert window.preview_label.mask_editing
        assert all(button.isEnabled() for button in window.preview_mode_buttons.values())
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
        window.set_preview_mode("処理結果")
        app.processEvents()
        assert window.mask_edit_active
        assert not window.preview_label.mask_editing
        assert window.preview_label.brush_position is None
        assert np.array_equal(window.edited_mask, edited_before_preview_switch)
        assert window.preview_rgb is not None
        assert np.any(window.preview_rgb != first_pixels)
        result_border = window.preview_label.pixmap().toImage().pixelColor(2, 2)
        assert result_border.red() > 200
        window.set_preview_mode("検出範囲")
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
