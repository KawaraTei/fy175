from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic, sleep

import numpy as np
from PIL import Image
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDropEvent
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import ProcessingResult
from auto_mosaic.image_ops import load_image_bgr
from auto_mosaic.ui import AutoMosaicWindow


def test_drop_multiple_images_and_analyze_selected_automatically() -> None:
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    assert window.windowTitle() == "FY175AutoMosaic"
    assert window.mode_combo.currentText() == "イラスト"

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
        Image.new("RGB", (12, 8), (200, 100, 50)).save(first)
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

        window.file_list.setCurrentRow(1)
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
