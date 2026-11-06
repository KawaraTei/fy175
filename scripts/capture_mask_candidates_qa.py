"""Native Qt + real SAM2 verification with a non-private occlusion fixture."""
from pathlib import Path
from time import monotonic, sleep

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from auto_mosaic.domain import Detection, ProcessingResult
from auto_mosaic.image_ops import save_image_bgr, apply_effect, bounded_mask
from auto_mosaic.ui import AutoMosaicWindow


def main():
    destination = Path('.codex-qa/2026-11-06/mask-candidates').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source = destination / 'occlusion.png'
    image = np.full((480, 640, 3), 235, np.uint8)
    cv2.ellipse(image, (320, 245), (245, 85), 0, 0, 360, (90, 170, 225), -1)
    cv2.ellipse(image, (320, 245), (245, 85), 0, 0, 360, (45, 80, 110), 3)
    points = np.array([[320, 45], [395, 65], [315, 440], [240, 420]], np.int32)
    cv2.fillPoly(image, [points], (165, 100, 80))
    cv2.polylines(image, [points], True, (85, 45, 30), 3)
    for text, point in [('A', (145, 255)), ('A', (460, 255)), ('B', (300, 245))]:
        cv2.putText(image, text, point, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 2, cv2.LINE_AA)
    save_image_bgr(source, image)
    app = QApplication.instance() or QApplication([])
    window = AutoMosaicWindow()
    window.show()
    window.move(25, 25)
    window.mask_expansion_spin.setValue(0)
    box = (72, 157, 568, 333)
    # Supply the reported wrong B mask explicitly; SAM2 generates alternatives live.
    wrong = np.zeros(image.shape[:2], np.uint8)
    cv2.fillPoly(wrong, [points], 1)
    wrong = bounded_mask(wrong.astype(bool), box)
    def analyze(path, settings):
        return ProcessingResult(path, apply_effect(image, wrong, settings.effect, settings.effect_size),
            wrong, [Detection('fixture-A', .95, box)], detection_masks=[wrong])
    window.pipeline.analyze = analyze

    def pump():
        app.processEvents()
        window._poll_events()
        sleep(.02)

    def wait():
        deadline = monotonic() + 60
        while monotonic() < deadline:
            pump()
            if not window.busy and window.current_result is not None:
                return
        raise AssertionError('inference timeout')

    def capture(name):
        for _ in range(5):
            pump()
        window.grab().save(str(destination / name))

    try:
        window._add_image_paths([source])
        wait()
        window.set_preview_mode('マスク範囲')
        capture('before.png')
        before = window.current_result.mask.copy()
        print(f'BEFORE area={before.sum()} A-left={before[245,180]} A-right={before[245,480]} B={before[260,315]}', flush=True)
        QTest.mouseClick(window.next_mask_button, Qt.MouseButton.LeftButton)
        wait()
        capture('after.png')
        after = window.edited_mask
        print(f'AFTER area={after.sum()} A-left={after[245,180]} A-right={after[245,480]} B={after[260,315]} STATUS={window.status_label.text()}', flush=True)
        if window._mask_candidates:
            for i, packed in enumerate(next(iter(window._mask_candidates.values()))):
                mask = np.unpackbits(packed, count=before.size).reshape(before.shape).astype(bool)
                print(f'CANDIDATE {i} area={mask.sum()} left={mask[245,180]} right={mask[245,480]} B={mask[260,315]}', flush=True)
        assert before[260, 315] and not before[245, 180]
        assert after[245, 180] and after[245, 480] and not after[260, 315]
        QTest.keyClick(window, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier)
        pump()
        np.testing.assert_array_equal(window.edited_mask, before)
        QTest.mouseClick(window.next_mask_button, Qt.MouseButton.LeftButton)
        capture('after-undo-and-next.png')
        window.resize(980, 650)
        capture('minimum.png')
        # Also verify target identification in a multiple-detection layout.
        window._end_mask_edit(False)
        window.current_result.detections.append(Detection('fixture-A', .9, (90, 175, 220, 310)))
        window.current_result.detection_masks.append(np.zeros_like(before))
        window._update_mask_edit_controls()
        window.mask_target_combo.setCurrentIndex(1)
        capture('multiple-targets-minimum.png')
        print(f'QA_PLATFORM={app.platformName()} CAPTURES={destination}', flush=True)
    finally:
        window._end_mask_edit(False)
        window.close()
        app.processEvents()


if __name__ == '__main__':
    main()
