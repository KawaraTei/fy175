# Design QA: preview zoom and mask-edit annotations

## Visual truth

- User requirements:
  - Mouse-wheel zoom is available in every preview mode.
  - Zoom uses the cursor position while the cursor is inside the image and the image center otherwise.
  - Zoom cannot go below the initial fit-to-window size.
  - Detection rectangles and labels are hidden while mask editing.
  - The mask tint and contour remain visible for editing.
- Synthetic source: `source.png` (1080x720), shared by every capture.

## Current evidence

- `normal-detection-1x.png` and `normal-detection-1x-preview.png`
- `normal-detection-1-5x.png` and `normal-detection-1-5x-preview.png`
- `mask-edit-1-5x.png` and `mask-edit-1-5x-preview.png`
- Full window: 1260x790.
- Focused preview: 691x667.
- Platform: PySide6 offscreen rendering, dark theme, synthetic detection and mask data.

## Coverage inventory

| State | Scale | Expected | Result |
| --- | ---: | --- | --- |
| Detection preview | 1.0x | Accepted and below-threshold rectangles and labels visible | Passed |
| Detection preview | 1.5x | Cursor-anchored enlargement with annotations retained | Passed |
| Mask editing | 1.5x | Zoom retained; detection rectangles and labels hidden; mask tint and contour retained | Passed |
| Source, detection, result modes | 1.0x to 1.25x and back | Wheel zoom available; no shrink below initial fit | Passed by UI interaction test |

## Findings

- No P0, P1, or P2 findings.
- The normal 1.5x and mask-edit 1.5x focused captures use the same viewport and source position. The accepted yellow rectangle/label and gray candidate rectangle/label are present only in the normal capture. The mask overlay and yellow contour remain in the edit capture.
- The full-window captures show no new clipping, overlap, accidental scrollbars, or control geometry changes.
- The offscreen Qt platform does not resolve the Japanese UI font on this host, so Japanese typography was not judged. This change does not modify fonts, copy, layout, or control sizing.

## Verification

- Focused image-operation and UI checks: passed.
- Full unit, UI, detector, and SAM2 test script: passed.
- Rebuilt Windows EXE smoke test: passed (exit code 0).
- Evidence was captured after the final visible source changes.

final result: passed
