$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    throw 'Run scripts\build.ps1 once to create the local environment.'
}

& '.venv\Scripts\python.exe' -m compileall -q auto_mosaic scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_detector import *; from tests.test_image_ops import *; from tests.test_pipeline import *; from tests.test_segmenter import *; test_as_rows_transposes_yolo_channel_first_output(); test_as_rows_keeps_prediction_rows(); test_mosaic_changes_only_masked_pixels(); test_paint_mask_stroke_adds_and_erases_with_round_brush(); test_bounded_mask_clips_distant_pixels(); test_refine_mask_expands_mask(); test_center_anchored_component_discards_larger_unrelated_region(); test_center_anchored_component_rejects_mask_outside_center_area(); test_save_uses_custom_suffix(); test_save_allows_empty_suffix_without_overwrite(); test_process_with_mask_uses_edited_mask_without_detection(); test_analyze_prefers_centered_mask_over_higher_scored_unrelated_mask(); test_analyze_falls_back_to_detection_box_when_no_mask_is_centered(); test_box_prompt_includes_positive_center_point(); print('unit checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:QT_QPA_PLATFORM = 'offscreen'
& '.venv\Scripts\python.exe' -c "from tests.test_ui import test_drop_multiple_images_and_analyze_selected_automatically, test_mask_edit_wheel_zoom_uses_cursor_or_image_center_anchor, test_remove_selected_moves_to_next_image_or_clears_tail_selection; test_drop_multiple_images_and_analyze_selected_automatically(); test_mask_edit_wheel_zoom_uses_cursor_or_image_center_anchor(); test_remove_selected_moves_to_next_image_or_clears_tail_selection(); print('UI interaction checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
& '.venv\Scripts\python.exe' -m tests.smoke_models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
