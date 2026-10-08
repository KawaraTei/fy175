$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if (-not (Test-Path '.venv\Scripts\python.exe')) {
    throw 'Run scripts\build.ps1 once to create the local environment.'
}

& '.venv\Scripts\python.exe' -m compileall -q auto_mosaic scripts tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_detector import *; from tests.test_image_ops import *; from tests.test_pipeline import *; from tests.test_segmenter import *; test_as_rows_transposes_yolo_channel_first_output(); test_as_rows_keeps_prediction_rows(); test_mosaic_changes_only_masked_pixels(); test_paint_mask_stroke_adds_and_erases_with_round_brush(); test_bounded_mask_clips_distant_pixels(); test_refine_mask_expands_mask(); test_center_anchored_component_discards_larger_unrelated_region(); test_center_anchored_component_rejects_mask_outside_center_area(); test_visualize_detection_draws_below_threshold_boxes_in_gray(); test_visualize_detection_can_hide_detection_annotations(); test_save_uses_custom_suffix(); test_save_allows_empty_suffix_without_overwrite(); test_process_with_mask_uses_edited_mask_without_detection(); test_analyze_prefers_centered_mask_over_higher_scored_unrelated_mask(); test_analyze_falls_back_to_detection_box_when_no_mask_is_centered(); test_box_prompt_includes_positive_center_point(); print('unit checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$env:QT_QPA_PLATFORM = 'offscreen'
& '.venv\Scripts\python.exe' -m unittest tests.test_sidecar -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_ui import test_edit_from_any_preview_and_undo_whole_strokes; test_edit_from_any_preview_and_undo_whole_strokes(); print('mask entry and Ctrl+Z checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_segmenter import test_point_prompt_scales_coordinates_and_uses_mask_threshold; from tests.test_pipeline import test_point_region_keeps_seed_component_and_reuses_embedding; from tests.test_ui import test_shift_click_region_add_erase_and_error_preserve_draft; test_point_prompt_scales_coordinates_and_uses_mask_threshold(); test_point_region_keeps_seed_component_and_reuses_embedding(); test_shift_click_region_add_erase_and_error_preserve_draft(); print('point selection checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_image_ops import test_refine_mask_keeps_contour_correction_with_zero_or_small_expansion; test_refine_mask_keeps_contour_correction_with_zero_or_small_expansion(); print('independent contour correction check passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_segmenter import test_mask_threshold_changes_pixel_inclusion_without_changing_scores; from tests.test_ui import test_mask_settings_reach_analysis_and_saving_and_lock_during_editing; test_mask_threshold_changes_pixel_inclusion_without_changing_scores(); test_mask_settings_reach_analysis_and_saving_and_lock_during_editing(); print('mask settings checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_ui import test_drop_multiple_images_and_analyze_selected_automatically, test_preview_wheel_zoom_uses_cursor_or_image_center_anchor, test_remove_selected_moves_to_next_image_or_clears_tail_selection; test_drop_multiple_images_and_analyze_selected_automatically(); test_preview_wheel_zoom_uses_cursor_or_image_center_anchor(); test_remove_selected_moves_to_next_image_or_clears_tail_selection(); print('UI interaction checks passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
& '.venv\Scripts\python.exe' -m tests.smoke_models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -m unittest tests.test_agent_api -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& '.venv\Scripts\python.exe' -c "from tests.test_detector import test_preview_candidates_have_confidence_floor_without_overriding_user_threshold; test_preview_candidates_have_confidence_floor_without_overriding_user_threshold(); print('preview confidence floor check passed')"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
