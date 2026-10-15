"""Transactional, disk-backed manual masks and forward motion propagation."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from threading import Event

import cv2
import numpy as np

from auto_mosaic.video_analysis import FrameCache, VideoAnalysis, _encode
from auto_mosaic.video_io import check_cancel


@dataclass(frozen=True)
class MaskEditResult:
    start: int
    end: int
    edited: dict[int, str]
    can_undo: bool


def _validate(analysis: VideoAnalysis, start: int, end: int):
    if not analysis.complete or not 0 <= start <= end < len(analysis.frames):
        raise ValueError("解析済みのフレーム範囲を指定してください。")
    analysis.info.verify_source()


def _mask_bytes(mask):
    # Even an empty override must be distinct from an absent override.
    return _encode(".png", mask.astype(np.uint8) * 255, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _snapshot(connection, start, end):
    connection.execute("DELETE FROM mask_undo")
    connection.execute("""INSERT INTO mask_undo
        SELECT f.frame, e.mask, e.removed, e.kind FROM frames f
        LEFT JOIN mask_edits e ON e.frame=f.frame WHERE f.frame BETWEEN ? AND ?""", (start, end))


def _put(connection, frame, mask, removed, kind):
    connection.execute("INSERT OR REPLACE INTO mask_edits VALUES (?, ?, ?, ?)",
                       (frame, _mask_bytes(mask), _mask_bytes(removed), kind))


def _read(connection, frame, shape):
    row = connection.execute("""SELECT f.mask, f.preview, e.mask, e.removed FROM frames f
        LEFT JOIN mask_edits e ON e.frame=f.frame WHERE f.frame=?""", (frame,)).fetchone()
    if row is None:
        raise ValueError("フレームの解析キャッシュがありません。")
    original = FrameCache.decode_mask(row[0], shape)
    effective = original if row[2] is None else FrameCache.decode_mask(row[2], shape)
    removed = FrameCache.decode_mask(row[3], shape)
    gray = cv2.imdecode(np.frombuffer(row[1], np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("追従用のプレビューを読み込めません。再解析してください。")
    return original, effective, removed, gray


def _result(analysis, start, end):
    edits, can_undo = analysis.cache.edit_state()
    return MaskEditResult(start, end, edits, can_undo)


def save_frame_mask(analysis: VideoAnalysis, frame: int, mask: np.ndarray,
                    cancel: Event, progress) -> MaskEditResult:
    _validate(analysis, frame, frame)
    shape = (analysis.info.height, analysis.info.width)
    if mask.shape != shape or mask.dtype != np.bool_:
        raise ValueError("編集マスクのサイズまたは形式が一致しません。")
    started = time.perf_counter()
    with closing(sqlite3.connect(analysis.cache.path)) as connection, connection:
        original, previous, removed, _ = _read(connection, frame, shape)
        removed = (original | previous | removed) & ~mask
        _snapshot(connection, frame, frame)
        _put(connection, frame, mask, removed, "manual")
        progress(1, 1, time.perf_counter() - started)
        check_cancel(cancel)
        analysis.info.verify_source()
    return _result(analysis, frame, frame)


def _backward_flow(previous, current, active):
    """Cached previews bound the optical-flow cost independently of source size."""
    if previous.shape != current.shape:
        raise ValueError("追従用フレームのサイズが変わりました。")
    small_previous = cv2.resize(previous, (64, 36), interpolation=cv2.INTER_AREA)
    small_current = cv2.resize(current, (64, 36), interpolation=cv2.INTER_AREA)
    if np.abs(small_current.astype(float) - small_previous).mean() > 35:
        raise ValueError("シーンの切り替わり、または大きな明るさの変化を検出しました。")
    backward = cv2.calcOpticalFlowFarneback(current, previous, None, .5, 4, 21, 3, 5, 1.2, 0)
    forward = cv2.calcOpticalFlowFarneback(previous, current, None, .5, 4, 21, 3, 5, 1.2, 0)
    height, width = previous.shape
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x, map_y = x + backward[:, :, 0], y + backward[:, :, 1]
    sampled_forward = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR)
    residual = np.linalg.norm(backward + sampled_forward, axis=2)
    active_small = cv2.resize(active.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    following = cv2.remap(active_small, map_x, map_y, cv2.INTER_NEAREST) > 0
    # Do not silently carry a mask across lost/occluded content or out of frame.
    if following.sum() < max(1, active_small.sum() * .5):
        raise ValueError("追従する領域を見失いました。")
    valid = (map_x >= 0) & (map_x <= width - 1) & (map_y >= 0) & (map_y <= height - 1)
    consistent = valid & (residual < 2.5)
    if np.mean(consistent[following]) < .7:
        raise ValueError("動きの対応が不安定になりました。")
    reconstructed = cv2.remap(previous, map_x, map_y, cv2.INTER_LINEAR)
    if np.abs(reconstructed.astype(float) - current)[following].mean() > 30:
        raise ValueError("追従領域の見え方が大きく変わりました。")
    return backward


def _warp(mask, flow):
    height, width = mask.shape
    sy, sx = height / flow.shape[0], width / flow.shape[1]
    full_flow = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
    full_flow[:, :, 0] *= sx
    full_flow[:, :, 1] *= sy
    # Strip-wise coordinate maps avoid holding several full-size float grids.
    result = np.empty_like(mask)
    binary = mask.astype(np.uint8)
    x = np.arange(width, dtype=np.float32)[None, :]
    for top in range(0, height, 256):
        bottom = min(height, top + 256)
        y = np.arange(top, bottom, dtype=np.float32)[:, None]
        result[top:bottom] = cv2.remap(binary, x + full_flow[top:bottom, :, 0],
                                      y + full_flow[top:bottom, :, 1], cv2.INTER_NEAREST) > 0
    return result


def propagate_mask(analysis: VideoAnalysis, start: int, end: int,
                   cancel: Event, progress) -> MaskEditResult:
    """Follow the effective start mask, preserving other automatic detections.

    The tracked mask is added to each reviewed frame and tracked manual erasures
    are removed. Nothing commits on cancellation, a cut, or unreliable flow.
    """
    _validate(analysis, start, end)
    if start == end:
        raise ValueError("終点には始点より後のフレームを指定してください。")
    started = time.perf_counter()
    shape = (analysis.info.height, analysis.info.width)
    with closing(sqlite3.connect(analysis.cache.path)) as connection, connection:
        protected = connection.execute("SELECT frame FROM mask_edits WHERE frame>? AND frame<=? AND kind='manual' LIMIT 1",
                                       (start, end)).fetchone()
        if protected:
            raise ValueError(f"フレーム {protected[0] + 1} は手動編集済みです。その手前を終点にしてください。")
        _, tracked, erased, previous = _read(connection, start, shape)
        if not (tracked | erased).any():
            raise ValueError("追従するマスクがありません。始点にマスクを追加してください。")
        _snapshot(connection, start + 1, end)
        for frame in range(start + 1, end + 1):
            check_cancel(cancel)
            _, existing, existing_removed, current = _read(connection, frame, shape)
            try:
                flow = _backward_flow(previous, current, tracked | erased)
            except ValueError as error:
                raise ValueError(f"フレーム {frame + 1}: {error} 補完は反映されていません。終点を手前にしてください。") from error
            tracked, erased = _warp(tracked, flow), _warp(erased, flow)
            mask = (existing | tracked) & ~erased
            _put(connection, frame, mask, (existing_removed | erased) & ~mask, "propagated")
            previous = current
            progress(frame - start, end - start, time.perf_counter() - started)
        check_cancel(cancel)
        analysis.info.verify_source()
    return _result(analysis, start + 1, end)


def clear_mask_edits(analysis: VideoAnalysis, start: int, end: int,
                     cancel: Event, progress) -> MaskEditResult:
    """Restore original detections in the range, retaining one-step undo."""
    _validate(analysis, start, end)
    started = time.perf_counter()
    with closing(sqlite3.connect(analysis.cache.path)) as connection, connection:
        check_cancel(cancel)
        _snapshot(connection, start, end)
        connection.execute("DELETE FROM mask_edits WHERE frame BETWEEN ? AND ?", (start, end))
        progress(end - start + 1, end - start + 1, time.perf_counter() - started)
        check_cancel(cancel)
        analysis.info.verify_source()
    return _result(analysis, start, end)


def undo_mask_edit(analysis: VideoAnalysis, cancel: Event, progress) -> MaskEditResult:
    _validate(analysis, 0, len(analysis.frames) - 1)
    with closing(sqlite3.connect(analysis.cache.path)) as connection, connection:
        bounds = connection.execute("SELECT MIN(frame), MAX(frame) FROM mask_undo").fetchone()
        if bounds[0] is None:
            raise ValueError("戻せるマスク変更はありません。")
        connection.execute("DELETE FROM mask_edits WHERE frame IN (SELECT frame FROM mask_undo)")
        connection.execute("INSERT INTO mask_edits SELECT frame, mask, removed, kind FROM mask_undo WHERE kind IS NOT NULL")
        connection.execute("DELETE FROM mask_undo")
        check_cancel(cancel)
    return _result(analysis, *bounds)
