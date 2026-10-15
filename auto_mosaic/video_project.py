"""Atomic video projects: a cache snapshot plus source identity and editor state."""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event

import numpy as np

from auto_mosaic.domain import EffectType
from auto_mosaic.sidecar import _read_settings
from auto_mosaic.video_analysis import FrameCache, FrameSummary, VideoAnalysis, VideoAnalysisSettings, _encode
from auto_mosaic.video_io import check_cancel, probe_video


PROJECT_SUFFIX = ".fyvideo"
PROJECT_FILTER = "動画編集プロジェクト (*.fyvideo)"


class ProjectSourceUnavailable(ValueError):
    """The user can select the original video without losing the current session."""


@dataclass(frozen=True)
class EditorState:
    frame: int = 0
    view: str = "検出範囲"
    timeline_seconds: int = 0
    tab: int = 0
    brush: int = 40
    start: int | None = None
    draft_dirty: bool = False

    def validate(self, count):
        for value, low, high in ((self.frame, 0, count - 1),
                                 (self.tab, 0, 1), (self.brush, 4, 200)):
            if type(value) is not int or not low <= value <= high:
                raise ValueError("保存された編集位置が範囲外です。")
        if self.start is not None and (type(self.start) is not int or not 0 <= self.start < count):
            raise ValueError("保存された補完の始点が不正です。")
        if (self.view not in {"元画像", "検出範囲", "処理結果"}
                or self.timeline_seconds not in {0, 1, 10} or type(self.draft_dirty) is not bool):
            raise ValueError("保存された表示設定が不正です。")


@dataclass
class LoadedProject:
    path: Path
    analysis: VideoAnalysis
    state: EditorState
    draft: np.ndarray | None


def _fingerprint(path, cancel):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            check_cancel(cancel)
            digest.update(block)
    check_cancel(cancel)
    return digest.hexdigest()


def _backup(source, destination, cancel, progress):
    started = time.perf_counter()
    def update(status, remaining, total):
        check_cancel(cancel)
        progress(total - remaining, total, time.perf_counter() - started)
    source.backup(destination, pages=256, progress=update)
    check_cancel(cancel)


def save_project(analysis: VideoAnalysis, destination: Path, effect: EffectType, size: int,
                 state: EditorState, draft: np.ndarray | None, cancel: Event, progress) -> Path:
    if not analysis.complete or not analysis.frames:
        raise ValueError("解析を完了してから編集を保存してください。")
    destination = destination.resolve()
    if destination.suffix.lower() != PROJECT_SUFFIX or destination == analysis.info.path:
        raise ValueError("編集の保存先には .fyvideo ファイルを指定してください。")
    state.validate(len(analysis.frames))
    analysis.info.verify_source()
    shape = (analysis.info.height, analysis.info.width)
    if draft is not None and (draft.shape != shape or draft.dtype != np.bool_):
        raise ValueError("編集中のマスクのサイズまたは形式が一致しません。")
    settings = asdict(replace(analysis.settings.processing, effect=effect, effect_size=size))
    settings["targets"] = sorted(settings["targets"])
    _read_settings(settings)
    metadata = {
        "format": "FY175AutoMosaicVideo", "version": 2,
        "source": {"path": str(analysis.info.path), "sha256": _fingerprint(analysis.info.path, cancel),
                   "width": analysis.info.width, "height": analysis.info.height, "fps": str(analysis.info.fps)},
        "settings": settings, "contour_interval": analysis.settings.contour_interval,
        "frames": [asdict(frame) for frame in analysis.frames],
        "elapsed": analysis.elapsed, "encoder_calls": analysis.encoder_calls,
        "editor": asdict(state),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, prefix=destination.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
        with closing(sqlite3.connect(analysis.cache.path)) as source, closing(sqlite3.connect(temporary)) as target:
            _backup(source, target, cancel, progress)
            with target:
                target.execute("DROP TABLE IF EXISTS project")
                target.execute("CREATE TABLE project (metadata TEXT NOT NULL, draft BLOB)")
                target.execute("INSERT INTO project VALUES (?, ?)", (json.dumps(metadata, ensure_ascii=False),
                    None if draft is None else _encode(".png", draft.astype(np.uint8) * 255)))
        analysis.info.verify_source()
        check_cancel(cancel)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_project(path: Path, cancel: Event, progress, *, source_path: Path | None = None) -> LoadedProject:
    path = path.resolve()
    cache = None
    try:
        check_cancel(cancel)
        with closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as source:
            source.execute("PRAGMA trusted_schema=OFF")
            row = source.execute("SELECT metadata, draft FROM project").fetchone()
            if row is None:
                raise ValueError("編集データがありません。")
            metadata = json.loads(row[0])
            if metadata["format"] != "FY175AutoMosaicVideo" or metadata["version"] != 2:
                raise ValueError("対応していない動画編集形式です。")
            processing = _read_settings(metadata["settings"])
            settings = VideoAnalysisSettings(processing, metadata["contour_interval"])
            frames = [FrameSummary(**frame) for frame in metadata["frames"]]
            if not frames or any(type(v) is not int or v < 0 for f in frames for v in (f.count, f.reused, f.fallbacks)):
                raise ValueError("解析フレームの情報が不正です。")
            state = EditorState(**metadata["editor"])
            state.validate(len(frames))
            if (type(metadata["elapsed"]) not in (int, float) or not math.isfinite(metadata["elapsed"])
                    or metadata["elapsed"] < 0 or type(metadata["encoder_calls"]) is not int or metadata["encoder_calls"] < 0):
                raise ValueError("保存された解析情報が不正です。")
            saved = metadata["source"]
            original = Path(saved["path"])
            candidates = [source_path] if source_path is not None else [path.parent / original.name, original]
            selected = None
            for candidate in dict.fromkeys(candidates):
                if candidate.is_file():
                    stat = candidate.stat()
                    if _fingerprint(candidate, cancel) == saved["sha256"]:
                        selected, source_stat = candidate, stat
                        break
            if selected is None:
                raise ProjectSourceUnavailable(f"保存時と同じ元動画を選択してください: {original.name}")
            info = probe_video(selected, cancel)
            if (info.source_size, info.source_mtime_ns) != (source_stat.st_size, source_stat.st_mtime_ns):
                raise ValueError("読み込み中に元動画が変更されました。")
            if (info.width, info.height, str(info.fps)) != (saved["width"], saved["height"], saved["fps"]):
                raise ValueError("元動画の寸法またはフレームレートが保存時と一致しません。")
            if source.execute("PRAGMA quick_check").fetchone() != ("ok",):
                raise ValueError("編集ファイルが破損しています。")
            if source.execute("SELECT COUNT(*), MIN(frame), MAX(frame) FROM frames").fetchone() != (len(frames), 0, len(frames) - 1):
                raise ValueError("解析フレームの保存データが不足しています。")
            cache = FrameCache()
            with closing(sqlite3.connect(cache.path)) as target:
                _backup(source, target, cancel, progress)
                with target:
                    target.execute("DROP TABLE project")
            edits, _ = cache.edit_state()
            if any(not 0 <= frame < len(frames) or kind not in {"manual", "propagated"} for frame, kind in edits.items()):
                raise ValueError("保存されたマスク編集情報が不正です。")
            draft = None if row[1] is None else FrameCache.decode_mask(row[1], (info.height, info.width))
            if state.draft_dirty and draft is None:
                raise ValueError("編集中のマスクがありません。")
            # Validate the frame that will be displayed before replacing the current UI state.
            cache.preview(state.frame, info)
            info.verify_source()
            check_cancel(cancel)
            analysis = VideoAnalysis(info, settings, cache, frames, metadata["elapsed"], metadata["encoder_calls"], True)
            result = LoadedProject(path, analysis, state, draft)
            cache = None
            return result
    except ProjectSourceUnavailable:
        raise
    except (sqlite3.Error, KeyError, TypeError, ValueError, OSError) as error:
        raise ValueError(f"{path.name}を復元できません: {error}") from error
    finally:
        if cache is not None:
            cache.close()
