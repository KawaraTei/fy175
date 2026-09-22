"""Local FFmpeg I/O. Analysis and export use the exact same CFR frame sequence."""
from __future__ import annotations

import json
import math
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryFile
from threading import Event, Thread

import numpy as np


SHARED_FFMPEG_DIR = Path("C:/Users/Admin/.codex/tools/ffmpeg/bin")
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"})


class VideoCancelled(Exception):
    pass


def check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise VideoCancelled("処理を中止しました。")


def tool_path(name: str) -> Path:
    """Packaged tools are authoritative; source runs use the shared installation."""
    if getattr(sys, "frozen", False):
        directory = Path(getattr(sys, "_MEIPASS")) / "ffmpeg"
    else:
        directory = SHARED_FFMPEG_DIR
    path = directory / f"{name}.exe"
    if not path.is_file():
        raise FileNotFoundError(f"{name}.exe がありません: {directory}")
    return path


@contextmanager
def media_process(arguments: list[str], cancel: Event, *, input_pipe=False, output_pipe=False):
    """Drain errors to disk and interrupt blocked pipe reads/writes on cancellation."""
    check_cancel(cancel)
    done = Event()
    with TemporaryFile() as errors:
        process = subprocess.Popen(
            arguments,
            stdin=subprocess.PIPE if input_pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE if output_pipe else subprocess.DEVNULL,
            stderr=errors,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def watch():
            while not done.wait(0.1):
                if cancel.is_set():
                    try:
                        process.kill()
                    except OSError:
                        pass
                    return

        watcher = Thread(target=watch, daemon=True)
        watcher.start()
        try:
            yield process
            if process.stdin is not None:
                process.stdin.close()
            code = process.wait()
            check_cancel(cancel)
            if code:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace")[-2500:]
                raise RuntimeError(f"動画の読み書きに失敗しました。\n{detail}")
        except (BrokenPipeError, OSError):
            check_cancel(cancel)
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace")[-2500:]
            raise RuntimeError(f"動画の読み書きに失敗しました。\n{detail}") from None
        finally:
            done.set()
            if process.poll() is None:
                process.kill()
            process.wait()
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            watcher.join(timeout=1)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    estimated_frames: int
    has_audio: bool
    audio_offset: float
    source_size: int
    source_mtime_ns: int

    def verify_source(self) -> None:
        stat = self.path.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.source_size, self.source_mtime_ns):
            raise ValueError("元動画が変更されています。動画を開き直して再解析してください。")


def probe_video(path: Path, cancel: Event) -> VideoInfo:
    path = path.resolve()
    stat = path.stat()
    with media_process([
        str(tool_path("ffprobe")), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(path),
    ], cancel, output_pipe=True) as process:
        raw = process.stdout.read()
        check_cancel(cancel)
        data = json.loads(raw)
    videos = [s for s in data["streams"] if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")]
    if not videos:
        raise ValueError("動画ストリームがありません。")
    stream = videos[0]
    # Avoid silently changing HDR colors or anamorphic geometry in an 8-bit path.
    if stream.get("color_transfer") in {"smpte2084", "arib-std-b67"}:
        raise ValueError("HDR動画は未対応です。SDRへ変換した動画を使用してください。")
    if stream.get("sample_aspect_ratio", "1:1") not in {"1:1", "0:1", "N/A"}:
        raise ValueError("非正方形ピクセルの動画は未対応です。正方形ピクセルへ変換してください。")
    fps = Fraction(0)
    for key in ("avg_frame_rate", "r_frame_rate"):
        try:
            fps = Fraction(stream.get(key, "0/1"))
        except (ValueError, ZeroDivisionError):
            continue
        if fps > 0:
            break
    if not 0 < float(fps) <= 240:
        raise ValueError("フレームレートを取得できないか、240 fpsを超えています。")
    width, height = int(stream["width"]), int(stream["height"])
    rotation = next((float(s["rotation"]) for s in stream.get("side_data_list", [])
                     if "rotation" in s), 0.0)
    if abs(rotation / 90 - round(rotation / 90)) > 0.01:
        raise ValueError("90度単位以外に回転した動画は未対応です。")
    if round(rotation) % 180:
        width, height = height, width
    duration = float(stream.get("duration") or data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("長さを取得できない動画は処理できません。")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    offset = (float(audio.get("start_time", 0)) - float(stream.get("start_time", 0))) if audio else 0
    return VideoInfo(path, width, height, fps, duration, max(1, round(duration * fps)),
                     audio is not None, offset, stat.st_size, stat.st_mtime_ns)


def decode_frames(info: VideoInfo, cancel: Event, *, first_only=False):
    # CFR normalization is explicit in the UI. Repeating this filter at export
    # guarantees that mask index N always refers to the same source image.
    args = [str(tool_path("ffmpeg")), "-hide_banner", "-loglevel", "error", "-xerror",
            "-nostdin", "-i", str(info.path), "-map", "0:V:0", "-an", "-sn",
            "-vf", f"setpts=PTS-STARTPTS,fps={info.fps}:start_time=0,scale={info.width}:{info.height},setsar=1"]
    if first_only:
        args += ["-frames:v", "1"]
    args += ["-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]
    frame_size = info.width * info.height * 3
    with media_process(args, cancel, output_pipe=True) as process:
        while True:
            check_cancel(cancel)
            data = bytearray()
            while len(data) < frame_size:
                chunk = process.stdout.read(frame_size - len(data))
                if not chunk:
                    break
                data.extend(chunk)
            if not data:
                break
            if len(data) != frame_size:
                check_cancel(cancel)
                raise RuntimeError("動画のフレームが途中で途切れました。")
            yield np.frombuffer(data, dtype=np.uint8).reshape(info.height, info.width, 3)


def encoder_arguments(info: VideoInfo, path: Path) -> list[str]:
    return [str(tool_path("ffmpeg")), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
            "-r", str(info.fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-preset", "fast",
            "-crf", "18", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,setsar=1",
            "-pix_fmt", "yuv420p", str(path)]


def mux_audio(info: VideoInfo, silent: Path, destination: Path, duration: float, cancel: Event):
    args = [str(tool_path("ffmpeg")), "-hide_banner", "-loglevel", "error", "-nostdin",
            "-y", "-i", str(silent)]
    if info.has_audio:
        args += ["-i", str(info.path), "-map", "0:v:0", "-map", "1:a:0", "-c:a", "aac", "-b:a", "192k"]
        audio_filter = "asetpts=PTS-STARTPTS"
        if info.audio_offset > 0:
            audio_filter += f",adelay={round(info.audio_offset * 1000)}:all=1"
        elif info.audio_offset < 0:
            audio_filter += f",atrim=start={-info.audio_offset:.6f},asetpts=PTS-STARTPTS"
        args += ["-af", audio_filter]
    else:
        args += ["-map", "0:v:0"]
    args += ["-c:v", "copy", "-map_metadata", "-1", "-t", f"{duration:.9f}",
             "-movflags", "+faststart", str(destination)]
    with media_process(args, cancel):
        pass
