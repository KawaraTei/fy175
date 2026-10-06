"""Self-describing loopback HTTP API over the application's actual Qt windows.

No web framework or parallel processing session: commands run on the Qt thread;
workers share the UI pipeline, with one read-only indexed-preview cache.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
from urllib.parse import parse_qs, urlsplit
import uuid

import cv2
import numpy as np
from PySide6.QtCore import QBuffer, QIODevice, QObject, QTimer, Signal

from auto_mosaic.image_ops import apply_effect, load_image_bgr, paint_mask_stroke, visualize_detection
from auto_mosaic.ui import MAX_IMAGES
from auto_mosaic.detector import BELOW_THRESHOLD_PREVIEW_MIN_CONFIDENCE
from auto_mosaic.domain import PREVIEW_VIEWS


VIEWS = PREVIEW_VIEWS


class ShutdownRelay(QObject):
    requested = Signal()


class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def require(condition, message, status=400):
    if not condition:
        raise ApiError(message, status)


def scalar(value, kind, name):
    require(type(value) is kind, f"{name} must be {kind.__name__}")
    if kind is float:
        require(math.isfinite(value), f"{name} must be finite")
    return value


class ObservedQueue:
    """Observe events when the UI consumes them, preserving its worker queue."""
    def __init__(self, original, callback):
        self.original, self.callback = original, callback

    def put(self, item):
        self.original.put(item)

    def get_nowait(self):
        item = self.original.get_nowait()
        self.callback(*item)
        return item


def operation(description, **arguments):
    return {"description": description, "arguments": arguments}


OPERATIONS = {
    "images.add": operation("Add PNG/JPEG paths; first selection starts analysis. Poll returned job.", paths="required array of absolute paths"),
    "images.select": operation("Select zero-based index; discards draft only with discard=true; starts analysis.", index="required integer", discard="optional boolean"),
    "images.remove": operation("Remove indices; optionally record filenames for manual review.", indices="required array of zero-based indices", manual_review="optional boolean", discard="optional boolean"),
    "images.clear": operation("Clear image list.", discard="optional boolean"),
    "images.manual_clear": operation("Clear manual review filenames."),
    "settings.update": operation("Update UI-owned settings; GET / describes current ranges. Detection changes require analyze; effect changes preserve masks.", workspace="image only (optional)", values="required object with keys from settings_schema.image", discard="optional boolean: discard existing masks if required"),
    "image.analyze": operation("Reanalyze selected image; replaces existing automatic mask.", discard="optional boolean"),
    "mask.edit": operation("Add/erase original-pixel strokes, rectangles, or polygons; edits remain a draft. GET /preview and /mask show it. image.save uses draft.", workspace="image only (optional)", action="add or erase", shape="stroke, rectangle, polygon, or replace", points="array of [x,y]; rectangle uses two inclusive corners", diameter="optional stroke diameter in source pixels; uses current UI brush if omitted; see mask_edit_schema", path="replace: absolute PNG mask path; nonzero pixels are selected"),
    "mask.discard": operation("Discard image mask draft.", workspace="image only (optional)"),
    "mask.undo": operation("Undo one mask operation on current image; retains up to 20 operations, bounded to 128 MiB of packed masks."),
    "mask.remove_detection": operation("Remove only pixels owned by this accepted detection; preserve pixels owned by overlapping detections. Creates an undoable draft.", index="required detection index from /state; stable until next analysis"),
    "mask.dilate": operation("Expand current mask, or one detection's owned mask; expansion is in original pixels and preserves other detections.", px="required integer 1..100", index="optional detection index; omitted means combined mask"),
    "mask.erode": operation("Shrink current mask, or one detection's owned mask; preserve overlapping other detections.", px="required integer 1..100", index="optional detection index; omitted means combined mask"),
    "image.save": operation("Save selected image using current draft if present; otherwise rerun analysis, exactly as UI. Set output_dir/suffix through settings.update, or pass an exact path.", path="optional absolute PNG/JPEG output filename; existing file rejected"),
    "images.save_all": operation("Analyze and save all images; requires no active draft."),
    "app.shutdown": operation("Close app after HTTP response is sent. Busy jobs must finish first. Unsaved mask requires discard=true.", discard="optional boolean"),
}


class AgentController:
    def __init__(self, window):
        self.window = window
        self.requests = queue.Queue()
        self.jobs = {}
        self.active = {}
        self.notifications = deque(maxlen=50)
        self.undo_history = deque()
        self.edit_detection_masks = None
        self.last_error = None
        self.preview_events = queue.Queue()
        self.preview_cache = None
        self.preview_key = None
        self.shutdown = ShutdownRelay(window)
        self.shutdown.requested.connect(window.close)
        self.window.agent_notice = lambda kind, title, text: self.notifications.append(
            {"kind": kind, "title": title, "text": text})
        self._observe(window, "image")
        self.timer = QTimer(window)
        self.timer.timeout.connect(self._drain)
        self.timer.start(10)

    def _observe(self, window, workspace):
        window.events = ObservedQueue(window.events, lambda event, payload: self._event(workspace, event, payload))

    def _event(self, workspace, event, payload):
        job = self.active.get(workspace)
        if job is None:
            return
        if event == "progress":
            job["progress"] = list(payload)
            return
        if workspace == "image":
            if event in {"analysis", "analysis_error"} and payload[0] != self.window.analysis_generation:
                return
            if event == "analysis":
                self._reset_history()
                result = self._result_details(payload[1])
            elif event == "single_complete":
                if payload[3] and self.edit_detection_masks is not None:
                    payload[0].detection_masks = self.edit_detection_masks
                result = {"outputs": [str(payload[1])], **self._result_details(payload[0]),
                          "mask_source": "edited" if payload[3] else "reanalyzed",
                          "used_edited_mask": bool(payload[3]), "detection_rerun": not payload[3]}
                self._reset_history()
            elif event == "complete":
                result = {"outputs": [str(path) for path in payload[0]], "mask_source": "reanalyzed",
                          "used_edited_mask": False, "detection_rerun": True,
                          "saved_images": [{"path": str(source), "output": str(output), "mask_source": "reanalyzed",
                                            "used_edited_mask": False, "detection_rerun": True}
                                           for source, output in zip(payload[1], payload[0])]}
            elif event in {"analysis_error", "error", "batch_error"}:
                error = payload[1] if event == "analysis_error" else payload[0] if event == "batch_error" else payload
                job.update(status="failed", error=str(error))
                self.last_error = str(error)
                if event == "batch_error":
                    job["completed_paths"] = [str(path) for path in payload[1]]
                result = None
            else:
                return
        if result is not None:
            job.update(status="succeeded", result=result)
        job["finished_at"] = time.time()
        self.active.pop(workspace, None)

    def _track(self, workspace, name):
        job = {"id": uuid.uuid4().hex, "operation": name, "workspace": workspace,
               "status": "running", "started_at": time.time()}
        self.jobs[job["id"]] = job
        self.active[workspace] = job
        self.last_error = None
        # Retain recent terminal jobs without losing running ones.
        if len(self.jobs) > 200:
            for key, value in list(self.jobs.items()):
                if value["status"] != "running":
                    del self.jobs[key]
                    break
        return job

    def submit(self, method, path, data):
        response = queue.Queue(maxsize=1)
        self.requests.put((method, path, data, response))
        # No timeout: a command must never execute after a timeout reported failure.
        return response.get()

    def _drain(self):
        try:
            key, result, error = self.preview_events.get_nowait()
        except queue.Empty:
            pass
        else:
            job = self.active.pop("preview")
            self.preview_key = None
            if error is None:
                self.preview_cache = (key, result)
                job.update(status="succeeded", result=self._result_details(result))
            else:
                job.update(status="failed", error=str(error))
                self.last_error = str(error)
            job["finished_at"] = time.time()
        for _ in range(8):
            try:
                method, path, data, response = self.requests.get_nowait()
            except queue.Empty:
                break
            try:
                response.put(self.handle(method, path, data))
            except ApiError as error:
                response.put((error.status, "application/json", {"error": str(error)}, {}))
            except Exception as error:
                response.put((500, "application/json", {"error": str(error), "type": type(error).__name__}, {}))

    def _idle(self):
        require(not self.active and not self.window.busy,
                "An operation is running; poll /jobs/{id} or /state first", 409)

    def _draft(self, workspace, discard=False):
        window = self.window
        require(not window.mask_edit_active or discard, "Draft mask exists; save it or use discard=true", 409)
        if window.mask_edit_active:
            window._end_mask_edit(False)
        self._reset_history()

    def _reset_history(self):
        self.undo_history.clear()
        self.edit_detection_masks = None

    @staticmethod
    def _result_details(result):
        return {"path": str(result.source_path), "detection_count": len(result.detections),
                "detections": [{"index": i, **asdict(d), "mask_available": i < len(result.detection_masks),
                                "mask_pixels": int(result.detection_masks[i].sum()) if i < len(result.detection_masks) else None}
                               for i, d in enumerate(result.detections)],
                "below_threshold_detections": [asdict(d) for d in result.below_threshold_detections],
                "mask_pixels": int(result.mask.sum()), "box_fallbacks": result.used_box_fallbacks}

    def _owned_masks(self):
        return (self.edit_detection_masks if self.edit_detection_masks is not None
                else self.window.current_result.detection_masks)

    def _install_mask(self, candidate, owned=None):
        w = self.window
        old = self._mask("image")
        previous_owned = self._owned_masks()
        if np.array_equal(old, candidate) and (owned is None or all(np.array_equal(a, b) for a, b in zip(previous_owned, owned))):
            return
        packed = [np.packbits(item).tobytes() for item in [old, *previous_owned]]
        require(sum(len(item) for item in packed) <= 128 * 1024 * 1024, "Mask snapshot exceeds undo memory limit", 413)
        self.undo_history.append((old.shape, packed))
        while len(self.undo_history) > 1 and (len(self.undo_history) > 20 or sum(sum(len(b) for b in entry[1]) for entry in self.undo_history) > 128 * 1024 * 1024):
            self.undo_history.popleft()
        if not w.mask_edit_active:
            w.set_preview_mode("マスク範囲")
            w._toggle_mask_edit()
        w.edited_mask = candidate
        self.edit_detection_masks = [item & candidate for item in (previous_owned if owned is None else owned)]
        w.mask_edit_dirty = True
        self.last_error = None
        w._refresh_preview()

    def undo(self):
        require(self.window.mask_edit_active and self.undo_history, "No mask operation to undo", 409)
        shape, packed = self.undo_history.pop()
        arrays = [np.unpackbits(np.frombuffer(item, np.uint8), count=shape[0] * shape[1]).reshape(shape).astype(bool) for item in packed]
        self.window.edited_mask, self.edit_detection_masks = arrays[0], arrays[1:]
        self.window.mask_edit_dirty = not np.array_equal(arrays[0], self.window.current_result.mask)
        self.window._refresh_preview()

    def transform_mask(self, name, args):
        combined = self._mask("image")
        owned = self._owned_masks()
        index = args.get("index")
        if index is not None or name == "mask.remove_detection":
            require(type(index) is int and 0 <= index < len(owned), "Detection index has no owned mask; inspect /state", 400)
            target = owned[index]
        else:
            target = combined
        if name == "mask.remove_detection":
            changed = np.zeros_like(target)
        else:
            px = scalar(args.get("px"), int, "px")
            require(1 <= px <= 100, "px must be 1..100")
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px * 2 + 1, px * 2 + 1))
            transform = cv2.dilate if name == "mask.dilate" else cv2.erode
            changed = transform(target.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
        if index is None:
            self._install_mask(changed)
        else:
            other = np.zeros_like(combined)
            for i, mask in enumerate(owned):
                if i != index:
                    other |= mask
            candidate = (combined & ~(target & ~other)) | changed
            updated = list(owned)
            updated[index] = changed
            self._install_mask(candidate, updated)

    def _schema(self, workspace):
        w = self.window
        fields = self._fields(workspace)
        schema = {}
        for name, (widget, kind, choices) in fields.items():
            entry = {"type": kind}
            if choices:
                if kind == "array":
                    entry.update(items={"type": "string", "enum": list(choices)}, minItems=1, uniqueItems=True)
                else:
                    entry["enum"] = list(choices)
            elif kind in {"integer", "number"}:
                entry.update(minimum=widget.minimum(), maximum=widget.maximum())
                if name == "confidence_threshold":
                    entry.update(minimum=widget.minimum() / 100, maximum=widget.maximum() / 100, multipleOf=0.01)
                elif kind == "number":
                    entry["multipleOf"] = 10 ** -widget.decimals()
            schema[name] = entry
        return schema

    def _fields(self, workspace):
        w = self.window
        fields = {
            "mode": (w.mode_combo, "string", {"photo": "実写", "illustration": "イラスト"}),
            "targets": ((w.penis_check, w.vagina_check), "array", {"penis": True, "vagina": True}),
            "confidence_threshold": (w.threshold_slider, "number", None),
            "effect": (w.effect_combo, "string", {"mosaic": "モザイク", "blur": "ぼかし"}),
            "effect_size": (w.effect_size_slider, "integer", None),
        }
        if workspace == "image":
            fields.update(mask_threshold=(w.mask_threshold_spin, "number", None),
                          mask_expansion=(w.mask_expansion_spin, "integer", None),
                          output_dir=(w.output_edit, "string", None), suffix=(w.suffix_edit, "string", None),
                          remove_after_process=(w.remove_after_process_check, "boolean", None))
        return fields

    def _settings(self, workspace):
        w = self.window
        values = {}
        for name, (widget, kind, choices) in self._fields(workspace).items():
            if name == "targets":
                value = [target for target, control in zip(("penis", "vagina"), widget) if control.isChecked()]
            elif choices:
                value = next(key for key, label in choices.items() if label == widget.currentText())
            elif name == "confidence_threshold":
                value = widget.value() / 100
            elif kind in {"number", "integer"}:
                value = widget.value()
            elif kind == "boolean":
                value = widget.isChecked()
            else:
                value = widget.text()
            values[name] = value
        return values

    def configure(self, workspace, values, discard):
        require(isinstance(values, dict) and values, "values must be a nonempty object")
        w = self.window
        fields, schema = self._fields(workspace), self._schema(workspace)
        require(not set(values) - set(fields), f"Unknown settings: {sorted(set(values) - set(fields))}")
        # Validate entire request before mutating widgets; Qt would otherwise silently clamp.
        for name, value in values.items():
            widget, kind, choices = fields[name]
            if name == "targets":
                require(isinstance(value, list) and value and all(type(v) is str and v in choices for v in value), "targets must contain penis and/or vagina")
            elif choices:
                require(type(value) is (int if kind == "integer" else str) and value in choices, f"Invalid {name}: {value}")
            elif kind in {"number", "integer"}:
                require(type(value) in ((int, float) if kind == "number" else (int,)) and math.isfinite(value), f"Invalid number: {name}")
                require(schema[name]["minimum"] <= value <= schema[name]["maximum"], f"{name} outside UI range")
                step = schema[name].get("multipleOf", 1)
                require(abs(value / step - round(value / step)) < 1e-6, f"{name} must be a multiple of {step}")
            elif kind == "boolean":
                scalar(value, bool, name)
            else:
                scalar(value, str, name)
                if name == "output_dir":
                    require(Path(value).is_absolute(), "output_dir must be absolute")
                if name == "suffix":
                    require(not any(c in value for c in '<>:"/\\|?*') and not value.endswith((" ", ".")), "Invalid filename suffix")
        detection_keys = set(fields) - {"effect", "effect_size", "output_dir", "suffix", "remove_after_process"}
        changing_detection = any(name in detection_keys and value != self._settings(workspace)[name] for name, value in values.items())
        if changing_detection:
            self._draft(workspace, discard)
        for name, value in values.items():
            widget, kind, choices = fields[name]
            if name == "targets":
                for target, control in zip(("penis", "vagina"), widget):
                    control.setChecked(target in value)
            elif choices:
                widget.setCurrentText(choices[value])
            elif name == "confidence_threshold":
                widget.setValue(round(value * 100))
            elif kind in {"number", "integer"}:
                widget.setValue(value)
            elif kind == "boolean":
                widget.setChecked(value)
            else:
                widget.setText(value)
        if workspace == "image" and w.current_result is not None and not w.mask_edit_active:
            reference = w.current_result
            w.current_result = w.pipeline.process_with_mask(reference.source_path, reference.mask, w._settings(), reference.detections,
                                                          reference.used_box_fallbacks, reference.below_threshold_detections, reference.detection_masks)
            w._refresh_preview()

    def state(self):
        w = self.window
        status = ("processing" if w.busy or self.active else "error" if self.last_error else "editing" if w.mask_edit_active
                  else "empty" if not w.image_paths else "awaiting_selection" if w._selected_path() is None else "ready")
        result = {"image": {"busy": bool(w.busy or self.active), "status": status, "status_text": w.status_label.text(), "progress": w.progress.value(),
                  "paths": [str(p) for p in w.image_paths], "selected_index": w.file_list.currentRow(),
                  "manual_review": [w.manual_review_list.item(i).text() for i in range(w.manual_review_list.count())],
                  "manual_review_paths": [str(path) for path in w.manual_review_paths],
                  "settings": self._settings("image"), "editing": w.mask_edit_active,
                  "mask_dirty": w.mask_edit_dirty,
                  "preview_view": next(key for key, label in VIEWS.items() if label == w.preview_mode()),
                  "preview_view_text": w.preview_mode(),
                  "undo_depth": len(self.undo_history) if w.mask_edit_active else 0, "last_error": self.last_error},
                  "notifications": list(self.notifications), "active_jobs": [dict(job) for job in self.active.values()]}
        if w.current_result is not None:
            mask = w.edited_mask if w.mask_edit_active else w.current_result.mask
            result["image"].update(width=mask.shape[1], height=mask.shape[0], mask_pixels=int(mask.sum()),
                                   detections=[{"index": i, **asdict(d), "mask_available": i < len(self._owned_masks()),
                                                "mask_pixels": int(self._owned_masks()[i].sum()) if i < len(self._owned_masks()) else None}
                                               for i, d in enumerate(w.current_result.detections)],
                                   below_threshold_detections=[asdict(d) for d in w.current_result.below_threshold_detections],
                                   box_fallbacks=w.current_result.used_box_fallbacks)
        return result

    def _mask(self, workspace):
        if workspace == "image":
            w = self.window
            require(w.current_result is not None, "Analyze an image first", 409)
            return w.edited_mask if w.mask_edit_active else w.current_result.mask

    def edit(self, workspace, args):
        mask = self._mask(workspace)
        height, width = mask.shape
        shape = args.get("shape", "stroke")
        require(shape in {"stroke", "rectangle", "polygon", "replace"}, "Unknown mask shape")
        action = args.get("action", "add")
        require(action in {"add", "erase"}, "action must be add or erase")
        candidate = mask.copy()
        if shape == "replace":
            path = self._path(args.get("path"))
            imported = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
            require(imported is not None and imported.shape == mask.shape, "Replacement mask must match original dimensions")
            candidate = imported > 0
        else:
            points = args.get("points")
            require(isinstance(points, list) and len(points) >= (3 if shape == "polygon" else 2 if shape == "rectangle" else 1), "Not enough points")
            require(len(points) <= 10000, "Too many points")
            for point in points:
                require(isinstance(point, list) and len(point) == 2 and all(type(p) is int for p in point), "Points must be integer [x,y]")
                require(0 <= point[0] < width and 0 <= point[1] < height, "Point outside original image; inspect preview transform")
            if shape == "stroke":
                brush = self.window.brush_size_slider
                diameter = args.get("diameter", brush.value())
                require(type(diameter) is int and brush.minimum() <= diameter <= brush.maximum(), "Brush diameter outside UI range; GET / for mask_edit_schema")
                for start, end in zip(points, points[1:] or points):
                    paint_mask_stroke(candidate, tuple(start), tuple(end), diameter, action == "erase")
            else:
                require(shape != "rectangle" or len(points) == 2, "rectangle needs exactly two corners")
                region = np.zeros(mask.shape, np.uint8)
                if shape == "rectangle":
                    cv2.rectangle(region, tuple(points[0]), tuple(points[1]), 1, -1)
                else:
                    cv2.fillPoly(region, [np.asarray(points, np.int32)], 1)
                candidate[region > 0] = action == "add"
        self._install_mask(candidate)
        if shape == "stroke":
            self.window.brush_size_slider.setValue(diameter)

    @staticmethod
    def _path(value, exists=True):
        require(type(value) is str and Path(value).is_absolute(), "path must be an absolute local path")
        path = Path(value).resolve()
        require(not exists or path.is_file(), f"File does not exist: {path}", 404)
        return path

    def command(self, name, args):
        require(name in OPERATIONS, f"Unknown operation: {name}", 404)
        require(isinstance(args, dict), "arguments must be an object")
        require(not set(args) - set(OPERATIONS[name]["arguments"]), f"Unknown arguments: {sorted(set(args) - set(OPERATIONS[name]['arguments']))}")
        workspace = args.get("workspace", "image")
        require(workspace == "image", "Only image workspace is supported; video mode is not supported")
        discard = args.get("discard", False)
        scalar(discard, bool, "discard")
        if "manual_review" in args:
            scalar(args["manual_review"], bool, "manual_review")
        self._idle()
        w, job = self.window, None
        if name == "settings.update":
            self.configure(workspace, args.get("values"), discard)
        elif name == "mask.edit":
            self.edit(workspace, args)
        elif name == "mask.discard":
            w._end_mask_edit(True)
            self._reset_history()
        elif name == "mask.undo":
            self.undo()
        elif name in {"mask.remove_detection", "mask.dilate", "mask.erode"}:
            self.transform_mask(name, args)
        elif name == "app.shutdown":
            self._draft("image", discard)
        elif name == "images.add":
            paths = args.get("paths")
            require(isinstance(paths, list) and paths, "paths must be a nonempty array")
            paths = [self._path(path) for path in paths]
            require(all(path.suffix.lower() in {".png", ".jpg", ".jpeg"} for path in paths), "Only PNG/JPEG are supported")
            require(len(set(paths) | set(w.image_paths)) <= MAX_IMAGES, f"Image list capacity is {MAX_IMAGES}")
            # Validate decode before selection starts UI auto-analysis.
            for path in paths:
                load_image_bgr(path)
            first = not w.image_paths
            if first:
                w._settings()
                job = self._track("image", name)
            w._add_image_paths(paths)
        elif name in {"images.select", "images.remove", "images.clear", "images.manual_clear"}:
            if name == "images.manual_clear":
                w._clear_manual_review()
            elif name == "images.clear":
                self._draft("image", discard)
                w._clear_images()
            else:
                indices = [args.get("index")] if name == "images.select" else args.get("indices")
                require(isinstance(indices, list) and indices and all(type(i) is int and 0 <= i < len(w.image_paths) for i in indices), "Invalid image indices")
                if name != "images.select" or indices[0] != w.file_list.currentRow():
                    self._draft("image", discard)
                w._settings()
                if name == "images.select":
                    if indices[0] != w.file_list.currentRow():
                        job = self._track("image", name)
                        w.file_list.setCurrentRow(indices[0])
                else:
                    manual = args.get("manual_review", False)
                    scalar(manual, bool, "manual_review")
                    w._remove_rows(indices, manual)
                    if w._selected_path() is not None:
                        job = self._track("image", name)
        elif name == "image.analyze":
            require(w._selected_path() is not None, "Select an image first", 409)
            w._settings()
            self._draft("image", discard)
            job = self._track("image", name)
            w._analyze_current()
        elif name in {"image.save", "images.save_all"}:
            require(w._selected_path() is not None, "Select an image first", 409)
            w._settings()
            w._filename_suffix()
            require(Path(w.output_edit.text()).is_absolute(), "Set an absolute output_dir first")
            if name == "images.save_all":
                self._draft("image")
            output_path = None
            if "path" in args:
                output_path = self._path(args["path"], exists=False)
                require(output_path.suffix.lower() in {".png", ".jpg", ".jpeg"}, "Output filename must be PNG/JPEG")
                require(not output_path.exists(), "Output file already exists", 409)
            job = self._track("image", name)
            if name == "image.save":
                w._save_current(output_path)
            else:
                w._process_all()
        self.last_error = None
        return {"job": dict(job) if job else None, "state": self.state()}

    def describe(self):
        return {"name": "FY175AutoMosaic Agent API", "version": 2,
                "pid": os.getpid(), "parent_pid": os.getppid(),
                "supported_media": ["image"],
                "image_list_capacity": MAX_IMAGES,
                "transport": "HTTP/1.1 on 127.0.0.1, persistent connections and Content-Length; chunked request bodies unsupported. No external dependencies or MCP registration required",
                "workflow": ["GET / for capabilities and current setting constraints", "POST /commands with operation and arguments",
                             "Poll GET /jobs/{id} until succeeded or failed (recommended 0.2s interval)",
                             "GET /state to inspect results", "GET /preview?workspace=image&view=detection to see coverage",
                             "Edit mask in ORIGINAL pixel coordinates; previews include source-to-preview transform in X-Agent-Metadata",
                             "GET /preview?view=result to check effect; adjust effect or mask; save"],
                "semantics": {"indices": "zero-based; rectangles include both corners",
                              "drafts": "ONE selected-image draft only. Save with image.save BEFORE selecting another image; switching with discard=true loses it. images.save_all always reruns analysis and never uses manual edits. Loop select/edit/image.save to save multiple edited images. No draft persistence across switching, reanalysis or app exit.",
                              "undo": "mask.undo undoes mask.edit, remove_detection, dilate, erode; up to 20 operations and 128 MiB of packed masks. Reset by discard, reanalysis, image switching or successful save. Effect/setting changes are not part of mask undo.",
                              "mask_expansion": "Automatic segmentation only; not reapplied to manual masks. Use mask.dilate/erode for explicit draft morphology. px uses an elliptical kernel of radius px; does not rerun detection or contour correction.",
                              "detection_masks": "Detection index is stable ONLY for current analysis. GET /mask?index=N retrieves that detection's owned full-size mask. remove_detection/targeted morphology preserve overlapping other detections. Freehand additions outside owned masks remain unattributed and are not removed by remove_detection. Whole-mask replacement or morphology also leaves new pixels unattributed. Original model boxes/confidence stay unchanged by edits.",
                              "state_codes": {"status": ["empty", "awaiting_selection", "processing", "editing", "ready", "error"], "status_text": "UI display text (Japanese)", "preview_view": list(VIEWS), "preview_view_text": "UI display text (Japanese)"},
                              "detection_settings": "Changing detection settings requires reanalysis; discard=true explicitly allows replacing edits.",
                              "effects": "Effect changes preserve masks and immediately update previews.",
                              "concurrency": "One operation at a time; all mutations happen on UI thread. UI state is shared.",
                              "previews": "/mask returns full original-size mask. Use preview metadata to map coordinates.",
                              "preview_metadata": {"source_width/source_height": "Original image dimensions after EXIF orientation",
                                                   "width/height": "Returned PNG dimensions",
                                                   "source_crop": "[x,y,width,height] in original pixels; actual bounds after rounding",
                                                   "source_units_per_pixel": "[sx,sy]; original point = [crop.x + preview.x*sx, crop.y + preview.y*sy]",
                                                   "image_index": "Zero-based image index", "view": "Requested view code",
                                                   "mask_source": "edited, automatic, or none (unanalysed unselected original view)",
                                                   "mask_pixels": "Full-size combined mask pixel count; null for an unanalysed, unselected original view"},
                              "detection_view": f"Colored masks/boxes are accepted detections. Gray boxes are preview-only candidates with confidence >= {BELOW_THRESHOLD_PREVIEW_MIN_CONFIDENCE} and below the chosen detection threshold; at most five, no masks. A lower user detection threshold still admits accepted detections below this preview floor. mask_overlay shows original pixels with colored coverage/contours and no boxes or labels. Manual editing hides box annotations. GET /state distinguishes accepted from below-threshold candidates.",
                              "files": "Absolute local paths. Images avoid overwrites by adding a sequence suffix.",
                              "jobs": "Terminal errors and partial batch completion are returned as data; no modal completion dialogs.",
                              "save_provenance": "Save job result.mask_source is edited or reanalyzed; used_edited_mask and detection_rerun are booleans. Batch results also include saved_images with per-image provenance.",
                              "indexed_previews": "image_index selects a zero-based image without changing selection, draft or undo. Selected image uses current draft. Other images use current settings and a one-image analysis cache invalidated by file/settings changes. First processed preview returns HTTP 202 with job and retry_url; poll job then retry URL. Original view needs no analysis. Unselected previews do not change the UI display.",
                              "shutdown": "POST /commands app.shutdown closes the listening process after its response. pid is the actual listening app PID; parent_pid may be a Windows launcher. Stop the app through the API rather than relying on the launcher PID.",
                              "security": "Loopback only; browser Origin requests rejected; no CORS. Local agents have same file access as app."},
                "endpoints": {"GET /": "This self-contained reference", "GET /state": "Shared UI state, detections, settings, progress, notifications",
                              "POST /commands": {"body": {"operation": "name below", "arguments": {}}, "returns": "job (or null), state"},
                              "GET /jobs/{id}": "Job status/result/error; last 200 retained",
                              "GET /preview": {"returns": "PNG + X-Agent-Metadata JSON header; HTTP 202 JSON job if indexed analysis is needed", "query": {"workspace": "image only (optional)", "image_index": "optional zero-based image index; default selected image", "view": "original|detection|mask_overlay|result (default result)", "max_size": "optional max edge 64..4096", "crop": "optional x,y,width,height in ORIGINAL pixels"}},
                              "GET /preview/metadata": "Same query and HTTP 202 behavior as /preview; returns the identical coordinate metadata as a JSON body, without PNG encoding",
                              "GET /mask": "PNG binary mask, original dimensions; optional index=N selects a detection's owned mask",
                              "GET /snapshot": "Actual Qt window PNG (available during processing)"},
                "settings_schema": {"image": self._schema("image")},
                "mask_edit_schema": {workspace: {"diameter": {"type": "integer", "minimum": brush.minimum(), "maximum": brush.maximum(), "current": brush.value()}}
                                     for workspace, brush in (("image", self.window.brush_size_slider),)},
                "operations": OPERATIONS,
                "example": {"operation": "mask.edit", "arguments": {"workspace": "image", "shape": "stroke", "action": "add", "points": [[20, 20], [40, 30]], "diameter": 16}}}

    def _indexed_result(self, source, settings):
        stat = source.stat()
        key = (source, stat.st_mtime_ns, stat.st_size, settings)
        if self.preview_cache is not None and self.preview_cache[0] == key:
            return self.preview_cache[1], None
        job = self.active.get("preview")
        if job is not None:
            require(self.preview_key == key, "Another preview is running; poll its job first", 409)
            return None, job
        self._idle()
        job = self._track("preview", "image.preview")
        self.preview_key = key
        def work():
            try:
                result = self.window.pipeline.analyze(source, settings)
                self.preview_events.put((key, result, None))
            except Exception as error:
                self.preview_events.put((key, None, error))
        threading.Thread(target=work, name="agent-preview", daemon=True).start()
        return None, job

    def image_response(self, path, params, raw_path):
        workspace = params.get("workspace", ["image"])[0]
        require(workspace == "image", "Only image workspace is supported; video mode is not supported")
        w = self.window
        if path == "/snapshot":
            buffer = QBuffer()
            buffer.open(QIODevice.OpenModeFlag.WriteOnly)
            require(w.grab().save(buffer, "PNG"), "Could not capture Qt window", 500)
            return 200, "image/png", bytes(buffer.data()), {}
        view = params.get("view", ["result"])[0]
        require(view in VIEWS, "Invalid preview view")
        try:
            image_index = int(params["image_index"][0]) if "image_index" in params else w.file_list.currentRow()
        except ValueError:
            raise ApiError("image_index must be an integer") from None
        require(0 <= image_index < len(w.image_paths), "Image index out of range")
        source = w.image_paths[image_index]
        if path != "/mask" and source != w._selected_path():
            settings = w._settings()
            if view == "original":
                self._idle()
                image = load_image_bgr(source)
                mask = np.zeros(image.shape[:2], bool)
            else:
                result, job = self._indexed_result(source, settings)
                if job is not None:
                    return 202, "application/json", {"job": dict(job), "retry_url": raw_path}, {}
                self._idle()
                mask = result.mask
                original = load_image_bgr(source)
                image = (apply_effect(original, mask, settings.effect, settings.effect_size) if view == "result" else
                         visualize_detection(original, mask, result.detections, result.below_threshold_detections,
                                             show_detection_annotations=view == "detection"))
        else:
            self._idle()
            mask = self._mask(workspace)
            image = None
        require(path != "/mask" or "image_index" not in params, "image_index is supported by /preview and /preview/metadata only")
        if path == "/mask" and "index" in params:
            try:
                index = int(params["index"][0])
            except ValueError:
                raise ApiError("index must be an integer") from None
            owned = self._owned_masks()
            require(0 <= index < len(owned), "Detection mask index out of range")
            mask = owned[index]
        height, width = mask.shape
        if path == "/mask":
            image = mask.astype(np.uint8) * 255
        elif image is None:
            if workspace == "image":
                w.set_preview_mode(VIEWS[view])
                if view == "result":
                    original = w.mask_edit_original_bgr if w.mask_edit_active else load_image_bgr(w._selected_path())
                    image = apply_effect(original, mask, w._settings().effect, w._settings().effect_size)
                    w._set_preview_bgr(image)
                else:
                    w._refresh_preview()
                    image = cv2.cvtColor(w.preview_rgb, cv2.COLOR_RGB2BGR)
        ih, iw = image.shape[:2]
        x, y, cw, ch = 0, 0, width, height
        if "crop" in params:
            try:
                x, y, cw, ch = map(int, params["crop"][0].split(","))
            except ValueError:
                raise ApiError("crop must be x,y,width,height") from None
            require(x >= 0 and y >= 0 and cw > 0 and ch > 0 and x + cw <= width and y + ch <= height, "Crop outside original image")
            left, top = math.floor(x * iw / width), math.floor(y * ih / height)
            right, bottom = math.ceil((x + cw) * iw / width), math.ceil((y + ch) * ih / height)
            image = image[top:bottom, left:right]
            # Actual crop bounds can differ after downsample rounding.
            x, y, cw, ch = left * width / iw, top * height / ih, (right - left) * width / iw, (bottom - top) * height / ih
        if "max_size" in params:
            try:
                limit = int(params["max_size"][0])
            except ValueError:
                raise ApiError("max_size must be an integer") from None
            require(64 <= limit <= 4096, "max_size must be 64..4096")
            scale = min(1, limit / max(image.shape[:2]))
            if scale < 1:
                image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                                   interpolation=cv2.INTER_NEAREST if path == "/mask" else cv2.INTER_AREA)
        metadata = {"source_width": width, "source_height": height, "width": image.shape[1], "height": image.shape[0],
                    "source_crop": [x, y, cw, ch], "source_units_per_pixel": [cw / image.shape[1], ch / image.shape[0]],
                    "mask_pixels": None if source != w._selected_path() and view == "original" else int(mask.sum()),
                    "image_index": image_index, "view": view,
                    "mask_source": "edited" if source == w._selected_path() and w.mask_edit_active else "automatic" if source == w._selected_path() or view != "original" else "none"}
        if path == "/preview/metadata":
            return 200, "application/json", metadata, {}
        ok, encoded = cv2.imencode(".png", image)
        require(ok, "PNG encoding failed", 500)
        return 200, "image/png", encoded.tobytes(), {"X-Agent-Metadata": json.dumps(metadata)}

    def handle(self, method, raw_path, data):
        parsed = urlsplit(raw_path)
        path, params = parsed.path, parse_qs(parsed.query)
        if method == "GET" and path in {"/preview", "/preview/metadata", "/mask", "/snapshot"}:
            return self.image_response(path, params, raw_path)
        if method == "GET" and path == "/":
            value = self.describe()
        elif method == "GET" and path == "/state":
            value = self.state()
        elif method == "GET" and path.startswith("/jobs/"):
            key = path.removeprefix("/jobs/")
            require(key in self.jobs, "Unknown or expired job", 404)
            value = dict(self.jobs[key])
        elif method == "POST" and path == "/commands":
            require(isinstance(data, dict) and type(data.get("operation")) is str, "Body needs operation and arguments")
            require(not set(data) - {"operation", "arguments"}, "Unknown request fields")
            value = self.command(data["operation"], data.get("arguments", {}))
        else:
            raise ApiError("Unknown endpoint; GET / for available operations", 404)
        return 200, "application/json", value, {}


class AgentServer:
    def __init__(self, window, port=8765):
        self.controller = AgentController(window)
        controller = self.controller

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            def log_message(self, *_args):
                pass

            def do_GET(self):
                self.dispatch()

            def do_POST(self):
                self.dispatch()

            def dispatch(self):
                try:
                    # Prevent drive-by browser requests and DNS-rebinding hosts.
                    require(not self.headers.get("Origin"), "Browser-origin requests are not accepted", 403)
                    require(self.headers.get("Host") == f"127.0.0.1:{self.server.server_port}", "Invalid Host", 403)
                    require(not self.headers.get("Transfer-Encoding"), "Use Content-Length; chunked requests are not supported")
                    length = int(self.headers.get("Content-Length", 0))
                    require(0 <= length <= 2_000_000, "Request too large", 413)
                    require(self.command == "POST" or length == 0, "GET requests must not include a body")
                    data = None
                    if self.command == "POST":
                        require(self.headers.get_content_type() == "application/json", "Use application/json", 415)
                        data = json.loads(self.rfile.read(length))
                    status, mime, value, headers = controller.submit(self.command, self.path, data)
                except (ApiError, ValueError, UnicodeError) as error:
                    self.close_connection = True
                    status, mime, value, headers = getattr(error, "status", 400), "application/json", {"error": str(error)}, {}
                body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8") if mime == "application/json" else value
                shutting_down = status == 200 and isinstance(data, dict) and data.get("operation") == "app.shutdown"
                if shutting_down:
                    self.close_connection = True
                self.send_response(status)
                self.send_header("Content-Type", mime + "; charset=utf-8" if mime == "application/json" else mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                if self.close_connection:
                    self.send_header("Connection", "close")
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()
                try:
                    self.wfile.write(body)
                    self.wfile.flush()
                    if shutting_down:
                        controller.shutdown.requested.emit()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.http.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.http.server_port}"
        self.thread = threading.Thread(target=self.http.serve_forever, name="agent-http", daemon=True)
        self.thread.start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()
        self.controller.timer.stop()
