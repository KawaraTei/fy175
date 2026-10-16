import os
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from auto_mosaic import inference


class InferenceDeviceTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {inference.DEVICE_ENV: "auto"})
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.preload = patch.object(inference, "_preload_cuda").start()
        self.available = patch.object(inference.ort, "get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]).start()
        self.session = Mock()
        self.session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.factory = patch.object(inference.ort, "InferenceSession", return_value=self.session).start()
        self.addCleanup(patch.stopall)

    def test_auto_uses_cuda_and_preserves_fp32(self):
        self.assertIs(inference.create_session(Path("model.onnx")), self.session)
        self.preload.assert_called_once()
        self.assertEqual(self.factory.call_args.kwargs["providers"], [("CUDAExecutionProvider", {"use_tf32": "0"}), "CPUExecutionProvider"])

    def test_cpu_override_skips_cuda_loading(self):
        os.environ[inference.DEVICE_ENV] = "cpu"
        inference.create_session(Path("model.onnx"))
        self.preload.assert_not_called()
        self.assertEqual(self.factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_cpu_package_works_in_auto_mode(self):
        self.available.return_value = ["CPUExecutionProvider"]
        inference.create_session(Path("model.onnx"))
        self.preload.assert_not_called()
        self.assertEqual(self.factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_auto_warns_when_runtime_silently_falls_back(self):
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        with self.assertLogs(inference.LOGGER, level="WARNING"):
            inference.create_session(Path("model.onnx"))
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])

    def test_explicit_cuda_does_not_silently_fall_back(self):
        os.environ[inference.DEVICE_ENV] = "cuda"
        self.session.get_providers.return_value = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "could not initialize"):
            inference.create_session(Path("model.onnx"))
        self.assertEqual(self.factory.call_count, 1)

    def test_explicit_cuda_fails_when_gpu_package_is_absent(self):
        os.environ[inference.DEVICE_ENV] = "cuda"
        self.available.return_value = ["CPUExecutionProvider"]
        with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
            inference.create_session(Path("model.onnx"))
        self.factory.assert_not_called()

    def test_auto_recovers_from_dll_load_exception(self):
        self.preload.side_effect = OSError("missing CUDA DLL")
        with self.assertLogs(inference.LOGGER, level="WARNING"):
            inference.create_session(Path("model.onnx"))
        self.assertEqual(self.factory.call_args.kwargs["providers"], ["CPUExecutionProvider"])


if __name__ == "__main__":
    unittest.main()
