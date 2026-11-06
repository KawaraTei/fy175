from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from auto_mosaic import platform_support as platform
from auto_mosaic import video_io


class PlatformTests(unittest.TestCase):
    def test_mac_output_is_outside_application_bundle(self):
        with patch.object(platform.sys, "platform", "darwin"), patch(
            "PySide6.QtCore.QStandardPaths.writableLocation", return_value="/Users/test/Documents"
        ):
            self.assertEqual(platform.default_output_dir(), Path("/Users/test/Documents/FY175AutoMosaic/output"))
            self.assertEqual(platform.command_key(), "Cmd")
            self.assertEqual(platform.file_manager_name(), "Finder")

    def test_windows_retains_portable_output_and_shared_ffmpeg(self):
        with patch.object(platform.sys, "platform", "win32"), patch.object(platform, "application_root", return_value=Path("portable")):
            self.assertEqual(platform.default_output_dir(), Path("portable/output"))
            self.assertEqual(platform.command_key(), "Ctrl")
            with patch.object(Path, "is_file", return_value=True):
                self.assertEqual(video_io.tool_path("ffmpeg"), video_io.SHARED_FFMPEG_DIR / "ffmpeg.exe")

    def test_mac_source_finds_extensionless_tools(self):
        with patch.object(platform.sys, "platform", "darwin"), patch.object(video_io.shutil, "which", return_value="/opt/homebrew/bin/ffmpeg"):
            self.assertEqual(video_io.tool_path("ffmpeg"), Path("/opt/homebrew/bin/ffmpeg").resolve())

    def test_mac_finder_checks_homebrew_when_path_is_missing(self):
        with patch.object(platform.sys, "platform", "darwin"), patch.object(video_io.shutil, "which", return_value=None), patch.object(
            Path, "is_file", side_effect=lambda: True
        ), patch.object(Path, "stat") as stat:
            stat.return_value.st_mode = 0o755
            self.assertEqual(video_io.tool_path("ffprobe"), Path("/opt/homebrew/bin/ffprobe").resolve())

    def test_bundle_tools_are_authoritative_and_mac_external_tools_are_optional(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(platform.sys, "platform", "darwin"), patch.object(platform.sys, "frozen", True, create=True), patch.object(
                platform.sys, "_MEIPASS", str(root), create=True
            ), patch.object(video_io.shutil, "which", return_value="/usr/local/bin/ffmpeg") as which:
                self.assertEqual(video_io.tool_path("ffmpeg"), Path("/usr/local/bin/ffmpeg").resolve())
                (root / "ffmpeg").mkdir()
                which.reset_mock()
                with self.assertRaises(FileNotFoundError):
                    video_io.tool_path("ffmpeg")
                which.assert_not_called()
                (root / "ffmpeg" / "ffmpeg").touch()
                self.assertEqual(video_io.tool_path("ffmpeg"), root / "ffmpeg" / "ffmpeg")


if __name__ == "__main__":
    unittest.main()
