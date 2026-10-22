# Third-party sources for FY175AutoMosaic 1.1.0

## Source-only layer support added after 1.1.0

The unreleased layer workflow adds `psdparse==0.13.0` and `clipparse==0.2.0`.
These packages are not included in the existing 1.1.0 assets described below.
Both upstream projects use the MIT license; the notices are retained in `LICENSES`.

- [psdparse source](https://github.com/wamsoft/psdparse), [0.13.0 distribution](https://pypi.org/project/psdparse/0.13.0/)
- [clipparse source](https://github.com/wamsoft/clipparse), [0.2.0 distribution](https://pypi.org/project/clipparse/0.2.0/)

Before a future binary release, inventory the native wheels' bundled zlib / SQLite
and other build dependencies and retain their applicable notices as well.

## Published 1.1.0

The Windows `FY175AutoMosaic.zip` is the CPU build. It contains no NVIDIA CUDA
or cuDNN runtime DLLs. The source application's AGPL-3.0-only license is unchanged.

The companion [ThirdPartySources.zip](https://github.com/KawaraTei/fy175/releases/download/1.1.0/ThirdPartySources.zip)
provides the source archives, fixed build recipes, source inventory and SHA-256
hashes for the bundled FFmpeg tools and Qt/PySide libraries. It is not required
to run the application. Original notices are also in `LICENSES/third-party/`
inside the Windows ZIP.

## FFmpeg / ffprobe

- License: GPL-3.0-or-later, including the enabled GPL dependencies.
- Binary: `n8.1.2-40-g852b0552f0-20260814`, Windows x64, static GPL 8.1 build.
- [FFmpeg source revision](https://github.com/FFmpeg/FFmpeg/tree/852b0552f072826723749db6891928d839f536d8).
- [BtbN build recipes](https://github.com/BtbN/FFmpeg-Builds/tree/590a6612d7d961e9258429e501619e0b7d7cbedf).
- [Original successful build run](https://github.com/BtbN/FFmpeg-Builds/actions/runs/31799673813).
- `FFMPEG_BUILD.txt` in the Windows ZIP records the configure flags and binary hashes.

The source ZIP contains the FFmpeg source, complete build-recipe snapshot,
the Windows GPL recipe dependency archives, required submodule sources,
shaderc DEPS sources, and rav1e's Cargo.lock dependency archives. SVN inputs
are LAME r6761 and Xvid r2204. AMF is limited to the headers used by the build;
AMD SDK binaries are not included. The inventory includes some build-time or
test dependencies in addition to code used at runtime.

Use the included `FFmpeg-Builds` snapshot for the original build procedure:
`./makeimage.sh win64 gpl 8.1` followed by `./build.sh win64 gpl 8.1`.
To reproduce this historical revision, check out the FFmpeg revision above
in the source clone created by `build.sh`, rather than taking the current 8.1
branch tip. The recorded configure line identifies the original binary options.
Dependency repositories/revisions and SHA-256 hashes are in `SOURCE_MANIFEST.json`.
The unmodified recipe also records its build-tool downloads and the `cargo update cc`
build-helper step. This is source availability, not a claim of bit-for-bit rebuild equality.

## Qt / PySide / shiboken

Version 6.11.1 is provided under LGPL-3.0-only. The source ZIP includes Qt Base
(Core, GUI, Network, Widgets and platform support), SVG, image format plugins,
translations, and the PySide/shiboken source tree. Qt remains dynamically linked;
replacement DLLs can be installed in `_internal/PySide6`. No restriction on
reverse engineering for debugging modifications to these LGPL libraries is imposed.

- [Qt 6.11.1 official module sources](https://download.qt.io/archive/qt/6.11/6.11.1/submodules/)
- [PySide / shiboken 6.11.1 official sources](https://download.qt.io/official_releases/QtForPython/pyside6/PySide6-6.11.1-src/)

The original archive contents preserve the relevant per-file notices,
attribution files and license texts. See `THIRD_PARTY_NOTICES.md` and
`MODEL_LICENSES.md` for the other bundled software and model notices.
