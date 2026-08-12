# Open-Source Release Checklist

This checklist covers repository publication and optional Windows binary
releases. It is operational guidance, not legal advice.

## Before publishing the repository

- [ ] Confirm that `Hiyoko Typing` is the correct copyright holder name in
  `NOTICE` and `README.md`.
- [ ] Publish the repository under `AGPL-3.0-only` and keep the root `LICENSE`.
- [ ] Keep `NOTICE`, `THIRD_PARTY_NOTICES.md`, `MODEL_LICENSES.md`, and the entire
  `LICENSES` directory in the repository.
- [ ] Ensure model weights, user images, build output, virtual environments,
  caches, and local settings remain excluded by `.gitignore`.
- [ ] Add the public repository URL to `README.md` and to the application's
  visible license/about notice once the URL is known.
- [ ] Confirm that every contributor agrees to submit changes under
  `AGPL-3.0-only`.

## Before publishing a Windows binary

- [ ] Build only from a tagged, public source revision and record that tag or
  commit in the release notes.
- [ ] Verify that `dist/AutoMosaic` includes `LICENSE`, `NOTICE`,
  `THIRD_PARTY_NOTICES.md`, `MODEL_LICENSES.md`, `DISTRIBUTION.md`, and
  `LICENSES/`.
- [ ] Provide the exact corresponding source code at no charge, including build
  scripts and dependency declarations. Link it prominently from the binary
  download page and the application's legal notice.
- [ ] Display an appropriate legal notice in the interactive application:
  copyright, AGPL-3.0-only, no-warranty statement, and source-code location.
- [ ] Keep Qt DLLs dynamically replaceable. Do not add installer restrictions
  that prohibit reverse engineering for debugging modifications to LGPL code.
- [ ] Make the corresponding source for the redistributed Qt/PySide6 6.11.1
  libraries available in the manner required by LGPL-3.0, and retain the Qt
  third-party notices applicable to Core, GUI, Network, SVG, and Widgets.
- [ ] Re-run dependency and model license review after every version or checksum
  change.
- [ ] Test the archive from a clean Windows account and confirm that all notices
  can be opened without installing development tools.

## Changes that require a new review

- Static linking, a single-file packager, obfuscation, DRM, or an installer that
  prevents library replacement.
- Replacing or fine-tuning a model, or adding a new model source.
- Adding CUDA/GPU runtime libraries or other native dependencies.
- Changing from open-source distribution to closed-source, paid, hosted, or
  service-based use. AGPL network-use obligations and model license options must
  be reviewed before that change.
