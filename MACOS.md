# macOS

## ソース版を使う

Python 3.13で動作します。依存パッケージの対応OSはApple SiliconがmacOS 14以降、
IntelがmacOS 13以降です。MacではCPUで推論します。CUDA用の設定は不要です。

Homebrewを利用する場合の準備と起動手順です。

```bash
brew install python@3.13 ffmpeg
bash scripts/setup_macos.sh
.venv/bin/python -m auto_mosaic.app
```

別の場所のPythonを使う場合は `PYTHON=/path/to/python3.13 bash scripts/setup_macos.sh`
と指定できます。Apple Siliconではarm64版Pythonを使用してください。
セットアップはプロジェクト内の `.venv` に固定依存を導入し、既存と同じSHA-256検証付きの
モデル取得処理を実行します。画像を外部へ送信する処理はありません。

ONNX Runtimeは `requirements.txt` でCPU別に選択します。Apple Siliconは1.28.0、
Intelはx86_64用Python 3.13 wheelがある1.23.2です。ほかの共通依存はWindows版と同じです。

## 操作と保存先

- 画像、PSD/CLIPレイヤー、動画モードを利用できます。各モードの形式上の制約はREADMEを参照してください。
- 既定の出力先は `~/Documents/FY175AutoMosaic/output` です。アプリ内から変更できます。
- マスクを戻す操作はCmd+Z、動画編集を保存する操作はCmd+Sです。
- マスクの削除ブラシはOption+ドラッグ、AI領域の削除はCmd+Shift+クリックです。
- 保存先を開くボタンはFinderを使用します。
- 動画モードは外部のFFmpeg / ffprobeを使用します。まずPATH、次にHomebrewの
  `/opt/homebrew/bin` / `/usr/local/bin` を確認するため、Finder起動でも利用できます。

## `.app` を作る

ビルドはMac上で実行します。Windows上からMac用アプリは生成できません。

```bash
.venv/bin/python -m scripts.build_macos
```

出力は `dist/macos-arm64/FY175AutoMosaic.app` または
`dist/macos-x86_64/FY175AutoMosaic.app` です。ビルドしたMac/Pythonと同じCPU向けになります。
Python・モデル・画像／レイヤー処理の依存は同梱し、動画用FFmpegは同梱しません。
動画を使うMacには `brew install ffmpeg` が必要です。

`.app` はローカル確認用のアドホック署名を付けて生成します。公開配布用のDeveloper ID署名・
Apple公証はこの手順には含みません。署名IDを持つ場合は `--codesign-identity` を指定できます。
実際のビルド環境と依存の版は `.app/Contents/Resources/LICENSES/macos-python/BUILD.json` に記録します。

## 検証

GitHub Actionsの `macOS` ワークフローはApple SiliconとIntelで、依存導入、
画像の実モデル推論、レイヤーの保存・復元、動画の読み書き、Cocoaでの画面キャプチャ、
`.app`ビルド、同梱モデルとネイティブライブラリの起動を確認します。
これは通常の利用環境での全機能保証や、署名・公証済みリリースを意味しません。

ローカルで同じ主要部分を確認する場合：

```bash
.venv/bin/python -m unittest tests.test_platform_support tests.test_layers tests.test_layer_ui tests.test_video -v
.venv/bin/python -m tests.smoke_models
.venv/bin/python -m scripts.capture_macos_qa
dist/macos-arm64/FY175AutoMosaic.app/Contents/MacOS/FY175AutoMosaic --smoke-test
dist/macos-arm64/FY175AutoMosaic.app/Contents/MacOS/FY175AutoMosaic --platform-smoke
```

Intelでは上記の `macos-arm64` を `macos-x86_64` に置き換えてください。

実装の参照先：[QtのMacキー割り当て](https://doc.qt.io/qt-6/qkeysequence.html)、
[PyInstallerのMac対応](https://pyinstaller.org/en/stable/feature-notes.html#macos-multi-arch-support)、
[ONNX Runtime配布物](https://pypi.org/project/onnxruntime/1.23.2/#files)。
