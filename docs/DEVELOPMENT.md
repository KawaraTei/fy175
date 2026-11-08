# ソース版のセットアップとビルド

[READMEへ戻る](../README.md) · [macOSの手順](../MACOS.md)

## Windowsのセットアップ


Python 3.13をインストールし、リポジトリを取得したフォルダでPowerShellを
開いて、次のコマンドを順番に実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe -m auto_mosaic.app
```

`download_models.py`が必要なモデルを自動で`models`フォルダへ配置し、ファイルが
正しいかも確認します。ダウンロードにはインターネット接続と約170MB以上の
空き容量が必要です。

NudeNetはパッケージをインストールせず、固定バージョンの配布ファイルからモデルだけを
取り出し、配布ファイルとモデルのSHA-256を確認します。

正常に配置されるファイルは次の4つです。

- `models/nudenet-320n.onnx`
- `models/anime-censor-detect-v1.0-n.onnx`
- `models/sam2_hiera_tiny.encoder.onnx`
- `models/sam2_hiera_tiny.decoder.onnx`

モデル不足のメッセージが表示された場合は、リポジトリ直下で
`.\.venv\Scripts\python.exe scripts\download_models.py`をもう一度実行してください。
モデルの出典とライセンスは[MODEL_LICENSES.md](../MODEL_LICENSES.md)に記載しています。

## NVIDIA GPUで高速化する場合

CUDA対応のNVIDIA GPUでは、対象検出とSAM2によるマスク生成をGPUで実行できます。
画像・動画・Shift+クリックのAI領域選択に共通で適用されます。
ソース版のセットアップは次のとおりです。

```powershell
.\scripts\setup.ps1 -Gpu
.\.venv\Scripts\python.exe -m auto_mosaic.app
```

GPU用の依存関係は `requirements-gpu.txt` に固定しています。CUDA 13対応ドライバーが必要です。
CUDA/cuDNNのDLLは仮想環境へ導入するため、CUDA Toolkitの別途インストールは不要です。
CPU版とGPU版のONNX Runtimeは同じモジュールを使うため、同時にインストールしないでください。
`setup.ps1` は競合する版を取り除きます。CPU版へ戻す場合は `setup.ps1` を引数なしで実行します。

通常はGPUを自動選択し、利用できない場合はCPUで動作します。明示的に指定する場合は
起動前に `$env:AUTO_MOSAIC_DEVICE = 'cpu'` または `'cuda'` を設定します。
`'cuda'` 指定ではGPUを初期化できないとエラーになるため、CPUへの切替を見逃さず確認できます。
`Remove-Item Env:AUTO_MOSAIC_DEVICE` で自動選択へ戻ります。

ローカルのGPU対応EXEは `.\scripts\build.ps1 -Gpu` で作成します。既存のCPU版EXEは
仮想環境の更新だけではGPU対応になりません。GPUビルドはCUDA/cuDNNのDLLも含むため容量が増えます。
再配布前の確認事項は [DISTRIBUTION.md](../DISTRIBUTION.md) を参照してください。

速度比較は次のコマンドで、CPU/GPUに同じ合成画像を渡して実行できます。

```powershell
.\.venv\Scripts\python.exe -m scripts.benchmark_inference --output .codex-qa\gpu\benchmark.json
```

モデル初期化を除き、ウォームアップ後の5回を測定し、利用プロバイダーとマスクの差も記録します。
実画像の精度評価ではありません。読み込み、モザイク処理、動画の圧縮などは引き続きCPU処理です。
GPU実装は [ONNX RuntimeのCUDA仕様](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)に従います。

## Windows CPU版をビルドする

```powershell
.\scripts\build.ps1
```

出力先は `dist/FY175AutoMosaic` です。CPU版でセットアップ済みの環境では
`-UseExistingEnvironment` を付けると依存の再取得を省けます。GPU用環境ではGPU版が生成されます。
配布用のCPU版はCPU専用の仮想環境でビルドしてください。

動画用のFFmpegとffprobeは `auto_mosaic/video_io.py` の `SHARED_FFMPEG_DIR` から取得します。
この環境では `C:\Users\Admin\.codex\tools\ffmpeg\bin` です。
ソース実行は共有ツールを、Windows配布版は同梱ツールを使用します。
再配布時は [DISTRIBUTION.md](../DISTRIBUTION.md) と [THIRD_PARTY_SOURCES.md](../THIRD_PARTY_SOURCES.md) を確認してください。
