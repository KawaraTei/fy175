# Auto Mosaic

Windows向けの個人用自動モザイク試作です。PNG/JPEGを最大100枚読み込み、`penis` と `vagina / pussy` を検出して、SAM2の輪郭マスクにモザイクまたはぼかしを適用します。

## 現在の機能

- PNG/JPEG画像をウィンドウへ複数同時にドラッグ＆ドロップ
- 画像を選択すると自動解析し、処理結果をプレビュー
- 表示中の1枚だけ、または読み込んだ全画像を処理して保存
- 出力ファイルのsuffixを指定（既定値 `_mosaic`、空欄も可）
- 「選択を削除」は通常除去、「選択を除去（メモ）」は画像名を手動対応メモへ転記
- 処理成功後に対象画像をリストから自動除去するオプション
- 処理対象画像と手動対応メモの高さをドラッグで変更可能
- 実写（NudeNet）とイラスト（DeepGHS）の手動切り替え
- `penis`、`vagina / pussy` のチェック選択
- 検出閾値、モザイク／ぼかし、処理サイズの指定
- 元画像、検出範囲、処理結果のプレビュー
- PNG/JPEGの複数読み込みと指定フォルダへの一括保存
- SAM2には検出矩形と中心点を渡し、中心につながる輪郭成分だけを対象ごとに整形
- SAM2マスクが不自然な位置・大きさの場合の矩形フォールバック

## 開発実行

PowerShellで次を実行します。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\download_models.py
.\.venv\Scripts\python.exe -m auto_mosaic.app
```

モデルのダウンロードには約170MB以上の空き容量が必要です。初回推論時はONNXモデルの読み込みに時間がかかります。画像は外部へ送信されず、処理はローカルで完結します。

## Python不要のWindows版を作る

```powershell
.\scripts\build.ps1
```

完成物は `dist\AutoMosaic\AutoMosaic.exe` です。`AutoMosaic`フォルダ全体を移動・圧縮してください。単体EXEではなくフォルダ配布形式にして、起動時の大規模モデル展開を避けています。

## 注意

- 自動検出には誤検出と検出漏れがあります。保存前に「検出範囲」表示を確認してください。
- この試作は個人利用を前提にしています。
- NudeNetおよびDeepGHSの重みはUltralytics系モデルです。第三者への配布や販売を行う前に、重みとUltralyticsのライセンス条件を改めて確認してください。
- 入力画像は正当な権限を持つ成人コンテンツだけを使用してください。

## モデル配置

`scripts\download_models.py` は次のファイルを `models` に配置します。

- `nudenet-320n.onnx`
- `anime-censor-detect-v1.0-n.onnx`
- `sam2_hiera_tiny.encoder.onnx`
- `sam2_hiera_tiny.decoder.onnx`

詳細な出典は `THIRD_PARTY_NOTICES.md` を参照してください。
