# FY175AutoMosaic

Windows上でPNG/JPEG画像をまとめて確認し、検出した対象へモザイクまたは
ぼかしを適用するデスクトップアプリです。画像は外部へ送信されず、すべて
ローカルで処理されます。

## できること

- PNG/JPEG画像を複数まとめて、またはドラッグ＆ドロップで読み込み
- 実写／イラストを切り替え、対象を自動検出
- 検出範囲と処理後の画像を保存前にプレビュー
- 選択ブラシで検出マスクの不足部分を追加し、余分な部分を削除
- モザイク／ぼかしの種類、強さ、検出しやすさを調整
- モザイク／ぼかし処理した画像を指定フォルダへ保存
- 自動処理から除外した画像名を、要手動対応メモとして管理

## 起動方法

### Windows配布版を使う場合

1. 配布されたZIPを展開します。
2. `FY175AutoMosaic`フォルダ内の`FY175AutoMosaic.exe`を起動します。

`FY175AutoMosaic.exe`だけを別の場所へ移動しないでください。モデルや動作に必要な
ファイルを含むため、展開した`FY175AutoMosaic`フォルダ全体をそのまま使用します。

### ソースコードから起動する場合

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

正常に配置されるファイルは次の4つです。

- `models/nudenet-320n.onnx`
- `models/anime-censor-detect-v1.0-n.onnx`
- `models/sam2_hiera_tiny.encoder.onnx`
- `models/sam2_hiera_tiny.decoder.onnx`

モデル不足のメッセージが表示された場合は、リポジトリ直下で
`.\.venv\Scripts\python.exe scripts\download_models.py`をもう一度実行してください。
モデルの出典とライセンスは`MODEL_LICENSES.md`に記載しています。


## 基本的な使い方

1. 処理する画像を読み込みます。
2. 画像に合わせて「実写」または「イラスト」を選びます。
3. 検出対象と検出閾値を指定します。
4. プレビューで検出範囲と処理結果を確認します。
   どの表示でもマウスホイールで画像を段階的に拡大・縮小できます。画像上では
   カーソル位置、それ以外では画像中央が基準になり、初期表示より小さくはなりません。
5. 必要なら「検出範囲」表示から「マスクを編集」を開始します。ドラッグで
   範囲を追加し、Altキーを押しながらドラッグすると範囲を削除できます。
   編集中は検出矩形とラベルが非表示になり、マスクの範囲だけを確認できます。
   編集中に「処理結果」へ切り替えても、編集内容は保持されます。
6. 出力先を選び、「表示中を処理」または「全て処理」で保存します。

初回の画像表示時はモデルの読み込みに少し時間がかかることがあります。

編集したマスクは「表示中の1枚を処理して保存」でのみ使用され、画像を切り替えたり
リストから除去したりすると破棄されます。
<img width="1524" height="1022" alt="Adobe Express 2026-08-15 12 57 30" src="https://github.com/user-attachments/assets/8cd4c9fc-3719-4803-a792-6ec421eff97f" />


## 注意

- 自動検出には誤検出と検出漏れがあります。保存前に検出範囲を確認してください。

- 出力物は必ず目視で確認し、自己責任にて使用するようにしてください。

## ライセンス

Copyright (C) 2026 Hiyoko Typing

FY175AutoMosaicの独自コードはGNU Affero General Public License version 3 only
（`AGPL-3.0-only`）で公開します。完全な条文は`LICENSE`を参照してください。

モデルと依存ライブラリには個別のライセンスおよび表示義務があります。

- `NOTICE`: 本アプリの著作権・無保証・第三者成果物の案内
- `THIRD_PARTY_NOTICES.md`: 依存ライブラリとモデルの一覧
- `MODEL_LICENSES.md`: モデル固有の出典、チェックサム、ライセンス判断
- `LICENSES/`: 同梱するライセンス本文と第三者通知
- `DISTRIBUTION.md`: リポジトリ／Windowsバイナリ公開前チェックリスト

バイナリを再配布する場合は、そのバイナリに対応する完全なソースコードを提供し、
上記文書をすべて同梱してください。公開リポジトリURLが決まったら、README、
NOTICE、アプリ内のライセンス表示へURLを追記してください。
