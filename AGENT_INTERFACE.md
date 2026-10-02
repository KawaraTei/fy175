# Agent interface

アプリ内蔵のローカルHTTP APIです。追加サービス、Pythonパッケージ、外部ドキュメントは不要です。
既存のUIのウィンドウ・設定・処理・マスク編集状態を操作します。エージェント専用の別セッションは作りません。
このブランチは画像モードのみ対応します。動画機能は `feature/video-mode` に分離しています。

## 起動と発見

```powershell
.\.venv\Scripts\python.exe -m auto_mosaic.app --agent-api
```

配布版は `FY175AutoMosaic.exe --agent-api` です。既定のURLは `http://127.0.0.1:8765`。
ポート変更は `--agent-api 8766`。通常起動ではAPIを開きません。

エージェントに伝える情報は「URLにGETして使い方を取得する」だけです。`GET /` は操作一覧、
引数、現在のUIに基づく設定範囲、座標の扱い、操作順序を返します。MCP登録は不要です。
HTTPを扱えるツールなら使用できます。HTTP機能がないツールでは同梱の標準ライブラリだけの
クライアント、またはPythonの `urllib.request` を使用できます。

```powershell
.\.venv\Scripts\python.exe -m auto_mosaic.agent_client describe
.\.venv\Scripts\python.exe -m auto_mosaic.agent_client state
.\.venv\Scripts\python.exe -m auto_mosaic.agent_client call images.add '@input.json' --wait
.\.venv\Scripts\python.exe -m auto_mosaic.agent_client get '/preview?view=detection' --output coverage.png
```

`input.json` は `{"paths":["C:/images/example.png"]}` のようなUTF-8 JSONです。
`call` の引数にはJSON文字列も指定できます。PowerShellの引用符を避けるには `@ファイル` が便利です。
クライアントの `--url` は `describe` / `call` 等の前に指定します。

## 操作

`POST /commands` に `{"operation":"操作名","arguments":{...}}` を送信します。
レスポンスには共有UIの状態と、非同期処理ならジョブIDが含まれます。
`GET /jobs/ID` が `succeeded` / `failed` / `cancelled` になるまで確認してください。
失敗内容、出力ファイル名、一括処理で途中まで保存された元画像もジョブに返します。

- 画像: 複数読込、選択、除去、要手動対応メモ、設定変更、再解析、現在画像・一括保存。
- マスク: ブラシ、矩形、多角形による追加・削除、PNGによる全マスク置換、下書き破棄。
- 確認: 元画像・検出範囲・処理結果のPNG、原寸マスクPNG、実際のUIウィンドウのPNG。

APIの設定をUIへ伝播し、設定の型・範囲もUIから取得します。APIで別の既定値を管理しません。
検出設定を変えたら再解析してください。効果設定は既存マスクを保持してプレビューに反映します。
編集中のマスクがある操作を破棄する場合は明示的に `discard:true` を指定します。
画像の下書きは `image.save` で使われます。
`images.save_all` はUIと同じく各画像を再解析します。

## 座標とプレビュー

画像のインデックスは0始まり。マスク操作の点とブラシ幅は元画像のピクセル単位です。
`GET /preview?view=detection&max_size=960` などで画像を取得します。
切出しは `crop=x,y,width,height` を追加します。ヘッダー `X-Agent-Metadata` のJSONには
元サイズ、切出し位置、出力サイズ、元画像の単位／プレビューピクセルが含まれます。
プレビュー上の座標は、切出し原点に「座標 × source_units_per_pixel」を加えて元画像へ変換します。
`GET /mask` は原寸で取得します。ブラシ幅の省略時はUIの現在値を使用します。
指定したブラシ幅もUIへ反映し、許容範囲は `GET /` の `mask_edit_schema` に返します。
検出範囲の灰色の枠は閾値未満の候補です。枠があっても適用マスクがあるとは限りません。
`GET /state` は採用検出と閾値未満候補を区別して返します。

例: `mask.edit` の `shape:rectangle, action:add, points:[[100,100],[150,150]]` は
元画像の両端を含む矩形を追加します。`action:erase` なら同じ範囲を削除します。

## ローカル運用

通信は127.0.0.1のみ。ブラウザーのOrigin付きリクエストと不正なHostを拒否します。
APIを有効にしたアプリにはローカルのエージェントからファイル操作を委ねる前提です。
画像は外部に送信しません。ファイルパスは絶対パスを使用します。
画像保存は既存ファイルを上書きせず連番を付けます。
人間とエージェントによる同時操作は対象外です。APIの操作はQtスレッドに集約し、
既存ワーカーの実行中は次の変更を409で拒否します。読取りの状態・ジョブ確認は継続できます。
終了時のサーバー停止もアプリの終了へ連動します。

## 検証

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_agent_api -v
```

HTTP→Qt→画像処理を通す簡易統合テストです。検出・輪郭モデルは決定的なテスト用実装を使い、
マスク追加・削除、プレビューと保存結果の一致、マスク外の保持、
不正な設定の拒否と解析失敗の通知、画像専用の操作範囲を確認します。
実モデルでの独立サブエージェント検証は `AGENT_ACCEPTANCE.md` に記録します。
