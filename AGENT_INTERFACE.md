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

`GET /` の `pid` は実際にHTTPを待ち受けるアプリのPID、`parent_pid` は親プロセスです。
Windowsのランチャー等で起動時のPIDと異なる場合も、このPIDでアプリを識別できます。
終了には `POST /commands` の `app.shutdown` を使用してください。
未保存マスクがあれば保存するか、`arguments:{"discard":true}` を指定します。

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
`GET /jobs/ID` が `succeeded` / `failed` になるまで確認してください。
失敗内容、出力ファイル名、一括処理で途中まで保存された元画像もジョブに返します。

- 画像: 複数読込、選択、除去、要手動対応メモ、設定変更、再解析、現在画像・一括保存。
- マスク: ブラシ、矩形、多角形による追加・削除、PNGによる全マスク置換、下書き破棄。
  検出単位の削除・拡張／縮小、全体の拡張／縮小、1操作ずつのundo。
- 確認: 元画像・検出範囲・マスク範囲・処理結果のPNG、原寸マスクPNG、実際のUIウィンドウのPNG。
- 保存・終了: `image.save {path}` で画像ごとに任意のPNG/JPEGファイル名を指定できます。
  既存ファイルは上書きせず拒否します。`app.shutdown` は応答送信後に終了し、処理中は拒否します。

APIの設定をUIへ伝播し、設定の型・範囲もUIから取得します。APIで別の既定値を管理しません。
検出設定を変えたら再解析してください。効果設定は既存マスクを保持してプレビューに反映します。
編集中のマスクがある操作を破棄する場合は明示的に `discard:true` を指定します。
画像の下書きは `image.save` で使われます。元画像の隣に `画像.png.fy` があれば、
画像を選択した際にマスクと処理設定を自動復元します。復元後の `image.save` は再解析せずマスクを使います。
`image.save_mask_settings` は現在のマスクと設定だけを `.fy` に保存します。画像は書き出しません。
`image.restore_mask_settings` は元画像の隣の `.fy` を復元します。別の場所なら `path` で `.fy` を指定します。
APIではファイル選択画面を開かず、対応するファイルがなければパス指定を求めるエラーを返します。
`settings.update` の `save_mask_settings:true` は、個別・一括の画像保存に合わせて `.fy` を更新する設定です。
`images.save_all` は保存済み `.fy` のマスクがあれば現在の設定で処理し、なければ再解析します。
下書きは表示中の1枚だけで、切替・再解析・終了をまたいでは保持しません。
複数の画像を編集して保存する場合は `select → edit → image.save` を画像ごとに繰り返してください。
未保存の下書きは `images.save_all` に反映されません。`.fy` に保存したマスクは一括処理でも使えます。
切替や終了時の下書き破棄には `discard:true` が必要です。

JSONレスポンスは `application/json; charset=utf-8` です。PowerShell 5.1でも日本語を取得できます。
APIバージョン2では `image.status` と `preview_view` は機械向けコードを返し、
表示文は `status_text` と `preview_view_text` に分けています。コード一覧は `GET /` に含めます。
解析・個別保存ジョブは `detection_count` に加え、検出のindex・種類・確信度・矩形・マスク画素数も返します。
保存ジョブの `result.mask_source` は `edited`（編集中）、`restored`（`.fy` から復元）、
`reanalyzed`（検出を再実行）、`mixed`（一括処理に複数経路が混在）です。
`used_edited_mask` と `detection_rerun` も真偽値で返します。
一括保存の `saved_images[]` に元画像の `path`、保存先の `output` と画像ごとの処理経路情報を返します。

### 検出単位の編集とundo

`GET /state` の `detections[].index` と `GET /mask?index=N` が対応します。
indexは現在の解析内だけで有効で、再解析後は取得し直してください。
`mask.remove_detection {index}`、`mask.dilate {index,px}`、`mask.erode {index,px}` は
その検出が所有する領域を変更し、他の検出と重なる領域を残します。
`dilate/erode` のindex省略時はマスク全体を対象にします。半径pxの楕円カーネルを使い、範囲は1〜100pxです。

ブラシで自動マスクの外に追加した部分など、検出に帰属しない手動領域は検出単位の削除では消しません。
全マスクの置換・全体の拡張で増えた画素も新たな検出へ帰属させません。必要ならブラシ等で削除します。
検出時の矩形・確信度は編集では変わらず、現在のマスク画素数を別に返します。

`mask.undo` はマスク操作を1手戻します。最大20操作、圧縮したマスクの合計128MiBまで保持します。
下書き破棄、再解析、画像切替、画像保存成功で履歴をリセットします。
`image.save_mask_settings` は編集中の履歴を保持します。設定変更はundo対象に含めません。
`mask_expansion` は自動輪郭の設定で、手動編集には再適用しません。手動調整には `mask.dilate/erode` を使用します。

## 座標とプレビュー

画像のインデックスは0始まり。マスク操作の点とブラシ幅は元画像のピクセル単位です。
`GET /preview?view=detection&max_size=960` などで画像を取得します。
切出しは `crop=x,y,width,height` を追加します。ヘッダー `X-Agent-Metadata` のJSONには
元サイズ、切出し位置、出力サイズ、元画像の単位／プレビューピクセルが含まれます。
プレビュー上の座標は、切出し原点に「座標 × source_units_per_pixel」を加えて元画像へ変換します。
`GET /preview/metadata` に同じクエリを指定すると、同じ座標変換情報をJSON本文で取得できます。
`image_index` と `view`、プレビューの `mask_source`（`edited` / `restored` / `automatic` / `none`）も含めます。

`GET /preview?image_index=1&view=mask_overlay` のように画像番号を指定すると、未選択画像も取得できます。
現在の選択・編集中マスク・undo履歴・UI表示は切り替えません。選択中の画像番号なら現在の下書きを使用します。
未選択画像の処理済みビューは保存済みマスクまたは現在の設定での解析を使い、直近1枚の結果をキャッシュします。
設定、元ファイル、`.fy` が変わると処理をやり直します。初回はHTTP 202のJSONで `job` と `retry_url` を返すので、
ジョブ完了を確認して同じURLを再取得してください。失敗時はジョブの `error` を確認します。
`/preview/metadata` も同じ動作です。`view=original` は解析せず取得でき、未選択画像の
`mask_source` は `none`、未取得の `mask_pixels` は `null` になります。
`/mask` の `index` は検出番号であり、画像番号の指定には対応しません。

`GET /mask` は原寸で取得します。ブラシ幅の省略時はUIの現在値を使用します。
指定したブラシ幅もUIへ反映し、許容範囲は `GET /` の `mask_edit_schema` に返します。
検出範囲の灰色の枠は確信度0.1以上かつ検出閾値未満の候補で、最大5件です。
0.1未満の候補はモデル出力を選別する段階で除き、人間向けUI・APIの両方に適用します。
ユーザーが検出閾値を0.1未満に下げて採用する検出には、この候補表示の下限を適用しません。
枠があっても適用マスクがあるとは限りません。
`GET /state` は採用検出と閾値未満候補を区別して返します。
`GET /preview?view=mask_overlay` は元画像に被覆範囲と輪郭を重ね、枠・ラベルを表示しません。
同じ表示は人間向けUIの「マスク範囲」でも選択できます。

例: `mask.edit` の `shape:rectangle, action:add, points:[[100,100],[150,150]]` は
元画像の両端を含む矩形を追加します。`action:erase` なら同じ範囲を削除します。

## ローカル運用

通信は127.0.0.1のみ。ブラウザーのOrigin付きリクエストと不正なHostを拒否します。
HTTP/1.1の持続接続に対応し、応答には `Content-Length` を付けます。
リクエスト本文は `Content-Length` で送信してください。chunked形式の本文には対応せず、エラーを返して接続を閉じます。
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
