# エージェントインターフェース実地検証

2026-10-02、ローカルAPI `http://127.0.0.1:8765` を、`GET /` が返す説明だけで操作した。実装コード・外部APIドキュメントは参照せず、実モデルを差し替えずに合成画像2枚で確認した。最終画像専用版でも再実施済み。`supported_media=["image"]` と、公開操作に動画操作がないことを自己説明から確認した。

## 確認結果

- 複数画像追加、選択、検出閾値/マスク閾値/拡張の変更、再解析が成功。非同期ジョブを最後までpollして成功を確認した。
- 元画像・検出・結果のPNGを取得。元画像400×240に対するcrop `[40,30,180,120]` / `max_size=96` は96×64となり、`source_units_per_pixel=[1.875,1.875]` を返した。元画像座標への対応を説明だけで判断できた。
- 原寸座標でブラシ・矩形・多角形の追加/消去を実行し、PNGマスク置換も成功。消去穴と編集後のぼかしを実物で確認した。
- 効果をblur/24に変更してマスクを維持。個別保存は編集プレビューとpixel完全一致し、元画像とは異なった。
- draft破棄後の一括保存は2枚とも成功し、既存の個別保存ファイルを上書きせず `_2` を付けた。
- 除去時の要手動対応メモが `/state` とQtウィンドウsnapshotに反映され、メモclearも成功。
- `effect_size=999` は400で範囲エラーを返した。
- 最終版でブラシ幅28を指定し、`GET /` の現在値28とQt snapshotの「28 px」表示を確認。続くdiameter省略の点描画は中心から13pxにマスクがあり15pxにマスクがなく、現在の28pxブラシを使ったことを確認した。
- 最終版で設定・再解析、3形状の追加/消去、PNG置換、個別/一括保存、メモ追加/clearを再実施し成功。最終版の編集プレビューと保存PNGもpixel完全一致した。

## 自己説明・制限

主要経路はAPI自身の説明だけで完結した。通常操作に予期しない失敗やアプリ側例外は生じなかった。
別途、HTTP→Qtの統合テスト3件で、編集保存とプレビューの一致、不正設定の拒否、
意図的な解析失敗がモーダル表示なしに失敗ジョブとして返ること、画像専用の公開範囲を確認した。
master上の既存画像UIチェック2件と構文・差分検査も通過した。

検出ビューには0%候補枠が表示される一方、解析結果は `detections=0`、適用マスクは0 pixelだった。先行版で挙げた説明不足は解決済み。最終版の `semantics.detection_view` は灰色枠が閾値未満候補でありマスクを生成しないこと、採用検出と候補をstateで区別することを説明している。検出0件は合成素材で正常な結果として扱い、実モデルの検出精度・見逃し防止能力は保証しない。

最初のサーバー接続拒否は、保持されるプロセスでの再起動後に解消した。先行して実行した動画操作ログは参考資料として残すが、今回の受入判定には含めない。最終画像専用版の受入経路に未解決の問題は確認していない。

## 成果物

保存先: `.codex-qa/agent-interface/acceptance/`

- `reference.json`: APIから取得した自己説明
- `run_acceptance.py`, `followup.py`: 実行手順（先行動画検証を含む）
- `log.json`: 全HTTP操作・応答・job結果・プレビュー座標metadata
- `image-contact-sheet.png`: 画像プレビュー、マスク、保存結果の確認一覧
- `image-window.png`, `image-manual-review.png`: 実際のQt画面
- `saved/`: 個別保存1枚・一括保存2枚
- `inspect_images.py`: 編集プレビューと保存PNGの一致確認
- `final_acceptance.py`: 最終画像専用版で実行した再確認手順
- `final/reference.json`, `final/log.json`, `final/assertions.json`: 最終版自己説明・全操作ログ・一致/幅確認結果
- `final/brush-28-window.png`: ブラシ幅28を反映したQt画面
- `final/image-edited-window.png`, `final/image-edited.png`, `final/saved/input_final.png`: 全形状の編集・保存一致を確認した成果物
- `final/manual-review-window.png`: メモ反映画面
- `final/final-window.png`, `final/final-preview.png`: 最後の表示状態と原寸プレビュー

画像プレビュー・マスク・保存成果物の実物、およびQt画面をdesign-qa手順で確認した。HTTP成功のみを表示確認とは扱っていない。

作業ブランチは `master` ベースの `codex/agent-interface`。
作業開始前からあった動画マスク編集差分は `feature/video-mode` の `a85db4f` へ保存し、
今回の画像APIの差分には含めていない。GrokBotでの確認と配布EXEの再ビルド・実行は未実施。

## 2026-10-03: GrokBotフィードバックへの対応

APIバージョン2で以下を変更した。上記の2026-10-02の検証は当時のAPIバージョン1の記録。

- JSONにUTF-8 charsetを付け、status/preview_viewを機械向けコード、表示文を別フィールドに分離。
- 閾値未満の灰色候補は確信度0.1以上・最大5件に制限。採用の検出閾値をユーザーが0.1未満に指定する場合は、その設定を尊重する。
- 解析結果に検出indexと対応する個別マスクを保持。検出単位の削除・拡張／縮小とマスク操作のundoを追加。
- ラベルや検出枠を表示しないマスク範囲ビューをAPI/UIへ追加。表示名の定義は共有。
- 解析・個別保存ジョブへ検出の詳細を返し、個別の出力ファイル名指定と応答送信後の終了操作を追加。
- 下書きは表示中の1枚のみ・編集後は個別保存・一括保存は再解析、という仕様をGET /とAGENT_INTERFACE.mdに明記。
  手動マスクにはmask_expansionを再適用しないこと、検出への帰属がない手動追加画素の扱いも明記。

検証:

- API統合テスト5件を通過。検出単位の削除で他の検出との重なりを保持、undoで画素単位に復元、
  拡張／縮小、原寸の個別マスク取得、任意の日本語出力名、既存ファイル拒否、保存とプレビューの一致を確認。
- 候補フィルタ・既存画像UI・解析の関連チェック6件を通過。構文・差分検査も通過。
- Windows PowerShell **5.1.26100.9444本体**でInvoke-RestMethodを実行し、
  日本語status_textがUTF-8から生成した期待文字列と完全一致することを確認。
  app.shutdownのHTTP応答を受信し、実アプリが終了コード0で終了した。
- 実際のQtウィンドウで検出ビューとマスク範囲ビューを比較し、検出の削除・拡張を実物確認。
  合成画像・決定的な検出を使用。元画像のサイズ・配置・表示領域は同条件で、原画像プレビューの画素も保持。
  offscreenの日本語フォント不足はMeiryoを読み込んで再取得した。表示範囲に未解決の問題なし。

証拠は `.codex-qa/agent-feedback/` の `powershell51.json`、`check-powershell51.ps1`、
`capture.py`、`detection-before.png`、`overlay-before.png`、`overlay-removed.png`、`overlay-dilated.png`、`visual-state.json`。
改修版のGrokBotによる再検証、配布EXEの再ビルド・実行、実モデルの検出精度は未確認。

## 2026-10-03: 保存経路・PID・番号指定プレビューの追加

- 個別／一括保存ジョブに `mask_source`、`used_edited_mask`、`detection_rerun` を追加。
  一括保存は `saved_images[]` で元画像・保存先・処理経路を対応付ける。
- `GET /` に実際のアプリPIDと親PID、終了方法を返す。HTTP/1.1の持続接続に対応。
  chunkedリクエスト本文には非対応で、Content-Lengthを使用する仕様を自己説明にも明記。
- `GET /preview/metadata` に座標変換情報をJSON本文で返す。
- `GET /preview?image_index=N` に未選択画像を指定できる。選択、編集中マスク、undo履歴、
  UI表示を保持する。初回の処理済みプレビューは202の解析ジョブを返し、完了後に同じURLを取得する。
  元画像ビューは解析不要。直近1枚の解析結果をキャッシュし、設定または元ファイル変更で無効化する。

検証:

- 関連するAPI統合テスト7件を通過。編集マスク保存／再検出保存／一括保存の処理経路、
  同一ソケット上のHTTP/1.1応答2回、番号指定プレビュー、JSONとPNGヘッダーの座標一致、
  解析キャッシュ、選択・下書き・undo保持、番号指定解析の失敗通知を確認。
- 実アプリを起動し、PowerShell 5.1.26100.9444でAPIの `pid=36336` が
  Get-NetTCPConnectionの待受所有PIDと一致することを確認。親PIDは19988。
  app.shutdownに正常応答し、起動セッションが終了コード0で終了。
- design-qa手順で、未選択画像のマスク範囲PNGと取得後のQt画面を実物確認。
  合成画像・決定的な検出を使用し、元の画像が選択されたまま、手動追加マスクと編集状態が残る。
  対象範囲に未解決の表示問題なし。証拠は `.codex-qa/agent-additions/` の
  `live.json`、`indexed-overlay.png`、`selected-before.png`、`selected-after.png`、`metadata.json`。

今回の追加機能のGrokBotによる再検証と配布EXEの実行は未実施。
