## Changes

- マスクと処理設定を元画像の隣の `.fy` ファイルへ保存し、次回の読み込み時に復元できるようにしました。「.fy保存」と「マスク復元」から利用できます
- マスク編集中に、Shift+クリックでAIが推定した領域を追加し、Ctrl+Shift+クリックで削除できるようにしました
- すべてのプレビュー表示からマスク編集を開始できるようにし、Ctrl+Zで直前の編集を戻せるようにしました
- 「マスク閾値」と「マスク拡張」を追加し、自動マスクの範囲を調整できるようにしました。拡張を0 pxにしても、細い隙間を埋める輪郭補正は維持されます
- 画像の読み込み・プレビュー・マスク編集・保存をエージェントから操作できるローカルHTTP APIを追加しました。`FY175AutoMosaic.exe --agent-api` で起動できます

## Source

- Tag: [1.0.5](https://github.com/KawaraTei/fy175/tree/1.0.5)
- Commit: [9655c50](https://github.com/KawaraTei/fy175/commit/9655c5021bdb9ec1353428bb6931f2061149ea31)
- [操作手順（README）](https://github.com/KawaraTei/fy175/blob/eb8d3c846d2a4bb99a3866f38cf719cd4d207200/README.md)
- [Full Changelog](https://github.com/KawaraTei/fy175/compare/1.0.4...1.0.5)
