## Changes

- 動画モードをβ版として追加しました。「動画モード(β)」から、動画の自動検出・モザイク／ぼかし処理・MP4への書き出しができます
- 動画のマスクを手動で修正し、動きに合わせて指定区間へ補完できるようにしました。編集内容は `.fyvideo` ファイルに保存し、続きから再開できます
- ソース版でNVIDIA GPUによる画像・動画の検出とマスク生成の高速化に対応しました。配布ZIPはCPU版です
- 「マスクの滑らかさ」を追加し、マスクの輪郭を調整できるようにしました

## Source

- Tag: [1.1.0](https://github.com/KawaraTei/fy175/tree/1.1.0)
- Commit: [0eecea0](https://github.com/KawaraTei/fy175/commit/0eecea0f4c2f1e86c922f6a1437d417e454a4ab2)
- [操作手順（README）](https://github.com/KawaraTei/fy175/blob/2f44e0c3a962523e815628fa66dbb9cba9fcb3a8/README.md)
- [同梱ライブラリのソース・ライセンス](https://github.com/KawaraTei/fy175/blob/2f44e0c3a962523e815628fa66dbb9cba9fcb3a8/THIRD_PARTY_SOURCES.md)（`ThirdPartySources.zip` はソース資料です。アプリの実行には不要です）
- [Full Changelog](https://github.com/KawaraTei/fy175/compare/1.0.5...1.1.0)
