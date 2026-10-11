# Discord 字幕

Discord の VC で相手が話している内容を、画面上にリアルタイムで表示する Windows 用ツールです。

```
たろう：こんにちは、今日のミーティングを始めましょう。
はなこ：資料は共有できていますか？
```

Bot は使いません。PC 上の exe だけで動きます。

## しくみ

| 役割 | 方法 |
|---|---|
| 音声 | PC のスピーカー出力（Discord から聞こえる音）をループバック録音 |
| 誰が話しているか | PC の Discord アプリにローカル接続（RPC）して「話し始め/話し終わり」を受け取る |
| 文字起こし | faster-whisper で [kotoba-whisper v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster)（日本語向けに学習し直した Whisper）をローカルの GPU で実行 |
| 同時に話したときの振り分け | 単語ごとの時刻を「話していた人」と照合。声が重なった部分は声紋で判定 |

自分の声は Discord から再生されないので表示されません（相手の声だけが表示されます）。

## 初回セットアップ（1回だけ）

名前を取得するため、Discord に自分専用の「アプリ」を登録します（Bot ではないので、サーバーには何も追加されません）。

1. https://discord.com/developers/applications を開き、**New Application** で適当な名前（例: `字幕`）を付けて作成
2. 左メニュー **OAuth2** を開く
   - **Client ID** をコピー
   - **Client Secret** の **Reset Secret** を押してコピー
   - **Redirects** に `http://localhost` を追加して **Save Changes**
3. `dist\DiscordJimaku\DiscordJimaku.exe` を起動し、Client ID と Client Secret を入力
4. Discord アプリに認証画面が出るので **認証** を押す

入力した値は exe と同じフォルダの `config.json` に、ログイン情報は `token.json` に保存されます。どちらも他人に渡さないでください。

## 使い方

1. Discord アプリを起動しておく
2. `DiscordJimaku.exe` を起動する（初回は音声認識モデル約 1.5GB のダウンロードで数分かかります）
3. VC に入ると自動で字幕が出ます。VC を移動しても自動でついていきます

| 操作 | 動作 |
|---|---|
| ドラッグ | ウィンドウを移動 |
| マウスホイール | 文字サイズを変更 |
| 右クリック | 終了 |

## 設定（config.json）

| 項目 | 既定値 | 説明 |
|---|---|---|
| `audio_device` | `""` | 録音する出力デバイス名の一部。空なら SteelSeries Sonar の「Chat」、無ければ Windows の既定の出力 |
| `whisper_model` | `kotoba-tech/kotoba-whisper-v2.0-faster` | 元の Whisper に戻すなら `large-v3-turbo`。軽くしたい場合は `small` や `medium`（`large-v3` は高精度だが遅すぎてリアルタイムには向かない） |
| `hint` | `""` | よく出る固有名詞・用語をスペース区切りで書くと認識されやすくなる（例: `"斬鉄 渦 ウォーリア"`） |
| `save_audio` | `false` | `true` にすると確定した発話の音声と結果を `debug_audio\` に保存（精度の調査用） |
| `language` | `ja` | 認識する言語。空にすると自動判定 |
| `max_lines` | `6` | 表示する最大行数 |
| `line_seconds` | `20` | 表示した行が消えるまでの秒数 |
| `font_size` | `20` | 文字サイズ |
| `opacity` | `0.85` | ウィンドウの不透明度（0〜1） |

## 注意点

- 録音デバイスに Discord 以外の音（動画・ゲームなど）が混ざると、それも拾うことがあります。誰かが話している間だけ録音しますが、Discord の出力先を専用のデバイス（Sonar の Chat など）に分けるとより正確になります。
- 複数人が話した区間は、単語ごとに話者を判定して行を分けます。声が重なった部分は声紋で判定します。声紋は 1 人だけが話した発言から自動で学習し、`speakers.json` に保存されます。使い始めは学習が進んでいないので、重なった部分の判定精度が低めです。
- 完全に声が重なると、Whisper は大きい方の声しか書き起こさないため、小さい方の声の一部が抜けることがあります。
- 話している途中から 0.5 秒ごとに仮の字幕（灰色）を更新し、話し終わる（約 0.7 秒の無音、または話者の交代）と確定版（白）に置き換えます。

## 開発者向け

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe src\main.py    rem exe にせず実行
build.bat                               rem dist\DiscordJimaku\ に exe を作成
```

`config.json` / `token.json` / `speakers.json` / `log.txt` / `models\` は exe（または `src\main.py`）と同じフォルダに作られます。
