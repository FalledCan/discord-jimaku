"""Discord 字幕: VC で相手が話した内容を「名前：文字」で画面に表示する。"""
import json
import os
import sys
import tkinter as tk
from tkinter import messagebox, simpledialog

from capture import LoopbackSegmenter
from discord_rpc import DiscordVoiceTracker
from overlay import Overlay
from transcriber import Transcriber
from voiceprint import VoicePrints

BASE_DIR = os.path.dirname(sys.executable if getattr(sys, "frozen", False) else os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")
TOKEN_PATH = os.path.join(BASE_DIR, "token.json")

# exe（コンソール無し）では stdout/stderr が無いのでログファイルに書く
if sys.stdout is None or sys.stderr is None:
    sys.stdout = sys.stderr = open(os.path.join(BASE_DIR, "log.txt"), "w", encoding="utf-8", buffering=1)

DEFAULT_CONFIG = {
    "client_id": "",
    "client_secret": "",
    "whisper_model": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "language": "ja",
    "audio_device": "",
    "hint": "",
    "save_audio": False,
    "max_lines": 6,
    "line_seconds": 20,
    "font_size": 20,
    "opacity": 0.85,
}


def load_config(root: tk.Tk) -> dict | None:
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            config.update(json.load(f))

    if not config["client_id"] or not config["client_secret"]:
        messagebox.showinfo(
            "Discord 字幕 - 初回設定",
            "Discord Developer Portal で作成したアプリの\nClient ID と Client Secret を入力してください。\n"
            "（手順は README.md を参照）",
            parent=root,
        )
        cid = simpledialog.askstring("初回設定", "Client ID:", parent=root)
        if not cid:
            return None
        secret = simpledialog.askstring("初回設定", "Client Secret:", parent=root, show="*")
        if not secret:
            return None
        config["client_id"], config["client_secret"] = cid.strip(), secret.strip()

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def main():
    root = tk.Tk()
    root.withdraw()
    config = load_config(root)
    if config is None:
        return
    root.deiconify()

    overlay = Overlay(root, config["max_lines"], config["line_seconds"], config["font_size"], config["opacity"])

    rpc_status = {"msg": ""}

    def on_rpc_status(msg):
        rpc_status["msg"] = msg
        if transcriber.ready.is_set():
            overlay.set_status(msg)

    tracker = DiscordVoiceTracker(config["client_id"], config["client_secret"], TOKEN_PATH, on_status=on_rpc_status)

    def on_lines(seg_id, lines, final):
        named = [(uid, tracker.name_of(uid), text) for uid, text in lines]
        if final:
            for _, name, text in named:
                print(f"{name}: {text}")
        overlay.set_segment(seg_id, named, final)

    def on_transcriber_status(msg):
        overlay.set_status(msg)

    voiceprints = VoicePrints(BASE_DIR, on_status=on_transcriber_status)
    transcriber = Transcriber(config["whisper_model"], config["language"], voiceprints, on_lines,
                              on_status=on_transcriber_status, hint=config["hint"],
                              debug_dir=os.path.join(BASE_DIR, "debug_audio") if config["save_audio"] else None,
                              models_dir=os.path.join(BASE_DIR, "models"))
    segmenter = LoopbackSegmenter(tracker, transcriber.submit_final, transcriber.submit_partial,
                                  on_status=overlay.set_status,
                                  device_name=config["audio_device"])

    def when_model_ready():
        if transcriber.ready.is_set():
            overlay.set_status(rpc_status["msg"] or "Discord に接続中…")
        else:
            root.after(500, when_model_ready)

    transcriber.start()
    tracker.start()
    segmenter.start()
    when_model_ready()
    root.mainloop()


if __name__ == "__main__":
    main()
