"""faster-whisper による文字起こし（GPU があれば GPU）。"""
import collections
import glob
import os
import sys
import threading

import numpy as np

# Whisper が無音・雑音でよく出す幻聴フレーズ（句読点・空白を除いて照合）
HALLUCINATIONS = (
    "ご視聴ありがとうございました",
    "ご清聴ありがとうございました",
    "チャンネル登録",
    "最後までご覧いただきありがとうございます",
    "thankyouforwatching",
)
CONTEXT_CHARS = 120  # 直前の会話をどれだけプロンプトに入れるか


def _add_cuda_dll_dirs():
    """pip の nvidia-cublas / nvidia-cudnn の DLL を読み込めるようにする（exe 化後も含む）。"""
    if sys.platform != "win32":
        return
    roots = list(sys.path)
    if hasattr(sys, "_MEIPASS"):
        roots.insert(0, sys._MEIPASS)
    for root in roots:
        for d in glob.glob(os.path.join(root, "nvidia", "*", "bin")):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


# 蒸留モデルは decoder の層が少ないのに、config.json の alignment_heads が元の large-v3（32層）の
# ままになっているものがある。そのまま単語タイムスタンプを使うと存在しない層を参照して落ちるので、
# 最終層の全ヘッド（Whisper 既定の決め方）に直したコピーを使う。  モデル名 -> (decoder 層数, ヘッド数)
DISTILLED_MODELS = {
    "kotoba-tech/kotoba-whisper-v2.0-faster": (2, 20),
}


def prepare_model(model_name: str, models_dir: str) -> str:
    """WhisperModel に渡すモデル名またはローカルパスを返す。"""
    if model_name not in DISTILLED_MODELS:
        return model_name
    import json
    from huggingface_hub import snapshot_download

    layers, heads = DISTILLED_MODELS[model_name]
    path = snapshot_download(model_name, local_dir=os.path.join(models_dir, model_name.split("/")[-1]))
    config_path = os.path.join(path, "config.json")
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    fixed = [[layers - 1, h] for h in range(heads)]
    if config.get("alignment_heads") != fixed:
        config["alignment_heads"] = fixed
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    return path


BLOCK_SEC = 0.1      # capture.py のブロック長
EVENT_SLACK = 0.15   # Discord の話し始め/終わりイベントと音声のずれの許容


def _is_text(text: str) -> bool:
    return any(ch.isalnum() for ch in text)


def _is_hallucination(text: str) -> bool:
    plain = "".join(ch for ch in text if ch.isalnum()).lower()
    return any(h in plain for h in HALLUCINATIONS)


def _normalize(audio: np.ndarray) -> np.ndarray:
    """小さい声を聞き取りやすい音量に揃える（最大 10 倍まで）。"""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak <= 0:
        return audio
    return (audio * min(0.9 / peak, 10.0)).astype(np.float32)


class Transcriber(threading.Thread):
    def __init__(self, model_name: str, language: str | None, voiceprints, on_lines, on_status=print,
                 hint: str = "", debug_dir: str | None = None, models_dir: str = "models"):
        super().__init__(daemon=True)
        self.model_name = model_name
        self.models_dir = models_dir
        self.language = language or None
        self.hint = hint.strip()  # よく出る固有名詞・用語（認識のヒント）
        self.debug_dir = debug_dir  # 指定すると確定した発話の音声と結果を保存する
        self._context: collections.deque = collections.deque(maxlen=6)
        self.voiceprints = voiceprints
        self.on_lines = on_lines  # (発話ID, [(user_id, text)], 確定か)
        self.on_status = on_status
        self.ready = threading.Event()
        self._cv = threading.Condition()
        self._finals: collections.deque = collections.deque()
        self._partial = None  # 仮の文字起こしは最新の 1 件だけ持つ
        self._finished_ids: set[int] = set()

    def submit_final(self, seg_id: int, audio, activity):
        with self._cv:
            self._finals.append((seg_id, audio, activity))
            self._finished_ids.add(seg_id)
            if self._partial and self._partial[0] == seg_id:
                self._partial = None
            self._cv.notify()

    def submit_partial(self, seg_id: int, audio, activity):
        with self._cv:
            if seg_id not in self._finished_ids:
                self._partial = (seg_id, audio, activity)
                self._cv.notify()

    def _next_job(self):
        with self._cv:
            while not self._finals and self._partial is None:
                self._cv.wait()
            if self._finals:
                return (*self._finals.popleft(), True)
            job, self._partial = self._partial, None
            return (*job, False)

    def _load(self):
        _add_cuda_dll_dirs()
        from faster_whisper import WhisperModel

        model_path = prepare_model(self.model_name, self.models_dir)
        try:
            model = WhisperModel(model_path, device="cuda", compute_type="float16")
            # CUDA ライブラリが欠けていると最初の推論で落ちるので、ここで試す
            list(model.transcribe(np.zeros(16000, dtype=np.float32), language=self.language, word_timestamps=True)[0])
            print(f"[whisper] {self.model_name} (GPU)")
            return model
        except Exception as e:
            print(f"[whisper] GPU が使えないため CPU で実行します: {e}")
            return WhisperModel(model_path, device="cpu", compute_type="int8")

    # ---- 話者の割り当て ----
    @staticmethod
    def _speakers_between(activity, t0: float, t1: float) -> frozenset:
        """t0〜t1 秒に話していた人。誰もいなければ一番近い時刻に話していた人。"""
        n = len(activity)
        a = max(0, int((t0 - EVENT_SLACK) / BLOCK_SEC))
        b = min(n, int((t1 + EVENT_SLACK) / BLOCK_SEC) + 1)
        found = frozenset().union(*activity[a:b]) if a < b else frozenset()
        if found:
            return found
        center = int((t0 + t1) / 2 / BLOCK_SEC)
        for d in range(1, n):
            for i in (center - d, center + d):
                if 0 <= i < n and activity[i]:
                    return activity[i]
        return frozenset()

    def _attribute(self, words, activity, audio):
        """単語ごとに話者を決め、[(user_id, text, start, end, 全単語が確定的か)] を返す。"""
        cands = [self._speakers_between(activity, w.start, w.end) for w in words]
        who: list[str | None] = [next(iter(c)) if len(c) == 1 else None for c in cands]

        # 複数人が話していた単語は、1単語ずつ声紋で判定
        undecided = [i for i, c in enumerate(cands) if len(c) > 1]
        for i in undecided:
            who[i] = self.voiceprints.best_match(cands[i], audio, words[i].start, words[i].end)
        # 声紋でも決まらなければ、前後の確定した話者に合わせる
        for i in undecided:
            if who[i] is None:
                prev = next((who[k] for k in range(i - 1, -1, -1) if who[k]), None)
                nxt = next((who[k] for k in range(i + 1, len(words)) if who[k]), None)
                who[i] = prev if prev in cands[i] else nxt if nxt in cands[i] else sorted(cands[i])[0]

        # 同じ人の発言の途中で 1〜2 単語だけ別の人になったものは、前後の人に戻す
        i = 0
        while i < len(words):
            j = i
            while j + 1 < len(words) and who[j + 1] == who[i]:
                j += 1
            if 0 < i and j + 1 < len(words) and j - i < 2:
                around = who[i - 1]
                if around == who[j + 1] and all(around in cands[k] for k in range(i, j + 1)):
                    for k in range(i, j + 1):
                        who[k] = around
            i = j + 1

        lines = []
        for w, u, c in zip(words, who, cands):
            if u is None:
                continue
            if lines and lines[-1][0] == u:
                lines[-1][1].append(w.word)
                lines[-1][3] = w.end
                lines[-1][4] = lines[-1][4] and len(c) == 1
            else:
                lines.append([u, [w.word], w.start, w.end, len(c) == 1])
        return [(u, "".join(t).strip(), s, e, sure) for u, t, s, e, sure in lines]

    def _prompt(self) -> str | None:
        context = "".join(self._context)[-CONTEXT_CHARS:]
        prompt = f"{self.hint} {context}".strip()
        return prompt or None

    def _transcribe(self, model, audio, activity, final: bool):
        if audio is None or len(audio) < 16000 * 0.3 or np.sqrt(np.mean(audio ** 2)) < 0.003:
            return []
        audio = _normalize(audio)
        try:
            segments, _ = model.transcribe(
                audio,
                language=self.language,
                # 確定版は精度優先、仮の表示は速度優先
                beam_size=5 if final else 1,
                vad_filter=True,
                condition_on_previous_text=False,
                initial_prompt=self._prompt(),
                chunk_length=15,  # 発話は最長 15 秒（kotoba-whisper の推奨設定でもある）
                word_timestamps=True,
            )
            words = [w for s in segments if s.no_speech_prob < 0.6 for w in (s.words or [])]
        except Exception as e:
            print(f"[whisper] 文字起こし失敗: {e}")
            return []
        if _is_hallucination("".join(w.word for w in words)):
            return []
        lines = []
        for user_id, text, start, end, sure in self._attribute(words, activity, audio):
            if not _is_text(text):
                continue
            if final and sure:
                self.voiceprints.enroll(user_id, audio[int(start * 16000):int(end * 16000)])
            lines.append((user_id, text))
        if final:
            self._context.append("".join(t for _, t in lines))
            if self.debug_dir:
                self._save_debug(audio, lines)
        return lines

    def _save_debug(self, audio, lines):
        import time
        import wave

        os.makedirs(self.debug_dir, exist_ok=True)
        name = os.path.join(self.debug_dir, time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}")
        with wave.open(name + ".wav", "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        with open(name + ".txt", "w", encoding="utf-8") as f:
            f.write("\n".join(f"{u}: {t}" for u, t in lines))

    def run(self):
        self.on_status("音声認識モデルを読み込み中…（初回はダウンロードで数分かかります）")
        model = self._load()
        self.voiceprints.load()
        self.ready.set()
        while True:
            seg_id, audio, activity, final = self._next_job()
            lines = self._transcribe(model, audio, activity, final)
            with self._cv:
                # 仮の結果が出る前に確定してしまった発話は表示しない
                stale = not final and seg_id in self._finished_ids
                if final:
                    self._finished_ids.discard(seg_id)
            if final or (lines and not stale):
                self.on_lines(seg_id, lines, final)
