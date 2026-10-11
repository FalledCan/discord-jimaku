"""PC のスピーカー出力をループバック録音し、Discord の「話している人」情報で発話ごとに区切る。"""
import collections
import itertools
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio
import soxr

TARGET_RATE = 16000
BLOCK_SEC = 0.1
PREROLL_BLOCKS = 4       # 話し始めイベントより少し前の音も含める
SILENCE_BLOCKS = 7       # 0.7 秒誰も話していなければ区切る
SWITCH_BLOCKS = 2        # 話者が入れ替わったとみなすまでの空白
MAX_SEGMENT_SEC = 10     # 長い発話はここで分割して確定させる
PARTIAL_EVERY = 5        # 話している途中も 0.5 秒ごとに仮の文字起こしを依頼する
MIN_PARTIAL_SEC = 0.8

_segment_ids = itertools.count(1)


class Segment:
    """発話のまとまり。0.1 秒ブロックごとの音声と「その時話していた人」を持つ。"""

    def __init__(self):
        self.id = next(_segment_ids)
        self.blocks: list[np.ndarray] = []
        self.activity: list[frozenset] = []
        self.speakers: set[str] = set()
        self.silence = 0
        # 誰かが話している途中に別の人が話し始めた位置と、その時話していた人たち
        self.handover_at: int | None = None
        self.handover_from: frozenset = frozenset()

    def add(self, block, active: frozenset):
        self.blocks.append(block)
        self.activity.append(active)
        self.speakers |= active

    def seconds(self):
        return len(self.blocks) * BLOCK_SEC


def pick_loopback_device(pa, wanted: str = ""):
    """録音するループバックデバイスを選ぶ。

    wanted が指定されていれば名前にその文字列を含むもの、無ければ
    SteelSeries Sonar 等の「Chat」デバイス（Discord の出力先）、それも無ければ既定の出力。
    """
    devices = list(pa.get_loopback_device_info_generator())
    for key in ([wanted] if wanted else []) + ["Sonar - Chat"]:
        for d in devices:
            if key.lower() in d["name"].lower():
                return d
    return pa.get_default_wasapi_loopback()


class LoopbackSegmenter(threading.Thread):
    def __init__(self, tracker, on_segment, on_partial, on_status=print, device_name: str = ""):
        super().__init__(daemon=True)
        self.device_name = device_name
        self.tracker = tracker
        # どちらも (発話ID, float32 16kHz mono, ブロックごとの話者集合のリスト) を受け取る
        self.on_segment = on_segment  # 発話が終わったとき（確定）
        self.on_partial = on_partial  # 話している途中（仮）
        self.on_status = on_status
        self._blocks: "collections.deque[np.ndarray]" = collections.deque()
        self._cv = threading.Condition()

    def _open_stream(self, pa):
        device = pick_loopback_device(pa, self.device_name)
        rate = int(device["defaultSampleRate"])
        channels = device["maxInputChannels"]
        frames = int(rate * BLOCK_SEC)

        # ブロックの境目でも途切れないよう、ストリーム用のリサンプラーで 16kHz に変換
        # （高域を落としてから間引くので、折り返しノイズが出ない）
        resampler = soxr.ResampleStream(rate, TARGET_RATE, 1, dtype="float32", quality="HQ")
        block_len = int(TARGET_RATE * BLOCK_SEC)
        pending = [np.zeros(0, dtype=np.float32)]

        def callback(data, *_):
            pcm = np.frombuffer(data, dtype=np.float32).reshape(-1, channels).mean(axis=1)
            out = resampler.resample_chunk(np.ascontiguousarray(pcm, dtype=np.float32))
            # リサンプラーの出力長はばらつくので、ちょうど 0.1 秒ずつに切り直す
            buf = np.concatenate([pending[0], out])
            n = len(buf) // block_len
            pending[0] = buf[n * block_len:]
            if n:
                with self._cv:
                    self._blocks.extend(np.split(buf[:n * block_len], n))
                    self._cv.notify()
            return None, pyaudio.paContinue

        stream = pa.open(format=pyaudio.paFloat32, channels=channels, rate=rate, input=True,
                         input_device_index=device["index"], frames_per_buffer=frames, stream_callback=callback)
        print(f"[audio] ループバック録音: {device['name']} ({rate}Hz, {channels}ch)")
        return stream, device["index"]

    def _finish(self, seg: Segment, drop_tail: int = 0):
        n = len(seg.blocks) - drop_tail
        if seg.speakers and n > 0:
            self.on_segment(seg.id, np.concatenate(seg.blocks[:n]), seg.activity[:n])
        else:
            self.on_segment(seg.id, None, [])

    def run(self):
        pa = pyaudio.PyAudio()
        stream, dev_index = self._open_stream(pa)
        preroll: collections.deque = collections.deque(maxlen=PREROLL_BLOCKS)
        seg: Segment | None = None
        last_check = time.time()

        while True:
            with self._cv:
                while not self._blocks:
                    self._cv.wait(timeout=1.0)
                    if not self._blocks:
                        break
                block = self._blocks.popleft() if self._blocks else None

            # 出力デバイスが切り替わったら録音し直す（ヘッドセット抜き差しなど）
            if time.time() - last_check > 3:
                last_check = time.time()
                try:
                    if pick_loopback_device(pa, self.device_name)["index"] != dev_index:
                        stream.close()
                        pa.terminate()
                        pa = pyaudio.PyAudio()
                        stream, dev_index = self._open_stream(pa)
                except Exception as e:
                    self.on_status(f"⚠ 録音デバイスエラー: {e}")

            if block is None:
                continue

            active = frozenset(self.tracker.speaking_now())

            if seg is None:
                if not active:
                    preroll.append(block)
                    continue
                seg = Segment()
                for b in preroll:
                    seg.add(b, frozenset())
                preroll.clear()

            # 前の人が黙ってから別の人だけが話し始めた → 区切る
            last_talkers = seg.activity[-seg.silence - 1] if len(seg.activity) > seg.silence else frozenset()
            if active and seg.silence >= SWITCH_BLOCKS and not (active & last_talkers):
                tail = seg.blocks[-seg.silence:]
                self._finish(seg, drop_tail=seg.silence)
                seg = Segment()
                for b in tail:  # 区切り直前の空白は次の人の話し始めとして持ち越す
                    seg.add(b, frozenset())

            # 話している途中に別の人が話し始め、元の人が話し終わった → 話し始めの位置で区切る
            prev = seg.activity[-1] if seg.activity else frozenset()
            if active - prev and prev and seg.handover_at is None:
                seg.handover_at, seg.handover_from = len(seg.blocks), prev
            elif seg.handover_at is not None:
                if active and not (active & seg.handover_from):
                    cut = seg.handover_at
                    old_blocks, old_activity = seg.blocks, seg.activity
                    seg.blocks, seg.activity = old_blocks[:cut], old_activity[:cut]
                    self._finish(seg)
                    seg = Segment()
                    for b, a in zip(old_blocks[cut:], old_activity[cut:]):
                        seg.add(b, a)
                elif not (active - seg.handover_from):
                    seg.handover_at = None  # 割り込んだ人がすぐ黙った（相づちなど）

            seg.add(block, active)
            seg.silence = 0 if active else seg.silence + 1

            if seg.silence >= SILENCE_BLOCKS:
                self._finish(seg)  # 語尾が遅れて届くこともあるので末尾は切らない（無音は Whisper の VAD が除く）
                seg = None
            elif seg.seconds() >= MAX_SEGMENT_SEC and seg.silence > 0:
                # 長い発話は息継ぎのところで区切って早めに表示
                self._finish(seg)
                seg = None
            elif seg.seconds() >= MAX_SEGMENT_SEC * 1.5:
                self._finish(seg)
                seg = None
            elif seg.silence == 0 and len(seg.blocks) % PARTIAL_EVERY == 0 and seg.seconds() >= MIN_PARTIAL_SEC:
                self.on_partial(seg.id, np.concatenate(seg.blocks), list(seg.activity))
