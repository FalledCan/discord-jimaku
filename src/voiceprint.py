"""声紋（話者埋め込み）で、声が重なった部分を誰の声か判定する。

1人だけが話していた発話から各ユーザーの声紋を自動で学習し、speakers.json に保存する。
"""
import json
import os
import urllib.request

import numpy as np

MODEL_FILE = "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx"
MODEL_URL = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/{MODEL_FILE}"
RATE = 16000
MIN_ENROLL_SEC = 1.5    # 学習に使う最短の長さ
MIN_MATCH_SEC = 0.8     # 判定に使う最短の長さ（短ければ前後を広げる）
MAX_SAMPLES = 30        # 声紋の平均に使う発話数の上限（それ以降はゆっくり更新）


class VoicePrints:
    def __init__(self, data_dir: str, on_status=print):
        self.model_path = os.path.join(data_dir, "models", MODEL_FILE)
        self.profile_path = os.path.join(data_dir, "speakers.json")
        self.on_status = on_status
        self.extractor = None
        self.profiles: dict[str, dict] = {}  # user_id -> {"emb": [...], "n": int}

    def load(self):
        try:
            if not os.path.exists(self.model_path):
                self.on_status("声紋モデルをダウンロード中…")
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                tmp = self.model_path + ".part"
                urllib.request.urlretrieve(MODEL_URL, tmp)
                os.replace(tmp, self.model_path)
            import sherpa_onnx

            config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=self.model_path, num_threads=2)
            self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        except Exception as e:
            print(f"[voiceprint] 声紋モデルを使えません（重なった発話は時刻だけで判定します）: {e}")
            return
        if os.path.exists(self.profile_path):
            with open(self.profile_path, encoding="utf-8") as f:
                self.profiles = json.load(f)
        print(f"[voiceprint] 準備完了（登録済み {len(self.profiles)} 人）")

    def _embed(self, audio: np.ndarray) -> np.ndarray | None:
        if self.extractor is None:
            return None
        stream = self.extractor.create_stream()
        stream.accept_waveform(sample_rate=RATE, waveform=audio)
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        emb = np.asarray(self.extractor.compute(stream), dtype=np.float32)
        return emb / (np.linalg.norm(emb) + 1e-9)

    def enroll(self, user_id: str, audio: np.ndarray):
        """1人だけが話していた音声で声紋を更新する。"""
        if len(audio) < MIN_ENROLL_SEC * RATE:
            return
        emb = self._embed(audio)
        if emb is None:
            return
        p = self.profiles.get(user_id)
        if p is None:
            self.profiles[user_id] = {"emb": emb.tolist(), "n": 1}
        else:
            n = min(p["n"], MAX_SAMPLES)
            mean = (np.asarray(p["emb"]) * n + emb) / (n + 1)
            p["emb"] = (mean / (np.linalg.norm(mean) + 1e-9)).tolist()
            p["n"] = p["n"] + 1
        try:
            with open(self.profile_path, "w", encoding="utf-8") as f:
                json.dump(self.profiles, f)
        except OSError:
            pass

    def best_match(self, candidates, audio: np.ndarray, start: float, end: float) -> str | None:
        """audio[start:end]（秒）が候補のうち誰の声に一番近いか。判定できなければ None。"""
        known = [u for u in candidates if u in self.profiles]
        if not known or self.extractor is None:
            return None
        pad = max(0.0, (MIN_MATCH_SEC - (end - start)) / 2)
        a, b = max(0, int((start - pad) * RATE)), min(len(audio), int((end + pad) * RATE))
        emb = self._embed(audio[a:b])
        if emb is None:
            return None
        scores = {u: float(np.dot(emb, self.profiles[u]["emb"])) for u in known}
        best = max(scores, key=scores.get)
        # 未登録の候補がいて、登録済みの人とも似ていなければ判断しない
        if len(known) < len(candidates) and scores[best] < 0.5:
            return None
        return best
