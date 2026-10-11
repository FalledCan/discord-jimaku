"""PC で動いている Discord アプリにローカル IPC (RPC) で接続し、
今いる VC のメンバー名と「誰が話しているか」を追跡する。
"""
import json
import os
import struct
import threading
import time
import urllib.parse
import urllib.request
import uuid

OP_HANDSHAKE, OP_FRAME, OP_CLOSE, OP_PING, OP_PONG = 0, 1, 2, 3, 4
SCOPES = ["rpc", "rpc.voice.read"]
REDIRECT_URI = "http://localhost"
TOKEN_URL = "https://discord.com/api/oauth2/token"
VOICE_EVENTS = ("SPEAKING_START", "SPEAKING_STOP", "VOICE_STATE_CREATE", "VOICE_STATE_UPDATE", "VOICE_STATE_DELETE")


class RPCError(Exception):
    pass


class DiscordVoiceTracker(threading.Thread):
    def __init__(self, client_id: str, client_secret: str, token_path: str, on_status=print):
        super().__init__(daemon=True)
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_path = token_path
        self.on_status = on_status

        self._lock = threading.Lock()
        self._names: dict[str, str] = {}
        self._speaking: set[str] = set()
        self.self_id: str | None = None
        self.channel_id: str | None = None
        self.channel_name: str | None = None

        self._pipe = None
        self._channel_changed = False

    # ---- 他スレッドから参照する API ----
    def speaking_now(self) -> set[str]:
        """自分以外で今話しているユーザー ID。"""
        with self._lock:
            return set(self._speaking) - {self.self_id}

    def name_of(self, user_id: str) -> str:
        with self._lock:
            return self._names.get(user_id, user_id)

    # ---- IPC ----
    def _connect_pipe(self):
        for i in range(10):
            try:
                self._pipe = open(rf"\\.\pipe\discord-ipc-{i}", "r+b", buffering=0)
                return
            except OSError:
                continue
        raise RPCError("Discord アプリが起動していません")

    def _send(self, op: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self._pipe.write(struct.pack("<II", op, len(data)) + data)

    def _recv(self) -> tuple[int, dict]:
        header = self._read_exact(8)
        op, length = struct.unpack("<II", header)
        return op, json.loads(self._read_exact(length).decode("utf-8"))

    def _read_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self._pipe.read(n - len(buf))
            if not chunk:
                raise RPCError("Discord との接続が切れました")
            buf += chunk
        return buf

    def _read_frame(self) -> dict:
        """1フレーム読み、イベントなら処理して返す。PING には応答する。"""
        while True:
            op, msg = self._recv()
            if op == OP_PING:
                self._send(OP_PONG, msg)
                continue
            if op == OP_CLOSE:
                raise RPCError(f"Discord が接続を閉じました: {msg.get('message', msg)}")
            if msg.get("cmd") == "DISPATCH":
                self._handle_event(msg.get("evt"), msg.get("data") or {})
            return msg

    def _request(self, cmd: str, args: dict | None = None, evt: str | None = None) -> dict:
        nonce = str(uuid.uuid4())
        payload = {"cmd": cmd, "args": args or {}, "nonce": nonce}
        if evt:
            payload["evt"] = evt
        self._send(OP_FRAME, payload)
        while True:
            msg = self._read_frame()
            if msg.get("nonce") != nonce:
                continue
            if msg.get("evt") == "ERROR":
                raise RPCError(f"{cmd} 失敗: {msg['data'].get('message')} ({msg['data'].get('code')})")
            return msg.get("data") or {}

    # ---- 認証 ----
    def _oauth(self, params: dict) -> dict:
        body = urllib.parse.urlencode({"client_id": self.client_id, "client_secret": self.client_secret, **params}).encode()
        req = urllib.request.Request(TOKEN_URL, data=body, headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "DiscordJimaku (local, 1.0)",
        })
        with urllib.request.urlopen(req, timeout=15) as r:
            token = json.loads(r.read())
        token["expires_at"] = time.time() + token.get("expires_in", 0) - 60
        with open(self.token_path, "w", encoding="utf-8") as f:
            json.dump(token, f)
        return token

    def _access_token(self) -> str:
        token = None
        if os.path.exists(self.token_path):
            with open(self.token_path, encoding="utf-8") as f:
                token = json.load(f)
            if token.get("expires_at", 0) < time.time():
                try:
                    token = self._oauth({"grant_type": "refresh_token", "refresh_token": token["refresh_token"]})
                except Exception:
                    token = None
        if token is None:
            self.on_status("Discord に表示された認証画面で「認証」を押してください")
            code = self._request("AUTHORIZE", {"client_id": self.client_id, "scopes": SCOPES})["code"]
            token = self._oauth({"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI})
        return token["access_token"]

    # ---- VC 追跡 ----
    @staticmethod
    def _display_name(state: dict) -> str:
        user = state.get("user") or {}
        return state.get("nick") or user.get("global_name") or user.get("username") or user.get("id", "?")

    def _handle_event(self, evt: str, data: dict):
        with self._lock:
            if evt == "SPEAKING_START":
                self._speaking.add(data["user_id"])
            elif evt == "SPEAKING_STOP":
                self._speaking.discard(data["user_id"])
            elif evt in ("VOICE_STATE_CREATE", "VOICE_STATE_UPDATE"):
                self._names[data["user"]["id"]] = self._display_name(data)
            elif evt == "VOICE_STATE_DELETE":
                self._speaking.discard(data["user"]["id"])
            elif evt == "VOICE_CHANNEL_SELECT":
                self._channel_changed = True

    def _follow_selected_channel(self):
        if self.channel_id:
            for e in VOICE_EVENTS:
                try:
                    self._request("UNSUBSCRIBE", {"channel_id": self.channel_id}, evt=e)
                except RPCError:
                    pass
        channel = self._request("GET_SELECTED_VOICE_CHANNEL")
        with self._lock:
            self._speaking.clear()
            self._names = {s["user"]["id"]: self._display_name(s) for s in (channel or {}).get("voice_states", [])}
            self.channel_id = (channel or {}).get("id")
            self.channel_name = (channel or {}).get("name")
        if not self.channel_id:
            self.on_status("VC に参加すると字幕が始まります")
            return
        for e in VOICE_EVENTS:
            self._request("SUBSCRIBE", {"channel_id": self.channel_id}, evt=e)
        self.on_status(f"🔊 {self.channel_name}（{len(self._names)}人）で字幕表示中")

    def _session(self):
        self._connect_pipe()
        self._send(OP_HANDSHAKE, {"v": 1, "client_id": self.client_id})
        ready = self._read_frame()
        if ready.get("evt") != "READY":
            raise RPCError(f"ハンドシェイク失敗: {ready}")

        try:
            auth = self._request("AUTHENTICATE", {"access_token": self._access_token()})
        except RPCError:
            # トークン失効など。次回は認証し直す
            if os.path.exists(self.token_path):
                os.remove(self.token_path)
            raise
        self.self_id = auth["user"]["id"]
        self._request("SUBSCRIBE", evt="VOICE_CHANNEL_SELECT")
        self._follow_selected_channel()

        while True:
            self._read_frame()
            if self._channel_changed:
                self._channel_changed = False
                self._follow_selected_channel()

    def run(self):
        while True:
            try:
                self.on_status("Discord に接続中…")
                self._session()
            except Exception as e:
                msg = str(e)
                if "invalid_client" in msg or "401" in msg:
                    msg = "Client ID / Client Secret が正しくありません (config.json を確認)"
                self.on_status(f"⚠ {msg} — 5秒後に再接続します")
            finally:
                with self._lock:
                    self._speaking.clear()
                    self.channel_id = None
                if self._pipe:
                    try:
                        self._pipe.close()
                    except OSError:
                        pass
                    self._pipe = None
            time.sleep(5)
