"""画面最前面に表示する半透明の字幕ウィンドウ。

ドラッグ: 移動 / ホイール: 文字サイズ / 右クリック: 終了
"""
import time
import tkinter as tk
import tkinter.font as tkfont

NAME_COLORS = ["#7fd1ff", "#ffb86c", "#a6e22e", "#ff79c6", "#f1fa8c", "#bd93f9", "#8be9fd", "#ff5555"]


class Overlay:
    def __init__(self, root: tk.Tk, max_lines: int, line_ttl: float, font_size: int, opacity: float):
        self.root = root
        self.max_lines = max_lines
        self.line_ttl = line_ttl
        # 発話ID -> [更新時刻, 確定か, [(userId, 名前, 本文), ...]]（挿入順 = 表示順）
        self.entries: dict[int, list] = {}
        self.colors: dict[str, str] = {}

        root.title("Discord 字幕")
        root.overrideredirect(True)
        root.attributes("-topmost", True)
        root.attributes("-alpha", opacity)
        root.configure(bg="#000000")
        w = int(root.winfo_screenwidth() * 0.6)
        x = (root.winfo_screenwidth() - w) // 2
        y = root.winfo_screenheight() - 330
        root.geometry(f"{w}x260+{x}+{y}")

        self.font = tkfont.Font(family="Yu Gothic UI", size=font_size, weight="bold")
        self.text = tk.Text(root, bg="#000000", fg="#ffffff", font=self.font, bd=0, highlightthickness=0,
                            wrap="char", padx=14, pady=8, cursor="fleur", state="disabled")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("partial", foreground="#aaaaaa")
        self.status = tk.Label(root, text="起動中…", bg="#000000", fg="#999999",
                               font=("Yu Gothic UI", 10), anchor="w", padx=14)
        self.status.pack(fill="x")

        for widget in (self.text, self.status):
            widget.bind("<ButtonPress-1>", self._drag_start)
            widget.bind("<B1-Motion>", self._drag_move)
            widget.bind("<Button-3>", lambda e: root.destroy())
            widget.bind("<MouseWheel>", self._zoom)
        self._tick()

    def _drag_start(self, e):
        self._dx, self._dy = e.x_root - self.root.winfo_x(), e.y_root - self.root.winfo_y()
        return "break"

    def _drag_move(self, e):
        self.root.geometry(f"+{e.x_root - self._dx}+{e.y_root - self._dy}")
        return "break"

    def _zoom(self, e):
        size = self.font.cget("size")
        self.font.configure(size=max(10, min(60, size + (2 if e.delta > 0 else -2))))

    # ---- 他スレッドから呼んでよい ----
    def set_status(self, msg: str):
        self.root.after(0, lambda: self.status.configure(text=msg))

    def set_segment(self, seg_id: int, lines: list[tuple[str, str, str]], final: bool):
        """発話 seg_id の表示を lines [(userId, 名前, 本文)] に置き換える。final=False は仮（灰色）。"""
        def _set():
            if lines:
                entry = self.entries.setdefault(seg_id, [0.0, False, []])
                entry[:] = [time.time(), final, lines]
            else:
                self.entries.pop(seg_id, None)
            self._render()
        self.root.after(0, _set)

    # ---- 描画 ----
    def _tag(self, user_id: str) -> str:
        if user_id not in self.colors:
            self.colors[user_id] = NAME_COLORS[len(self.colors) % len(NAME_COLORS)]
            self.text.tag_configure(f"u{user_id}", foreground=self.colors[user_id])
        return f"u{user_id}"

    def _render(self):
        now = time.time()
        for seg_id in [k for k, (t, _, _) in self.entries.items() if now - t > self.line_ttl]:
            del self.entries[seg_id]
        rows = [(uid, name, body, final) for _, final, lines in self.entries.values() for uid, name, body in lines]
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for i, (uid, name, body, final) in enumerate(rows[-self.max_lines:]):
            if i:
                self.text.insert("end", "\n")
            self.text.insert("end", f"{name}：", self._tag(uid))
            self.text.insert("end", body, () if final else ("partial",))
        self.text.see("end")
        self.text.configure(state="disabled")

    def _tick(self):
        self._render()
        self.root.after(1000, self._tick)
