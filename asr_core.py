# asr_core.py
# -*- coding: utf-8 -*-
import os, json, asyncio
from typing import Any, Dict, List, Optional, Callable, Tuple

# 防打断：ASR 句末 final 到达后，等待该秒数内无新 final 才真正喂给 LLM。
# 期间用户继续说话产生新 final → 计时器重置，避免半句话就触发 AI。
ASR_FINAL_DEBOUNCE_SEC = float(os.getenv("ASR_FINAL_DEBOUNCE_SEC", "0.8"))

ASR_DEBUG_RAW = os.getenv("ASR_DEBUG_RAW", "0") == "1"

def _shorten(s: str, limit: int = 200) -> str:
    if not s:
        return ""
    return s if len(s) <= limit else (s[:limit] + "…")

def _safe_to_dict(x: Any) -> Dict[str, Any]:
    if isinstance(x, dict): return x
    for attr in ("to_dict", "model_dump", "__dict__"):
        try:
            v = getattr(x, attr, None)
        except Exception:
            v = None
        if callable(v):
            try:
                d = v()
                if isinstance(d, dict): return d
            except Exception:
                pass
        elif isinstance(v, dict):
            return v
    try:
        s = str(x)
        if s and s.lstrip().startswith("{") and s.rstrip().endswith("}"):
            return json.loads(s)
    except Exception:
        pass
    return {"_raw": str(x)}

def _extract_sentence(event_obj: Any) -> Tuple[Optional[str], Optional[bool]]:
    d = _safe_to_dict(event_obj)
    cands: List[Dict[str, Any]] = [d]
    for k in ("output", "data", "result"):
        v = d.get(k)
        if isinstance(v, dict):
            cands.append(v)
    for obj in cands:
        sent = obj.get("sentence")
        if isinstance(sent, dict):
            text = sent.get("text")
            is_end = sent.get("sentence_end")
            if is_end is not None:
                is_end = bool(is_end)
            return text, is_end
    for obj in cands:
        if "text" in obj and isinstance(obj.get("text"), str):
            return obj.get("text"), None
    return None, None

# ====== 仅热词触发的“全清零复位”配置 ======
INTERRUPT_KEYWORDS = set(
    os.getenv("INTERRUPT_KEYWORDS", "停下,别说了,停止").split(",")
)

def _normalize_cn(s: str) -> str:
    try:
        import unicodedata
        s = "".join(" " if unicodedata.category(ch) == "Zs" else ch for ch in s)
        s = s.strip().lower()
    except Exception:
        s = (s or "").strip().lower()
    return s

# ============ ASR 全局总闸 ============
_current_recognition: Optional[object] = None
_rec_lock = asyncio.Lock()

async def set_current_recognition(r):
    global _current_recognition
    async with _rec_lock:
        _current_recognition = r

async def stop_current_recognition():
    global _current_recognition
    async with _rec_lock:
        r = _current_recognition
        _current_recognition = None
    if r:
        try:
            r.stop()  # DashScope SDK 的实时识别停止
        except Exception:
            pass

# ============ ASR 回调 ============
class ASRCallback:
    """
    设计目标：
    1) “停下 / 别说了 …”等热词一出现 → 立刻全清零复位（恢复到刚启动后的状态）。
    2) 除此之外【不接受打断】；AI 正在播报时，用户说话只做展示，不触发新一轮。
    3) 不再用 partial 叠加字符串；partial 只用于 UI 临时展示；只有 final sentence 用于驱动 AI。
    """

    def __init__(
        self,
        on_sdk_error: Callable[[str], None],
        post: Callable[[asyncio.Future], None],
        ui_broadcast_partial,
        ui_broadcast_final,
        is_playing_now_fn: Callable[[], bool],
        start_ai_with_text_fn,               # async (text)
        full_system_reset_fn,                 # async (reason)
        interrupt_lock: asyncio.Lock,
    ):
        self._on_sdk_error = on_sdk_error
        self._post = post
        self._last_partial_for_ui: str = ""   # 只用于 UI 展示
        self._last_final_text: str = ""       # 以句末 final 为准
        self._hot_interrupted: bool = False   # 本句是否因热词触发过复位（防抖）

        # 防打断：累积 final，等静音窗口过后再喂 LLM；新 final 到达则重置计时
        self._pending_finals: List[str] = []
        self._fire_task: Optional[asyncio.Task] = None
        self._debounce_sec = ASR_FINAL_DEBOUNCE_SEC

        self._ui_partial = ui_broadcast_partial
        self._ui_final   = ui_broadcast_final
        self._is_playing = is_playing_now_fn
        self._start_ai   = start_ai_with_text_fn
        self._full_reset = full_system_reset_fn
        self._interrupt_lock = interrupt_lock

    def _cancel_pending_fire(self):
        """取消挂起的 LLM 触发任务；保留已累积的 final 文本（新 final 会追加，旧 task 不再触发）"""
        t = self._fire_task
        self._fire_task = None
        if t is not None and not t.done():
            try:
                t.cancel()
            except Exception:
                pass

    def _reschedule_fire(self):
        """收到一个 final 后，(重新) 排定一个静音去抖任务"""
        self._cancel_pending_fire()
        try:
            self._fire_task = self._post(self._fire_after_debounce())
        except Exception:
            self._fire_task = None

    async def _fire_after_debounce(self):
        """等待 debounce_sec 内无新 final；超时则把累积文本喂给 LLM。被取消即表示新 final 已到达，不触发。"""
        try:
            await asyncio.sleep(self._debounce_sec)
        except asyncio.CancelledError:
            return
        # 取出累积文本
        texts = self._pending_finals
        self._pending_finals = []
        self._fire_task = None
        if not texts:
            return
        combined = "".join(texts).strip()
        if not combined:
            return
        # AI 正在播报时不打断（保持原有"不打断 AI"策略）：直接丢弃本次输入
        if self._is_playing():
            print(f"[ASR] AI 播报中，丢弃用户输入: {combined}", flush=True)
            return
        print(f"[LLM INPUT TEXT] {combined} (debounced)", flush=True)
        try:
            async with self._interrupt_lock:
                await self._start_ai(combined)
        except Exception as e:
            print(f"[ASR] LLM 启动失败: {e}", flush=True)

    def on_open(self):  pass
    def on_close(self): pass
    def on_complete(self): pass

    def on_error(self, err):
        try:
            self._post(self._ui_partial(""))
            self._on_sdk_error(str(err))
        except Exception:
            pass

    def on_result(self, result): self._handle(result)
    def on_event(self,  event):  self._handle(event)

    def _has_hotword(self, text: str) -> bool:
        t = _normalize_cn(text)
        if not t: return False
        for w in INTERRUPT_KEYWORDS:
            if w and _normalize_cn(w) in t:
                return True
        return False

    def _handle(self, event: Any):
        if ASR_DEBUG_RAW:
            try:
                rawd = _safe_to_dict(event)
                print("[ASR EVENT RAW]", json.dumps(rawd, ensure_ascii=False), flush=True)
            except Exception:
                pass

        text, is_end = _extract_sentence(event)
        if text is None:
            return
        text = text.strip()
        if not text:
            return

        # ---------- ① 热词优先：命中就全清零并短路，绝不送 LLM ----------
        if not self._hot_interrupted and self._has_hotword(text):
            self._hot_interrupted = True
            # 取消任何挂起的 LLM 触发，避免热词后仍喂入文本
            self._cancel_pending_fire()

            async def _hot_reset():
                async with self._interrupt_lock:
                    print(f"[ASR HOTWORD] '{text}' -> FULL RESET, skip LLM", flush=True)
                    await self._full_reset("Hotword interrupt")
            try:
                self._post(_hot_reset())
            except Exception:
                pass
            return

        # ---------- ② partial：仅用于 UI 展示 ----------
        self._last_partial_for_ui = text
        try:
            print(f"[ASR PARTIAL] len={len(text)} text='{_shorten(text)}'", flush=True)
            self._post(self._ui_partial(self._last_partial_for_ui))
        except Exception:
            pass

        # ---------- ③ final：累积 + 静音去抖后再喂 LLM（防打断） ----------
        if is_end is True:
            final_text = text
            try:
                print(f"[ASR FINAL]  len={len(final_text)} text='{final_text}'", flush=True)
                self._post(self._ui_final(final_text))
            except Exception:
                pass

            if final_text:
                # 累积本句 final，重新排定静音去抖触发任务
                self._pending_finals.append(final_text)
                self._reschedule_fire()

            # 复位进入下一句
            self._last_partial_for_ui = ""
            self._last_final_text = ""
            self._hot_interrupted = False
