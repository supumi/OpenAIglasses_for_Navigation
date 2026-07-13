# audio_stream.py
# -*- coding: utf-8 -*-
import asyncio
from dataclasses import dataclass
from typing import Optional, Set, List, Tuple, Any, Dict
from fastapi import Request
from fastapi.responses import StreamingResponse

# ===== 下行 WAV 流基础参数 =====
STREAM_SR = 8000  # 改为8kHz，ESP32支持
STREAM_CH = 1
STREAM_SW = 2
BYTES_PER_20MS_16K = STREAM_SR * STREAM_SW * 20 // 1000  # 320B (8kHz)

# ===== AI 播放任务总闸 =====
current_ai_task: Optional[asyncio.Task] = None

async def cancel_current_ai():
    """取消当前大模型语音任务，并等待其退出。"""
    global current_ai_task
    task = current_ai_task
    current_ai_task = None
    if task and not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

def is_playing_now() -> bool:
    t = current_ai_task
    return (t is not None) and (not t.done())

# ===== /stream.wav 连接管理 =====
@dataclass(frozen=True)
class StreamClient:
    q: asyncio.Queue
    abort_event: asyncio.Event

stream_clients: "Set[StreamClient]" = set()
STREAM_QUEUE_MAX = 96  # 小缓冲，避免积压

def _wav_header_unknown_size(sr=16000, ch=1, sw=2) -> bytes:
    import struct
    byte_rate = sr * ch * sw
    block_align = ch * sw
    data_size = 0x7FFFFFF0
    riff_size = 36 + data_size
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", riff_size, b"WAVE",
        b"fmt ", 16,
        1, ch, sr, byte_rate, block_align, sw * 8,
        b"data", data_size
    )

async def hard_reset_audio(reason: str = ""):
    """
    **一键清场**：取消当前AI任务 + 清空各 /stream.wav 客户端队列里的积压旧音频，
    但不断开连接，保证 ESP32 扩音器持续在线接住新音频。
    """
    # 1) 清空各客户端队列里的积压旧音频（保持连接，让新音频从头开始）
    for sc in list(stream_clients):
        try:
            while not sc.q.empty():
                sc.q.get_nowait()
        except Exception:
            pass

    # 2) 取消当前AI任务
    await cancel_current_ai()

    # 3) 日志
    if reason:
        print(f"[HARD-RESET] {reason}")

async def broadcast_pcm16_realtime(pcm16: bytes):
    """以 20ms 节拍把 pcm16 发送给所有仍存活的连接；队列满丢尾，保持实时。"""
    # 【新增】录制音频（在分发之前整体录制，避免分片）
    try:
        import sync_recorder
        sync_recorder.record_audio(pcm16, text="[Omni对话]")
    except Exception:
        pass  # 静默失败，不影响播放
    
    loop = asyncio.get_event_loop()
    next_tick = loop.time()
    off = 0
    n_clients = len(stream_clients)
    if n_clients == 0:
        print(f"[STREAM] 警告：broadcast 时无 /stream.wav 客户端，音频丢失 (pcm16={len(pcm16)}B)", flush=True)
    else:
        print(f"[STREAM] broadcast 开始 clients={n_clients} pcm16={len(pcm16)}B", flush=True)
    while off < len(pcm16):
        take = min(BYTES_PER_20MS_16K, len(pcm16) - off)
        piece = pcm16[off:off + take]

        dead: List[StreamClient] = []
        for sc in list(stream_clients):
            if sc.abort_event.is_set():
                dead.append(sc)
                continue
            try:
                if sc.q.full():
                    try: sc.q.get_nowait()
                    except Exception: pass
                sc.q.put_nowait(piece)
            except Exception:
                dead.append(sc)
        for sc in dead:
            try: stream_clients.discard(sc)
            except Exception: pass

        next_tick += 0.020
        now = loop.time()
        if now < next_tick:
            await asyncio.sleep(next_tick - now)
        else:
            next_tick = now
        off += take

# ===== FastAPI 路由注册器 =====
def register_stream_route(app):
    @app.get("/stream.wav")
    async def stream_wav(_: Request):
        # —— 多连接共存：新连接加入，旧连接保留（ESP32 重连时不断旧的，平滑过渡）——
        q: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
        abort_event = asyncio.Event()
        sc = StreamClient(q=q, abort_event=abort_event)
        stream_clients.add(sc)
        print(f"[STREAM] /stream.wav 新连接建立，当前客户端数={len(stream_clients)}", flush=True)

        async def gen():
            yield _wav_header_unknown_size(STREAM_SR, STREAM_CH, STREAM_SW)
            try:
                while True:
                    if abort_event.is_set():
                        break
                    try:
                        chunk = await asyncio.wait_for(q.get(), timeout=0.5)
                    except asyncio.TimeoutError:
                        continue
                    if abort_event.is_set():
                        break
                    if chunk is None:
                        break
                    if chunk:
                        yield chunk
            finally:
                stream_clients.discard(sc)
                print(f"[STREAM] /stream.wav 连接结束，剩余客户端数={len(stream_clients)}", flush=True)
        return StreamingResponse(gen(), media_type="audio/wav")