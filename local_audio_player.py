# local_audio_player.py
# 本地扬声器播放 - 使用 PyAudio 在系统扬声器上播放 Omni AI 音频
# 【修改】PyAudio 输出流改为按需打开 + 空闲自动关闭，避免持续 underrun 造成的刺耳杂音
# 【修改2】默认禁用本地扬声器输出（音频改由 ESP32 扩音器播放）；保留线程骨架以兼容现有调用
import os
import pyaudio
import threading
import queue
import time

# 本地扬声器总开关：false=完全不发声（音频只走 /stream.wav → ESP32 扩音器）
_LOCAL_SPEAKER_ENABLED = os.getenv("USE_LOCAL_SPEAKER", "false").lower() == "true"

class LocalAudioPlayer:
    def __init__(self, rate=24000, preload_chunks=4):
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._lock = threading.Lock()
        self._rate = rate
        self._preload_chunks = preload_chunks
        self._buffer = queue.Queue(maxsize=200)
        self._play_thread = None
        self._playing = False
        self._stop_event = threading.Event()
        self._first_chunk_time = None
        self._bytes_per_chunk = rate * 2 * 20 // 1000
        # 空闲多久后关闭输出流（秒）。低于该阈值保持流打开，避免频繁开关导致爆音。
        self._idle_close_sec = 0.3
        # 输出流的缓冲帧数（按帧/样本数计）。调大可降低 underrun 噪声。
        self._frames_per_buffer = max(1024, int(self._rate * 0.04))  # ≈40ms

    def _open_stream(self):
        """在持有 self._lock 时调用：打开 PyAudio 输出流"""
        if self._stream and self._stream.is_active():
            return
        try:
            self._stream = self._pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self._rate,
                output=True,
                frames_per_buffer=self._frames_per_buffer,
            )
            # 开流后立即写少量静音做预热，避免首帧爆音
            try:
                self._stream.write(b"\x00\x00" * self._frames_per_buffer, exception_on_underflow=False)
            except Exception:
                pass
            print(f"[LOCAL_AUDIO] 输出流已打开 rate={self._rate} fpb={self._frames_per_buffer}", flush=True)
        except Exception as e:
            print(f"[LOCAL_AUDIO] 打开输出流失败: {e}", flush=True)
            self._stream = None

    def _close_stream(self):
        """在持有 self._lock 时调用：关闭输出流"""
        if self._stream:
            try:
                self._stream.stop_stream()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            print("[LOCAL_AUDIO] 输出流已关闭", flush=True)

    def start(self):
        """启动后台播放线程，但暂不打开 PyAudio 输出流（等首帧到达再打开）。
        当本地扬声器被禁用时，仍启动线程以兼容调用，但永远不会打开输出流。"""
        with self._lock:
            self._stop_event.clear()
            self._playing = False
            self._first_chunk_time = None
            if self._play_thread is None or not self._play_thread.is_alive():
                self._play_thread = threading.Thread(target=self._play_loop, daemon=True)
                self._play_thread.start()

    def _play_loop(self):
        last_write_ts = 0.0
        while not self._stop_event.is_set():
            try:
                chunk = self._buffer.get(timeout=0.1)
            except queue.Empty:
                # 空闲：若输出流仍开着，超过阈值则关闭，避免 underrun 杂音
                now = time.monotonic()
                with self._lock:
                    if self._stream and self._stream.is_active() and now - last_write_ts > self._idle_close_sec:
                        self._close_stream()
                    self._playing = False
                continue

            if chunk is None:
                break

            # 本地扬声器被禁用：丢弃所有音频，仅消费队列保持管道不积压
            if not _LOCAL_SPEAKER_ENABLED:
                continue

            # 首帧到达：打开输出流
            with self._lock:
                if self._stream is None or not self._stream.is_active():
                    self._open_stream()
                if not self._playing:
                    self._playing = True

            try:
                with self._lock:
                    if self._stream and self._stream.is_active():
                        self._stream.write(chunk, exception_on_underflow=False)
                        last_write_ts = time.monotonic()
                    else:
                        last_write_ts = time.monotonic()
            except Exception as e:
                print(f"[LOCAL_AUDIO] write 失败: {e}")
                with self._lock:
                    self._close_stream()

        # 线程退出：关闭流
        with self._lock:
            self._close_stream()
            self._playing = False

    def write(self, pcm_data: bytes):
        if self._stop_event.is_set():
            return
        if not pcm_data:
            return
        try:
            self._buffer.put_nowait(pcm_data)
        except queue.Full:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                pass
            try:
                self._buffer.put_nowait(pcm_data)
            except queue.Full:
                pass

    def stop(self):
        self._stop_event.set()
        try:
            self._buffer.put_nowait(None)
        except queue.Full:
            pass
        with self._lock:
            self._close_stream()
        if self._play_thread and self._play_thread.is_alive():
            self._play_thread.join(timeout=1.0)
        self._play_thread = None
        self._playing = False
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

    def clear(self):
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break
        self._playing = False
        self._first_chunk_time = None
        # 立刻关闭输出流，停止任何残留输出
        with self._lock:
            self._close_stream()

    def __del__(self):
        self.stop()
        try:
            self._pa.terminate()
        except Exception:
            pass

local_player = LocalAudioPlayer(rate=24000)