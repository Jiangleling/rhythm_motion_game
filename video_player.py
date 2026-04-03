"""
基于 OpenCV 的视频播放器，替代 QMediaPlayer + QVideoWidget。

在 Windows 上 PyQt5 的 QMediaPlayer 绑定 QVideoWidget 后可能因
DirectShow 后端兼容性问题无法播放 H.264 视频。本模块使用 OpenCV
在后台线程逐帧读取视频，通过信号将 QPixmap 送到主线程显示。
音频使用 sounddevice + wave 流式播放。
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import sounddevice as sd
from PyQt5.QtCore import QMutex, QObject, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap


class _ReaderThread(QThread):
    """后台线程：按视频帧率读取帧并发射信号。"""

    frame_ready = pyqtSignal(QPixmap, int)  # pixmap, position_ms
    finished_playing = pyqtSignal()

    def __init__(self, path: str, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._path = path
        self._running = True
        self._seek_ms: Optional[float] = None
        self._mutex = QMutex()

    def request_stop(self) -> None:
        self._running = False

    def seek(self, ms: int) -> None:
        self._mutex.lock()
        self._seek_ms = float(ms)
        self._mutex.unlock()

    def run(self) -> None:
        capture = cv2.VideoCapture(self._path)
        if not capture.isOpened():
            return
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        interval = 1.0 / fps

        while self._running:
            self._mutex.lock()
            if self._seek_ms is not None:
                capture.set(cv2.CAP_PROP_POS_MSEC, self._seek_ms)
                self._seek_ms = None
            self._mutex.unlock()

            t0 = time.perf_counter()
            ret, frame = capture.read()
            if not ret:
                self.finished_playing.emit()
                break

            position_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimg)
            self.frame_ready.emit(pixmap, position_ms)

            elapsed = time.perf_counter() - t0
            sleep_ms = max(1, int((interval - elapsed) * 1000))
            self.msleep(sleep_ms)

        capture.release()


class _AudioPlayer:
    """使用 sounddevice 流式播放 WAV 音频。"""

    def __init__(self) -> None:
        self._stream: Optional[sd.OutputStream] = None
        self._wf: Optional[wave.Wave_read] = None
        self._lock = threading.Lock()
        self._playing = False

    def open(self, wav_path: str) -> bool:
        self.stop()
        path = Path(wav_path)
        if not path.exists():
            return False
        try:
            self._wf = wave.open(str(path), "rb")
        except Exception:
            return False
        return True

    def play(self) -> None:
        if self._wf is None:
            return
        self._wf.rewind()
        self._playing = True
        sr = self._wf.getframerate()
        ch = self._wf.getnchannels()
        self._stream = sd.OutputStream(
            samplerate=sr,
            channels=ch,
            dtype="int16",
            blocksize=4096,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._playing = False
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._wf is not None:
            try:
                self._wf.close()
            except Exception:
                pass
            self._wf = None

    def _callback(self, outdata: np.ndarray, frames: int, time_info: object, status: object) -> None:
        if not self._playing or self._wf is None:
            outdata.fill(0)
            return
        with self._lock:
            try:
                raw = self._wf.readframes(frames)
            except Exception:
                outdata.fill(0)
                return
        if len(raw) == 0:
            outdata.fill(0)
            self._playing = False
            return
        data = np.frombuffer(raw, dtype=np.int16)
        ch = outdata.shape[1] if outdata.ndim > 1 else 1
        expected = frames * ch
        if len(data) < expected:
            padded = np.zeros(expected, dtype=np.int16)
            padded[: len(data)] = data
            data = padded
        if ch > 1:
            outdata[:] = data.reshape(-1, ch)
        else:
            outdata[:, 0] = data[:frames]


class CvVideoPlayer(QObject):
    """OpenCV 驱动的视频播放器，接口与 QMediaPlayer 兼容。"""

    # 状态常量
    NoMedia = 1
    LoadedMedia = 3
    EndOfMedia = 7
    InvalidMedia = 8

    # 错误常量
    NoError = 0
    ResourceError = 1

    positionChanged = pyqtSignal(int)
    statusChanged = pyqtSignal(int)
    error = pyqtSignal(int)
    frameReady = pyqtSignal(QPixmap)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._video_path: Optional[str] = None
        self._audio_path: Optional[str] = None
        self._fps: float = 30.0
        self._duration_ms: int = 0
        self._current_position_ms: int = 0
        self._reader: Optional[_ReaderThread] = None
        self._audio = _AudioPlayer()

    # ---- 公共接口 ----

    def setNotifyInterval(self, ms: int) -> None:
        pass

    def setMedia(self, video_path: str, audio_path: Optional[str] = None) -> None:
        """加载视频路径和可选的音频 WAV 路径。"""
        self.stop()
        self._video_path = video_path
        self._audio_path = audio_path
        self._probe(video_path)
        if audio_path:
            self._audio.open(audio_path)

    def play(self) -> None:
        if not self._video_path:
            return
        self._stop_reader()
        self._reader = _ReaderThread(self._video_path, self)
        self._reader.frame_ready.connect(self._on_frame)
        self._reader.finished_playing.connect(self._on_finished)
        self._reader.start()
        self._audio.play()

    def stop(self) -> None:
        self._stop_reader()
        self._audio.stop()
        self._current_position_ms = 0

    def setPosition(self, ms: int) -> None:
        self._current_position_ms = ms
        if self._reader is not None and self._reader.isRunning():
            self._reader.seek(ms)

    def position(self) -> int:
        return self._current_position_ms

    def duration(self) -> int:
        return self._duration_ms

    def errorString(self) -> str:
        return ""

    # ---- 内部方法 ----

    def _probe(self, path: str) -> None:
        if not path or not Path(path).exists():
            self.error.emit(self.ResourceError)
            self.statusChanged.emit(self.InvalidMedia)
            return
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            self.error.emit(self.ResourceError)
            self.statusChanged.emit(self.InvalidMedia)
            return
        self._fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self._duration_ms = int(total_frames / self._fps * 1000) if self._fps > 0 else 0
        capture.release()
        self.statusChanged.emit(self.LoadedMedia)

    def _stop_reader(self) -> None:
        if self._reader is not None:
            self._reader.request_stop()
            self._reader.wait(3000)
            self._reader = None

    def _on_frame(self, pixmap: QPixmap, position_ms: int) -> None:
        self._current_position_ms = position_ms
        self.positionChanged.emit(position_ms)
        self.frameReady.emit(pixmap)

    def _on_finished(self) -> None:
        self._audio.stop()
        self.statusChanged.emit(self.EndOfMedia)
