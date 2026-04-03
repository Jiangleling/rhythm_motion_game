"""
主程序入口与流程控制器。

职责:
1. 启动 PyQt5 应用并装配四个页面
2. 使用 QMediaPlayer 驱动示范视频播放与时间轴同步
3. 使用 QThread 在后台执行摄像头采集与姿态检测
4. 在关键得分帧完成动作评分并生成结算结果
"""

from __future__ import annotations

import sys
import time
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

import cv2

from config import COLORS, LEVEL_INFO_FALLBACK, PATHS, RUNTIME, SCORE_RULES, ensure_runtime_directories
from pose_utils import (
    LevelBundle,
    PoseDetector,
    PoseSmoother,
    classify_score,
    combo_multiplier,
    cosine_similarity_score,
    draw_pose_overlay,
    infer_correction_hint,
    load_level_bundle,
)
from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QKeyEvent, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from video_player import CvVideoPlayer
from ui_pages import AnimatedStackedWidget, CountdownOverlay, GamePage, LaunchPage, PreparePage, ResultPage


class PoseCaptureThread(QThread):
    """摄像头采集与姿态检测线程，避免阻塞 UI 主线程。"""

    frame_ready = pyqtSignal(object)
    fatal_error = pyqtSignal(str)

    def __init__(self, camera_id: int = 0, parent: Optional[QMainWindow] = None) -> None:
        super().__init__(parent)
        self.camera_id = camera_id
        self._running = False

    def stop(self) -> None:
        """请求线程结束。"""

        self._running = False
        self.wait(1200)

    def run(self) -> None:
        """持续采集摄像头画面并发送给主线程。"""

        self._running = True
        detector = PoseDetector(
            static_image_mode=False,
            model_complexity=RUNTIME.pose_model_complexity,
            min_detection_confidence=RUNTIME.detection_confidence,
            min_tracking_confidence=RUNTIME.tracking_confidence,
        )
        smoother = PoseSmoother(alpha=RUNTIME.pose_smoothing_alpha)

        backend = getattr(cv2, "CAP_DSHOW", 0)
        capture = cv2.VideoCapture(self.camera_id, backend) if backend else cv2.VideoCapture(self.camera_id)
        if not capture.isOpened():
            capture = cv2.VideoCapture(self.camera_id)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, RUNTIME.camera_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, RUNTIME.camera_height)

        if not capture.isOpened():
            detector.close()
            self.fatal_error.emit("摄像头打不开，请检查摄像头权限、占用情况或设备连接。")
            return

        failure_count = 0
        try:
            while self._running:
                success, frame = capture.read()
                capture_time = time.perf_counter()
                if not success or frame is None:
                    failure_count += 1
                    if failure_count >= RUNTIME.camera_retry_limit:
                        self.fatal_error.emit("摄像头读取失败，已停止本次挑战。")
                        break
                    self.msleep(30)
                    continue

                failure_count = 0
                frame = cv2.flip(frame, 1)
                detection = detector.detect(frame)
                sample_vector = None
                if detection.visible and detection.landmarks:
                    draw_pose_overlay(frame, detection.landmarks)
                    if detection.normalized_vector is not None:
                        sample_vector = smoother.update(detection.normalized_vector)
                    else:
                        smoother.reset()
                else:
                    smoother.reset()
                    cv2.putText(
                        frame,
                        "未检测到完整人体，请保持全身入镜",
                        (28, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (248, 114, 114),
                        2,
                        cv2.LINE_AA,
                    )

                image = self._to_qimage(frame)
                self.frame_ready.emit(
                    {
                        "capture_time": capture_time,
                        "image": image,
                        "visible": bool(detection.visible),
                        "vector": sample_vector,
                        "core_visibility": float(detection.core_visibility),
                        "missing": list(detection.missing_core_joints),
                    }
                )
                self.msleep(10)
        finally:
            capture.release()
            detector.close()

    @staticmethod
    def _to_qimage(frame_bgr: Any) -> QImage:
        """将 OpenCV BGR 图像转换为 QImage。"""

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_frame.shape
        bytes_per_line = channel * width
        return QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).copy()


def build_level_view(bundle: Optional[LevelBundle], warning_text: str = "") -> Dict[str, Any]:
    """将关卡对象转换为页面渲染需要的字典。"""

    if bundle is None:
        return {
            "action_name": LEVEL_INFO_FALLBACK["action_name"],
            "subtitle": LEVEL_INFO_FALLBACK["subtitle"],
            "difficulty": LEVEL_INFO_FALLBACK["difficulty"],
            "coach_name": LEVEL_INFO_FALLBACK["coach_name"],
            "notes": LEVEL_INFO_FALLBACK["notes"],
            "video_duration_ms": 0,
            "thumbnail_path": str(PATHS.thumbnail_path),
            "keyframes": [],
            "warning_text": warning_text,
        }

    return {
        "action_name": bundle.action_name,
        "subtitle": bundle.subtitle,
        "difficulty": bundle.difficulty,
        "coach_name": bundle.coach_name,
        "notes": bundle.notes,
        "video_duration_ms": bundle.video_duration_ms,
        "thumbnail_path": str(bundle.thumbnail_path),
        "keyframes": [
            {
                "timestamp_ms": item.timestamp_ms,
                "frame_index": item.frame_index,
                "label": item.label,
                "action": item.action,
            }
            for item in bundle.keyframes
        ],
        "warning_text": warning_text,
    }


def build_rating(final_score: int, pass_rate: float) -> Tuple[str, str]:
    """根据最终成绩与达标率生成评级。"""

    if pass_rate >= 0.9 and final_score > 0:
        return "S 评级", COLORS.success
    if pass_rate >= 0.75:
        return "A 评级", COLORS.primary
    if pass_rate >= 0.55:
        return "B 评级", COLORS.warning
    return "C 评级", COLORS.error


class MainWindow(QMainWindow):
    """应用主窗口，负责页面切换与关卡流程调度。"""

    def __init__(self) -> None:
        super().__init__()
        ensure_runtime_directories()
        self.setWindowTitle("AI 动作匹配健康训练系统")
        self.resize(RUNTIME.window_width, RUNTIME.window_height)
        self.setMinimumSize(1366, 768)
        self.setStyleSheet("background: %s;" % COLORS.background)

        self.level_bundle: Optional[LevelBundle] = None
        self.level_warning = ""
        self.session_active = False
        self.pose_thread: Optional[PoseCaptureThread] = None
        self.pose_samples: Deque[Dict[str, Any]] = deque(maxlen=90)
        self.next_keyframe_index = 0
        self.total_score = 0
        self.current_combo = 0
        self.max_combo = 0
        self.pass_count = 0
        self.completed_actions = 0
        self.session_reason = "completed"

        self.pages = AnimatedStackedWidget(self)
        self.setCentralWidget(self.pages)

        self.launch_page = LaunchPage(self)
        self.prepare_page = PreparePage(self)
        self.game_page = GamePage(self)
        self.result_page = ResultPage(self)

        self.pages.addWidget(self.launch_page)
        self.pages.addWidget(self.prepare_page)
        self.pages.addWidget(self.game_page)
        self.pages.addWidget(self.result_page)

        self.countdown_overlay = CountdownOverlay(self.pages)
        self.countdown_overlay.finished.connect(self._begin_session)

        self.media_player = CvVideoPlayer(self)
        self.media_player.frameReady.connect(self._on_video_frame)
        self.media_player.setNotifyInterval(30)
        self.media_player.positionChanged.connect(self._on_media_position_changed)
        self.media_player.statusChanged.connect(self._on_media_status_changed)
        self.media_player.error.connect(self._on_media_error)

        self.launch_page.start_clicked.connect(self._show_prepare_page)
        self.prepare_page.start_clicked.connect(self._start_countdown)
        self.result_page.restart_clicked.connect(self._return_to_launch)

        self._load_level_bundle()
        self._refresh_pages()

    def _load_level_bundle(self) -> None:
        """读取预处理产出的模板与关键帧配置。"""

        try:
            self.level_bundle = load_level_bundle()
            video_exists = self.level_bundle.video_path.exists()
            if not video_exists:
                self.level_warning = "已找到模板数据，但教学视频路径不存在，请检查视频文件是否仍在项目目录中。"
                self.level_bundle = None
            else:
                self.level_warning = ""
        except Exception as exc:
            self.level_bundle = None
            self.level_warning = "未找到可用模板数据，请先运行 preprocess_video.py 生成标准模板。\n{}".format(exc)

    def _refresh_pages(self) -> None:
        """根据关卡数据刷新启动页和准备页。"""

        if self.level_bundle is None:
            self.launch_page.set_status(self.level_warning, warning=True)
        else:
            self.launch_page.set_status("模板数据已就绪，可以开始挑战。", warning=False)

        view_model = build_level_view(self.level_bundle, self.level_warning)
        self.prepare_page.set_level_info(view_model, ready=self.level_bundle is not None, warning_text=self.level_warning)
        self.game_page.reset_session(view_model)

    def _reset_session_state(self) -> None:
        """开始新一轮挑战前清空会话状态。"""

        self.pose_samples.clear()
        self.next_keyframe_index = 0
        self.total_score = 0
        self.current_combo = 0
        self.max_combo = 0
        self.pass_count = 0
        self.completed_actions = 0
        self.session_reason = "completed"
        if self.level_bundle is not None:
            self.game_page.reset_session(build_level_view(self.level_bundle))

    def _show_prepare_page(self) -> None:
        """启动页进入准备页。"""

        self._load_level_bundle()
        self._refresh_pages()
        self.pages.setCurrentIndexAnimated(1)

    def _start_countdown(self) -> None:
        """准备页点击后触发三秒倒计时。"""

        if self.level_bundle is None:
            QMessageBox.warning(self, "模板缺失", self.level_warning or "请先完成视频预处理。")
            return
        self._reset_session_state()
        self.pages.setCurrentIndexAnimated(2)
        QTimer.singleShot(280, lambda: self.countdown_overlay.start(RUNTIME.countdown_seconds))

    def _begin_session(self) -> None:
        """倒计时结束后正式开始关卡。"""

        if self.level_bundle is None:
            return

        self.session_active = True
        self.game_page.show_status("正在连接摄像头并加载示范视频...", level="info")
        self._start_pose_thread()

        self.media_player.stop()
        audio_path = str(self.level_bundle.audio_path) if self.level_bundle.audio_path else None
        self.media_player.setMedia(str(self.level_bundle.video_path), audio_path)
        self.media_player.setPosition(0)
        QTimer.singleShot(120, self.media_player.play)

    def _start_pose_thread(self) -> None:
        """启动摄像头线程。"""

        self._stop_pose_thread()
        self.pose_thread = PoseCaptureThread(RUNTIME.camera_id, self)
        self.pose_thread.frame_ready.connect(self._on_pose_frame_ready)
        self.pose_thread.fatal_error.connect(self._on_pose_fatal_error)
        self.pose_thread.start()

    def _stop_pose_thread(self) -> None:
        """停止摄像头线程。"""

        if self.pose_thread is None:
            return
        self.pose_thread.frame_ready.disconnect()
        self.pose_thread.fatal_error.disconnect()
        self.pose_thread.stop()
        self.pose_thread = None

    def _return_to_launch(self) -> None:
        """结果页回到启动页。"""

        self._stop_pose_thread()
        self.media_player.stop()

        self.session_active = False
        self._load_level_bundle()
        self._refresh_pages()
        self.pages.setCurrentIndexAnimated(0)

    def _abort_to_prepare(self, message: str) -> None:
        """在无法开始关卡时回退到准备页并提示原因。"""

        self.session_active = False
        self.media_player.stop()

        self._stop_pose_thread()
        self.level_warning = message
        self._refresh_pages()
        self.pages.setCurrentIndexAnimated(1)
        QMessageBox.warning(self, "挑战中断", message)

    def _on_pose_frame_ready(self, payload: Dict[str, Any]) -> None:
        """接收摄像头线程的最新结果并刷新右侧画面。"""

        image = payload.get("image")
        if isinstance(image, QImage):
            self.game_page.set_camera_image(QPixmap.fromImage(image))
        self.pose_samples.append(payload)

    def _on_pose_fatal_error(self, message: str) -> None:
        """摄像头线程发生致命错误时的兜底处理。"""

        if self.completed_actions == 0:
            self._abort_to_prepare(message)
            return
        self.game_page.show_status(message, level="fail")
        self._finish_session(reason="camera_error")

    def _on_video_frame(self, pixmap: QPixmap) -> None:
        """将视频帧缩放后显示在 QLabel 上。"""

        label = self.game_page.video_label
        scaled = pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        label.setPixmap(scaled)

    def _on_media_position_changed(self, position_ms: int) -> None:
        """视频播放时间推进时，驱动关键帧评分。"""

        if not self.session_active or self.level_bundle is None:
            return
        self.game_page.set_progress(position_ms)
        while self.next_keyframe_index < len(self.level_bundle.keyframes):
            keyframe = self.level_bundle.keyframes[self.next_keyframe_index]
            if position_ms + SCORE_RULES.score_trigger_lookahead_ms < keyframe.timestamp_ms:
                break
            self._score_keyframe(self.next_keyframe_index, keyframe)
            self.next_keyframe_index += 1

    def _find_best_pose_sample(self, target_time: float) -> Tuple[Optional[Dict[str, Any]], int]:
        """从最近样本中找到最接近当前评分时刻的姿态结果。"""

        if not self.pose_samples:
            return None, 10**9
        best_sample = None
        best_delta_ms = 10**9
        for sample in self.pose_samples:
            delta_ms = int(abs(float(sample["capture_time"]) - target_time) * 1000.0)
            if delta_ms < best_delta_ms:
                best_delta_ms = delta_ms
                best_sample = sample
        return best_sample, best_delta_ms

    def _score_keyframe(self, keyframe_index: int, keyframe: Any) -> None:
        """在关键得分帧上执行姿态匹配评分。"""

        self.game_page.highlight_marker(keyframe_index)
        now_time = time.perf_counter()
        sample, delay_ms = self._find_best_pose_sample(now_time)

        grade = "fail"
        current_multiplier = 1.0
        points = 0
        similarity_score = 0.0
        title_text = "未完成动作"
        detail_text = "未检测到有效人体姿态，请保持全身入镜"

        if sample is None or delay_ms > SCORE_RULES.max_sample_delay_ms:
            self.current_combo = 0
            detail_text = "采样延迟过大，请保持镜头稳定并继续跟练"
        elif not sample.get("visible", False):
            self.current_combo = 0
            detail_text = "未检测到完整人体，请退后并保持全身入镜"
        elif sample.get("vector") is None:
            self.current_combo = 0
            detail_text = "关键点识别失败，请稳定站位后继续"
        elif float(sample.get("core_visibility", 0.0)) < SCORE_RULES.min_core_visibility:
            self.current_combo = 0
            detail_text = "关键点可见性不足，请避免遮挡并保证动作完整"
        else:
            similarity_score, _ = cosine_similarity_score(sample["vector"], keyframe.template_vector)
            grade = classify_score(similarity_score, keyframe.pass_threshold, keyframe.warn_threshold)
            if grade == "pass":
                self.current_combo += 1
                self.pass_count += 1
                current_multiplier = combo_multiplier(self.current_combo)
                points = int(round(similarity_score * current_multiplier))
                self.total_score += points
                self.max_combo = max(self.max_combo, self.current_combo)
                title_text = "非常棒！+{}分".format(points)
                detail_text = "关键帧匹配成功，相似度 {:.0f} 分".format(similarity_score)
                if self.current_combo >= SCORE_RULES.combo_trigger_count:
                    self.game_page.show_combo("连击 x{}  倍率 x{:.1f}".format(self.current_combo, current_multiplier))
            elif grade == "warn":
                self.current_combo = 0
                title_text = "要加油"
                detail_text = infer_correction_hint(
                    sample["vector"],
                    keyframe.template_vector,
                    keyframe.action,
                    keyframe.correction_hint,
                )
            else:
                self.current_combo = 0
                title_text = "再努力下哦"
                detail_text = infer_correction_hint(
                    sample["vector"],
                    keyframe.template_vector,
                    keyframe.action,
                    keyframe.correction_hint,
                )

        self.completed_actions += 1
        self.game_page.set_score_and_combo(self.total_score, self.current_combo, current_multiplier)
        self.game_page.show_feedback(grade, title_text, detail_text)
        self.game_page.show_status(
            "已完成 {}/{} 个关键动作".format(self.completed_actions, len(self.level_bundle.keyframes)),
            level=grade,
        )

    def _build_result_summary(self, reason: str) -> Dict[str, Any]:
        """整理结算页所需数据。"""

        total_keyframes = len(self.level_bundle.keyframes) if self.level_bundle is not None else 0
        pass_rate = float(self.pass_count) / float(total_keyframes) if total_keyframes else 0.0
        rating_label, rating_color = build_rating(self.total_score, pass_rate)

        if reason == "manual_exit":
            status_text = "你已提前结束，本次成绩按已完成关键帧结算。"
        elif reason == "camera_error":
            status_text = "摄像头中断，本次成绩按已完成关键帧结算。"
        else:
            status_text = "视频播放完毕，本次成绩如下。"

        return {
            "status_text": status_text,
            "final_score": int(self.total_score),
            "pass_rate": "{:.0f}%".format(pass_rate * 100.0),
            "max_combo": int(self.max_combo),
            "completed_actions": int(self.completed_actions),
            "rating_label": rating_label,
            "rating_color": rating_color,
        }

    def _finish_session(self, reason: str = "completed") -> None:
        """结束本次关卡并进入结算页。"""

        if not self.session_active:
            return
        self.session_active = False
        self.session_reason = reason
        self.media_player.stop()

        self._stop_pose_thread()
        self.game_page.highlight_marker(-1)
        summary = self._build_result_summary(reason)
        self.result_page.show_result(summary)
        self.pages.setCurrentIndexAnimated(3)

    def _on_media_status_changed(self, status: int) -> None:
        """监听视频播放状态，结束后自动结算。"""

        if status == CvVideoPlayer.EndOfMedia and self.session_active:
            self._finish_session(reason="completed")

    def _on_media_error(self, error_code: Any) -> None:
        """示范视频播放失败时给出友好提示。"""

        if error_code == CvVideoPlayer.NoError:
            return
        message = self.media_player.errorString() or "教学视频播放失败，请检查视频编码或文件路径。"
        if self.session_active and self.completed_actions == 0:
            self._abort_to_prepare(message)
            return
        if self.session_active:
            self.game_page.show_status(message, level="fail")
            self._finish_session(reason="completed")
        else:
            self.level_warning = message
            self._refresh_pages()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """支持 ESC 提前退出关卡。"""

        if event.key() == Qt.Key_Escape and self.session_active:
            self._finish_session(reason="manual_exit")
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: Any) -> None:
        """窗口关闭时释放媒体播放器和摄像头线程。"""

        self.session_active = False
        self.media_player.stop()

        self._stop_pose_thread()
        super().closeEvent(event)


def main() -> None:
    """程序主入口。"""

    app = QApplication(sys.argv)
    app.setApplicationName("AI 动作匹配健康训练系统")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
