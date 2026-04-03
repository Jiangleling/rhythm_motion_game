"""
PyQt5 页面与动效组件。

本文件集中定义启动页、准备页、关卡页、结算页以及通用动画控件，
主程序只负责流程编排和业务逻辑，避免 UI 与 CV 逻辑相互耦合。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from PyQt5.QtCore import (
    QEasingCurve,
    QPoint,
    QParallelAnimationGroup,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSequentialAnimationGroup,
    QSize,
    Qt,
    QTimer,
    pyqtProperty,
    pyqtSignal,
)
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRegion
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QVBoxLayout,
    QWidget,
)

from config import COLORS, FONTS
from pose_utils import ms_to_timestamp


def qcolor(color_hex: str, alpha: int = 255) -> QColor:
    """将十六进制颜色转换为 QColor。"""

    color = QColor(color_hex)
    color.setAlpha(alpha)
    return color


def build_font(pixel_size: int, bold: bool = False) -> QFont:
    """统一创建字体对象。"""

    font = QFont(FONTS.family)
    font.setStyleStrategy(QFont.PreferAntialias)
    font.setPixelSize(pixel_size)
    font.setWeight(QFont.Bold if bold else QFont.Medium)
    return font


def apply_shadow(widget: QWidget, blur_radius: int = 30, y_offset: int = 12, alpha: int = 38) -> None:
    """给控件增加轻微阴影。"""

    shadow = QGraphicsDropShadowEffect(widget)
    shadow.setBlurRadius(blur_radius)
    shadow.setOffset(0, y_offset)
    shadow.setColor(qcolor("#0F172A", alpha))
    widget.setGraphicsEffect(shadow)


class AnimatedButton(QPushButton):
    """支持 hover/press 缩放的主按钮。"""

    def __init__(self, text: str, accent_color: str = COLORS.primary, parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._scale_value = 1.0
        self._animation = QPropertyAnimation(self, b"scaleValue", self)
        self._animation.setDuration(160)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(64)
        self.setMinimumWidth(220)
        self.setFont(build_font(FONTS.button_px, bold=True))
        self.setStyleSheet(
            """
            QPushButton {
                background-color: %s;
                color: %s;
                border: none;
                border-radius: 12px;
                padding: 0 28px;
            }
            QPushButton:disabled {
                background-color: #BFC8D6;
                color: #F8F9FA;
            }
            """
            % (accent_color, COLORS.white)
        )
        apply_shadow(self, blur_radius=24, y_offset=10, alpha=45)

    def _animate_to(self, target_value: float, duration: int = 160) -> None:
        self._animation.stop()
        self._animation.setDuration(duration)
        self._animation.setStartValue(self._scale_value)
        self._animation.setEndValue(target_value)
        self._animation.start()

    def enterEvent(self, event: Any) -> None:
        self._animate_to(1.05)
        super().enterEvent(event)

    def leaveEvent(self, event: Any) -> None:
        self._animate_to(1.0)
        super().leaveEvent(event)

    def mousePressEvent(self, event: Any) -> None:
        self._animate_to(0.95, duration=90)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self.rect().contains(event.pos()):
            self._animate_to(1.05)
        else:
            self._animate_to(1.0)
        super().mouseReleaseEvent(event)

    def paintEvent(self, event: Any) -> None:
        option = QStyleOptionButton()
        self.initStyleOption(option)
        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        center_x = self.width() / 2.0
        center_y = self.height() / 2.0
        painter.translate(center_x, center_y)
        painter.scale(self._scale_value, self._scale_value)
        painter.translate(-center_x, -center_y)
        painter.drawControl(QStyle.CE_PushButton, option)

    @pyqtProperty(float)
    def scaleValue(self) -> float:
        return self._scale_value

    @scaleValue.setter
    def scaleValue(self, value: float) -> None:
        self._scale_value = value
        self.update()


class AnimatedNumberLabel(QLabel):
    """数字滚动标签，用于分数与统计信息展示。"""

    def __init__(self, prefix: str = "", suffix: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._prefix = prefix
        self._suffix = suffix
        self._displayed_value = 0
        self._animation = QPropertyAnimation(self, b"displayedValue", self)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self.setText(self._compose_text(0))

    def _compose_text(self, value: int) -> str:
        return "{}{}{}".format(self._prefix, value, self._suffix)

    def animate_to(self, target_value: int, duration: int = 650) -> None:
        self._animation.stop()
        self._animation.setDuration(duration)
        self._animation.setStartValue(self._displayed_value)
        self._animation.setEndValue(int(target_value))
        self._animation.start()

    def set_instant_value(self, target_value: int) -> None:
        self._animation.stop()
        self.displayedValue = int(target_value)

    @pyqtProperty(int)
    def displayedValue(self) -> int:
        return self._displayed_value

    @displayedValue.setter
    def displayedValue(self, value: int) -> None:
        self._displayed_value = int(value)
        self.setText(self._compose_text(self._displayed_value))


class RoundedPanel(QFrame):
    """统一的圆角卡片面板。"""

    def __init__(
        self,
        background_color: str = COLORS.white,
        border_color: str = COLORS.border,
        radius: int = 24,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._background_color = background_color
        self._border_color = border_color
        self._radius = radius
        self._flash_opacity = 0.0
        self._flash_color = COLORS.primary
        self._flash_group: Optional[QSequentialAnimationGroup] = None
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent; border: none;")
        apply_shadow(self)

    def resizeEvent(self, event: Any) -> None:
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect()), self._radius, self._radius)
        self.setMask(QRegion(path.toFillPolygon().toPolygon()))
        super().resizeEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), self._radius, self._radius)
        painter.fillPath(path, qcolor(self._background_color))
        painter.setPen(QPen(qcolor(self._border_color), 1.5))
        painter.drawPath(path)
        if self._flash_opacity > 0:
            painter.setPen(QPen(qcolor(self._flash_color, int(255 * self._flash_opacity)), 6))
            painter.drawPath(path)

    def flash(self, color_hex: str, loops: int = 1) -> None:
        self._flash_color = color_hex
        self._flash_group = QSequentialAnimationGroup(self)
        for _ in range(max(1, loops)):
            fade_in = QPropertyAnimation(self, b"flashOpacity", self)
            fade_in.setDuration(120)
            fade_in.setStartValue(0.0)
            fade_in.setEndValue(1.0)
            fade_in.setEasingCurve(QEasingCurve.OutCubic)
            fade_out = QPropertyAnimation(self, b"flashOpacity", self)
            fade_out.setDuration(160)
            fade_out.setStartValue(1.0)
            fade_out.setEndValue(0.0)
            fade_out.setEasingCurve(QEasingCurve.InOutCubic)
            self._flash_group.addAnimation(fade_in)
            self._flash_group.addAnimation(fade_out)
        self._flash_group.start()

    @pyqtProperty(float)
    def flashOpacity(self) -> float:
        return self._flash_opacity

    @flashOpacity.setter
    def flashOpacity(self, value: float) -> None:
        self._flash_opacity = max(0.0, min(1.0, float(value)))
        self.update()


class GradientPage(QWidget):
    """带柔和渐变背景的页面基类。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAutoFillBackground(False)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, qcolor(COLORS.background))
        gradient.setColorAt(0.55, qcolor("#EEF4FF"))
        gradient.setColorAt(1.0, qcolor(COLORS.white))
        painter.fillRect(self.rect(), gradient)
        painter.setPen(Qt.NoPen)
        painter.setBrush(qcolor(COLORS.primary, 20))
        painter.drawEllipse(self.width() - 280, -80, 360, 360)
        painter.setBrush(qcolor(COLORS.success, 18))
        painter.drawEllipse(-120, self.height() - 260, 320, 320)
        super().paintEvent(event)


class TimelineWidget(QWidget):
    """顶部进度条，显示视频进度和关键得分帧位置。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._duration_ms = 1
        self._current_ms = 0
        self._markers: List[int] = []
        self._active_index = -1
        self._blink_on = True
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._toggle_blink)
        self._blink_timer.start(220)
        self.setMinimumHeight(70)

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(1, int(duration_ms))
        self.update()

    def set_markers(self, markers: Sequence[int]) -> None:
        self._markers = [int(value) for value in markers]
        self.update()

    def set_current(self, current_ms: int) -> None:
        self._current_ms = max(0, int(current_ms))
        self.update()

    def set_active_marker(self, marker_index: int) -> None:
        self._active_index = int(marker_index)
        self.update()

    def _toggle_blink(self) -> None:
        self._blink_on = not self._blink_on
        if self._active_index >= 0:
            self.update()

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track_rect = QRectF(24, 24, max(80, self.width() - 48), 16)
        painter.setPen(Qt.NoPen)
        painter.setBrush(qcolor(COLORS.border))
        painter.drawRoundedRect(track_rect, 8, 8)
        progress_ratio = max(0.0, min(1.0, float(self._current_ms) / float(self._duration_ms)))
        progress_rect = QRectF(track_rect)
        progress_rect.setWidth(track_rect.width() * progress_ratio)
        painter.setBrush(qcolor(COLORS.primary))
        painter.drawRoundedRect(progress_rect, 8, 8)
        for index, marker_ms in enumerate(self._markers):
            ratio = max(0.0, min(1.0, float(marker_ms) / float(self._duration_ms)))
            x_pos = track_rect.left() + ratio * track_rect.width()
            is_active = index == self._active_index
            radius = 7 if not is_active else (11 if self._blink_on else 8)
            color_hex = COLORS.warning if is_active else COLORS.white
            border_hex = COLORS.primary if not is_active else COLORS.warning
            painter.setBrush(qcolor(color_hex))
            painter.setPen(QPen(qcolor(border_hex), 3))
            painter.drawEllipse(QPoint(int(x_pos), int(track_rect.center().y())), radius, radius)
        painter.setPen(qcolor(COLORS.muted))
        painter.setFont(build_font(14, bold=False))
        painter.drawText(QRect(24, 44, self.width() - 48, 20), Qt.AlignLeft, ms_to_timestamp(self._current_ms))
        painter.drawText(QRect(24, 44, self.width() - 48, 20), Qt.AlignRight, ms_to_timestamp(self._duration_ms))

class CountdownOverlay(QWidget):
    """全屏倒计时覆盖层。"""

    finished = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._remaining = 0
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.hide()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        self.number_holder = QWidget(self)
        self.number_holder.setFixedSize(420, 420)

        self.number_label = QLabel("3", self.number_holder)
        self.number_label.setAlignment(Qt.AlignCenter)
        self.number_label.setFont(build_font(180, bold=True))
        self.number_label.setStyleSheet("color: %s;" % COLORS.white)
        self.number_label.setGeometry(50, 50, 320, 320)
        self.number_opacity = QGraphicsOpacityEffect(self.number_label)
        self.number_label.setGraphicsEffect(self.number_opacity)
        self.number_opacity.setOpacity(0.0)

        self.caption_label = QLabel("准备开始", self)
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setFont(build_font(28, bold=True))
        self.caption_label.setStyleSheet("color: %s;" % COLORS.white)

        layout.addWidget(self.number_holder, 0, Qt.AlignCenter)
        layout.addWidget(self.caption_label, 0, Qt.AlignCenter)

    def resizeEvent(self, event: Any) -> None:
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        super().resizeEvent(event)

    def paintEvent(self, event: Any) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), qcolor(COLORS.dark_overlay, 210))

    def start(self, seconds: int = 3) -> None:
        self._remaining = seconds
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.show()
        self.raise_()
        self._play_step()

    def _play_step(self) -> None:
        if self._remaining <= 0:
            self.hide()
            self.finished.emit()
            return

        self.number_label.setText(str(self._remaining))
        start_rect = QRect(120, 120, 180, 180)
        end_rect = QRect(50, 50, 320, 320)
        self.number_label.setGeometry(start_rect)
        self.number_opacity.setOpacity(0.0)

        geometry_anim = QPropertyAnimation(self.number_label, b"geometry", self)
        geometry_anim.setDuration(760)
        geometry_anim.setStartValue(start_rect)
        geometry_anim.setEndValue(end_rect)
        geometry_anim.setEasingCurve(QEasingCurve.OutBack)

        opacity_in = QPropertyAnimation(self.number_opacity, b"opacity", self)
        opacity_in.setDuration(240)
        opacity_in.setStartValue(0.0)
        opacity_in.setEndValue(1.0)
        opacity_in.setEasingCurve(QEasingCurve.OutCubic)

        opacity_out = QPropertyAnimation(self.number_opacity, b"opacity", self)
        opacity_out.setDuration(320)
        opacity_out.setStartValue(1.0)
        opacity_out.setEndValue(0.0)
        opacity_out.setEasingCurve(QEasingCurve.InCubic)

        group = QSequentialAnimationGroup(self)
        parallel = QParallelAnimationGroup(group)
        parallel.addAnimation(geometry_anim)
        parallel.addAnimation(opacity_in)
        group.addAnimation(parallel)
        group.addPause(120)
        group.addAnimation(opacity_out)
        group.finished.connect(self._on_step_finished)
        group.start()
        self._countdown_group = group

    def _on_step_finished(self) -> None:
        self._remaining -= 1
        self._play_step()


class AnimatedStackedWidget(QStackedWidget):
    """页面切换时使用淡入淡出效果。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._transition_group: Optional[QParallelAnimationGroup] = None
        self._overlay_labels: List[QLabel] = []

    def setCurrentIndexAnimated(self, target_index: int) -> None:
        if target_index == self.currentIndex():
            return
        old_widget = self.currentWidget()
        if old_widget is None:
            self.setCurrentIndex(target_index)
            return

        old_pixmap = old_widget.grab()
        self.setCurrentIndex(target_index)
        QApplication.processEvents()
        new_pixmap = self.currentWidget().grab()

        for label in self._overlay_labels:
            label.deleteLater()
        self._overlay_labels = []

        old_label = QLabel(self)
        old_label.setPixmap(old_pixmap)
        old_label.setScaledContents(True)
        old_label.setGeometry(self.rect())
        old_label.show()

        new_label = QLabel(self)
        new_label.setPixmap(new_pixmap)
        new_label.setScaledContents(True)
        new_label.setGeometry(self.rect())
        new_label.show()

        old_effect = QGraphicsOpacityEffect(old_label)
        new_effect = QGraphicsOpacityEffect(new_label)
        old_label.setGraphicsEffect(old_effect)
        new_label.setGraphicsEffect(new_effect)
        old_effect.setOpacity(1.0)
        new_effect.setOpacity(0.0)
        self._overlay_labels = [new_label, old_label]

        fade_out = QPropertyAnimation(old_effect, b"opacity", self)
        fade_out.setDuration(360)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.InOutCubic)

        fade_in = QPropertyAnimation(new_effect, b"opacity", self)
        fade_in.setDuration(360)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)
        fade_in.setEasingCurve(QEasingCurve.InOutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(fade_out)
        group.addAnimation(fade_in)
        group.finished.connect(self._clear_overlays)
        group.start()
        self._transition_group = group

    def _clear_overlays(self) -> None:
        for label in self._overlay_labels:
            label.deleteLater()
        self._overlay_labels = []


class LaunchPage(GradientPage):
    """启动页。"""

    start_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(160, 120, 160, 80)
        layout.addStretch(3)

        title = QLabel("AI 动作匹配健康训练系统", self)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(build_font(FONTS.title_px, bold=True))
        title.setStyleSheet("color: %s;" % COLORS.text)

        subtitle = QLabel("单关卡 · 单视频 · 关键帧评分闭环", self)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setFont(build_font(FONTS.subtitle_px, bold=False))
        subtitle.setStyleSheet("color: %s;" % COLORS.muted)

        self.status_label = QLabel("导入教学视频后，即可开始一次完整跟练。", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(build_font(FONTS.body_px))
        self.status_label.setStyleSheet("color: %s;" % COLORS.muted)

        self.start_button = AnimatedButton("开始挑战", parent=self)
        self.start_button.clicked.connect(self.start_clicked.emit)

        layout.addWidget(title)
        layout.addSpacing(18)
        layout.addWidget(subtitle)
        layout.addSpacing(14)
        layout.addWidget(self.status_label)
        layout.addStretch(2)
        layout.addWidget(self.start_button, 0, Qt.AlignHCenter)
        layout.addSpacing(18)

    def set_status(self, text: str, warning: bool = False) -> None:
        self.status_label.setText(text)
        self.status_label.setStyleSheet("color: %s;" % (COLORS.error if warning else COLORS.muted))


class PreparePage(GradientPage):
    """准备页。"""

    start_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(110, 80, 110, 60)
        root_layout.setSpacing(26)

        title = QLabel("准备页", self)
        title.setFont(build_font(42, bold=True))
        title.setStyleSheet("color: %s;" % COLORS.text)
        root_layout.addWidget(title)

        card_layout = QHBoxLayout()
        card_layout.setSpacing(26)

        self.thumbnail_panel = RoundedPanel(background_color=COLORS.white)
        self.thumbnail_panel.setMinimumSize(680, 420)
        thumb_layout = QVBoxLayout(self.thumbnail_panel)
        thumb_layout.setContentsMargins(20, 20, 20, 20)

        self.thumbnail_label = QLabel("暂无缩略图", self.thumbnail_panel)
        self.thumbnail_label.setAlignment(Qt.AlignCenter)
        self.thumbnail_label.setStyleSheet("background: #E9EEF5; color: %s; border-radius: 18px;" % COLORS.muted)
        self.thumbnail_label.setMinimumHeight(360)
        thumb_layout.addWidget(self.thumbnail_label)

        self.info_panel = RoundedPanel(background_color=COLORS.white)
        info_layout = QVBoxLayout(self.info_panel)
        info_layout.setContentsMargins(34, 34, 34, 34)
        info_layout.setSpacing(16)

        self.action_name_label = QLabel("单关卡健康跟练", self.info_panel)
        self.action_name_label.setFont(build_font(34, bold=True))
        self.action_name_label.setStyleSheet("color: %s;" % COLORS.text)

        self.subtitle_label = QLabel("跟随示范动作完成本次练习", self.info_panel)
        self.subtitle_label.setFont(build_font(18))
        self.subtitle_label.setWordWrap(True)
        self.subtitle_label.setStyleSheet("color: %s;" % COLORS.muted)

        self.meta_label = QLabel("时长 -- | 难度 -- | 示范来源 --", self.info_panel)
        self.meta_label.setFont(build_font(18, bold=True))
        self.meta_label.setStyleSheet("color: %s;" % COLORS.primary)

        tips_title = QLabel("动作注意事项", self.info_panel)
        tips_title.setFont(build_font(20, bold=True))
        tips_title.setStyleSheet("color: %s;" % COLORS.text)

        self.notes_label = QLabel("- 保持全身入镜", self.info_panel)
        self.notes_label.setFont(build_font(FONTS.body_px))
        self.notes_label.setWordWrap(True)
        self.notes_label.setStyleSheet("color: %s; line-height: 180%%;" % COLORS.text)

        self.warning_label = QLabel("", self.info_panel)
        self.warning_label.setFont(build_font(16, bold=True))
        self.warning_label.setWordWrap(True)
        self.warning_label.setStyleSheet("color: %s;" % COLORS.error)

        info_layout.addWidget(self.action_name_label)
        info_layout.addWidget(self.subtitle_label)
        info_layout.addWidget(self.meta_label)
        info_layout.addSpacing(6)
        info_layout.addWidget(tips_title)
        info_layout.addWidget(self.notes_label)
        info_layout.addStretch(1)
        info_layout.addWidget(self.warning_label)

        card_layout.addWidget(self.thumbnail_panel, 3)
        card_layout.addWidget(self.info_panel, 2)
        root_layout.addLayout(card_layout, 1)

        self.start_button = AnimatedButton("准备开始", parent=self)
        root_layout.addWidget(self.start_button, 0, Qt.AlignHCenter)
        self.start_button.clicked.connect(self.start_clicked.emit)

    def set_level_info(self, info: Dict[str, Any], ready: bool, warning_text: str = "") -> None:
        self.action_name_label.setText(str(info.get("action_name", "单关卡健康跟练")))
        self.subtitle_label.setText(str(info.get("subtitle", "跟随示范动作完成本次练习")))
        duration_ms = int(info.get("video_duration_ms", 0))
        self.meta_label.setText(
            "时长 {} | 难度 {} | 示范来源 {}".format(
                ms_to_timestamp(duration_ms),
                info.get("difficulty", "初级"),
                info.get("coach_name", "标准示范视频"),
            )
        )
        notes = info.get("notes", []) or []
        self.notes_label.setText("\n".join(["- {}".format(item) for item in notes]))
        self.warning_label.setText(warning_text)
        self.start_button.setEnabled(ready)

        thumbnail_path = info.get("thumbnail_path")
        thumbnail = QPixmap(str(thumbnail_path)) if thumbnail_path else QPixmap()
        if not thumbnail.isNull():
            pixmap = thumbnail.scaled(
                self.thumbnail_label.size() if self.thumbnail_label.width() > 1 else QSize(620, 360),
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation,
            )
            self.thumbnail_label.setPixmap(pixmap)
            self.thumbnail_label.setScaledContents(True)
            self.thumbnail_label.setText("")
        else:
            self.thumbnail_label.setPixmap(QPixmap())
            self.thumbnail_label.setText("暂无缩略图")

class FeedbackPopup(QWidget):
    """关卡页右下反馈区的动效消息。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(120)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

        self.card = RoundedPanel(background_color=COLORS.white, border_color=COLORS.border, radius=20, parent=self)
        self.card.setGeometry(0, 18, 480, 92)
        card_layout = QVBoxLayout(self.card)
        card_layout.setContentsMargins(22, 14, 22, 14)
        card_layout.setSpacing(4)

        self.title_label = QLabel("等待关键帧评分", self.card)
        self.title_label.setFont(build_font(24, bold=True))
        self.subtitle_label = QLabel("当前将根据关键帧姿态进行匹配评分", self.card)
        self.subtitle_label.setFont(build_font(16))
        self.subtitle_label.setWordWrap(True)

        card_layout.addWidget(self.title_label)
        card_layout.addWidget(self.subtitle_label)

        self.opacity_effect = QGraphicsOpacityEffect(self.card)
        self.card.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(1.0)
        self._group: Optional[QSequentialAnimationGroup] = None

    def show_message(self, level: str, title: str, subtitle: str) -> None:
        color_map = {
            "pass": COLORS.success,
            "warn": COLORS.warning,
            "fail": COLORS.error,
            "info": COLORS.primary,
        }
        accent = color_map.get(level, COLORS.primary)
        self.title_label.setText(title)
        self.subtitle_label.setText(subtitle)
        self.title_label.setStyleSheet("color: %s;" % accent)
        self.subtitle_label.setStyleSheet("color: %s;" % COLORS.text)
        self.card.flash(accent, loops=2 if level == "fail" else 3 if level == "pass" else 1)

        start_rect = QRect(0, 36, self.width(), 84)
        end_rect = QRect(0, 8, self.width(), 92)
        self.card.setGeometry(start_rect)
        self.opacity_effect.setOpacity(0.0)

        move_anim = QPropertyAnimation(self.card, b"geometry", self)
        move_anim.setDuration(520)
        move_anim.setStartValue(start_rect)
        move_anim.setEndValue(end_rect)
        move_anim.setEasingCurve(QEasingCurve.OutBack)

        fade_in = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        fade_in.setDuration(220)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        fade_out = QPropertyAnimation(self.opacity_effect, b"opacity", self)
        fade_out.setDuration(360)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.15)

        group = QSequentialAnimationGroup(self)
        parallel = QParallelAnimationGroup(group)
        parallel.addAnimation(move_anim)
        parallel.addAnimation(fade_in)
        group.addAnimation(parallel)
        group.addPause(650)
        group.addAnimation(fade_out)
        group.start()
        self._group = group


class GamePage(GradientPage):
    """关卡核心页。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(42, 26, 42, 30)
        root_layout.setSpacing(18)

        header_layout = QVBoxLayout()
        header_layout.setSpacing(12)

        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.level_title = QLabel("单关卡健康跟练", self)
        self.level_title.setFont(build_font(26, bold=True))
        self.level_title.setStyleSheet("color: %s;" % COLORS.text)
        self.level_subtitle = QLabel("请在关键得分帧将动作做到位", self)
        self.level_subtitle.setFont(build_font(16))
        self.level_subtitle.setStyleSheet("color: %s;" % COLORS.muted)
        title_row.addWidget(self.level_title)
        title_row.addStretch(1)
        title_row.addWidget(self.level_subtitle)

        self.timeline = TimelineWidget(self)
        header_layout.addLayout(title_row)
        header_layout.addWidget(self.timeline)
        root_layout.addLayout(header_layout)

        center_layout = QHBoxLayout()
        center_layout.setSpacing(26)

        self.video_panel = RoundedPanel(background_color="#0F172A", border_color="#23324D", radius=28)
        self.video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout = QVBoxLayout(self.video_panel)
        video_layout.setContentsMargins(14, 14, 14, 14)
        self.video_label = QLabel(self.video_panel)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #0F172A; border-radius: 20px;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        video_layout.addWidget(self.video_label)

        self.camera_panel = RoundedPanel(background_color="#0F172A", border_color="#23324D", radius=28)
        self.camera_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_layout = QVBoxLayout(self.camera_panel)
        camera_layout.setContentsMargins(14, 14, 14, 14)
        self.camera_label = QLabel("摄像头连接中...", self.camera_panel)
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setFont(build_font(24, bold=True))
        self.camera_label.setStyleSheet("background: #0F172A; color: #E2E8F0; border-radius: 20px;")
        self.camera_label.setMinimumSize(720, 720)
        camera_layout.addWidget(self.camera_label)

        center_layout.addWidget(self.video_panel, 1)
        center_layout.addWidget(self.camera_panel, 1)
        root_layout.addLayout(center_layout, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(26)

        self.metric_panel = RoundedPanel(background_color=COLORS.white, border_color=COLORS.border)
        self.metric_panel.setFixedWidth(420)
        metric_layout = QGridLayout(self.metric_panel)
        metric_layout.setContentsMargins(26, 24, 26, 24)
        metric_layout.setHorizontalSpacing(16)
        metric_layout.setVerticalSpacing(10)

        score_title = QLabel("实时得分", self.metric_panel)
        score_title.setFont(build_font(18, bold=True))
        score_title.setStyleSheet("color: %s;" % COLORS.text)
        combo_title = QLabel("连击数", self.metric_panel)
        combo_title.setFont(build_font(18, bold=True))
        combo_title.setStyleSheet("color: %s;" % COLORS.text)

        self.score_value = AnimatedNumberLabel(parent=self.metric_panel)
        self.score_value.setFont(build_font(FONTS.score_px, bold=True))
        self.score_value.setStyleSheet("color: %s;" % COLORS.primary)
        self.combo_value = QLabel("0", self.metric_panel)
        self.combo_value.setFont(build_font(46, bold=True))
        self.combo_value.setStyleSheet("color: %s;" % COLORS.warning)

        self.multiplier_label = QLabel("倍率 x1.0", self.metric_panel)
        self.multiplier_label.setFont(build_font(16, bold=True))
        self.multiplier_label.setStyleSheet("color: %s;" % COLORS.muted)
        self.session_hint = QLabel("等待关键帧到达", self.metric_panel)
        self.session_hint.setFont(build_font(16))
        self.session_hint.setWordWrap(True)
        self.session_hint.setStyleSheet("color: %s;" % COLORS.muted)

        metric_layout.addWidget(score_title, 0, 0)
        metric_layout.addWidget(combo_title, 0, 1)
        metric_layout.addWidget(self.score_value, 1, 0)
        metric_layout.addWidget(self.combo_value, 1, 1)
        metric_layout.addWidget(self.multiplier_label, 2, 0)
        metric_layout.addWidget(self.session_hint, 2, 1)

        self.feedback_panel = RoundedPanel(background_color=COLORS.white, border_color=COLORS.border)
        feedback_layout = QVBoxLayout(self.feedback_panel)
        feedback_layout.setContentsMargins(24, 20, 24, 16)
        feedback_layout.setSpacing(10)

        self.feedback_title = QLabel("动作反馈区", self.feedback_panel)
        self.feedback_title.setFont(build_font(18, bold=True))
        self.feedback_title.setStyleSheet("color: %s;" % COLORS.text)

        self.status_label = QLabel("视频播放后将在关键帧自动抓拍评分。", self.feedback_panel)
        self.status_label.setFont(build_font(16))
        self.status_label.setStyleSheet("color: %s;" % COLORS.muted)
        self.status_label.setWordWrap(True)

        self.feedback_popup = FeedbackPopup(self.feedback_panel)
        self.feedback_popup.setFixedWidth(520)

        self.combo_banner = QLabel("", self.feedback_panel)
        self.combo_banner.setFixedHeight(42)
        self.combo_banner.setAlignment(Qt.AlignCenter)
        self.combo_banner.setFont(build_font(20, bold=True))
        self.combo_banner.setStyleSheet(
            "background: rgba(251, 189, 35, 0.15); color: %s; border-radius: 12px;" % COLORS.warning
        )
        self.combo_banner.hide()
        self.combo_banner_effect = QGraphicsOpacityEffect(self.combo_banner)
        self.combo_banner.setGraphicsEffect(self.combo_banner_effect)
        self.combo_banner_effect.setOpacity(0.0)

        feedback_layout.addWidget(self.feedback_title)
        feedback_layout.addWidget(self.status_label)
        feedback_layout.addWidget(self.feedback_popup)
        feedback_layout.addWidget(self.combo_banner)
        feedback_layout.addStretch(1)

        bottom_layout.addWidget(self.metric_panel)
        bottom_layout.addWidget(self.feedback_panel, 1)
        root_layout.addLayout(bottom_layout)

    def reset_session(self, info: Dict[str, Any]) -> None:
        self.level_title.setText(str(info.get("action_name", "单关卡健康跟练")))
        self.level_subtitle.setText(str(info.get("subtitle", "请在关键得分帧将动作做到位")))
        duration_ms = int(info.get("video_duration_ms", 1))
        marker_values = [int(item.get("timestamp_ms", 0)) for item in info.get("keyframes", [])]
        self.timeline.set_duration(duration_ms)
        self.timeline.set_markers(marker_values)
        self.timeline.set_current(0)
        self.timeline.set_active_marker(-1)
        self.score_value.set_instant_value(0)
        self.combo_value.setText("0")
        self.multiplier_label.setText("倍率 x1.0")
        self.session_hint.setText("等待关键帧到达")
        self.status_label.setText("视频播放后将在关键帧自动抓拍评分。")
        self.feedback_popup.show_message("info", "准备开始", "请跟随左侧示范视频同步完成动作")
        self.combo_banner.hide()
        self.camera_label.setText("摄像头连接中...")
        self.camera_label.setPixmap(QPixmap())

    def set_camera_image(self, pixmap: QPixmap) -> None:
        scaled = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.camera_label.setPixmap(scaled)
        self.camera_label.setText("")
        self.camera_label.setScaledContents(True)

    def set_camera_placeholder(self, text: str) -> None:
        self.camera_label.setPixmap(QPixmap())
        self.camera_label.setScaledContents(False)
        self.camera_label.setText(text)

    def set_progress(self, position_ms: int) -> None:
        self.timeline.set_current(position_ms)

    def highlight_marker(self, marker_index: int) -> None:
        self.timeline.set_active_marker(marker_index)

    def set_score_and_combo(self, score_value: int, combo_value: int, multiplier: float) -> None:
        self.score_value.animate_to(score_value, duration=420)
        self.combo_value.setText(str(combo_value))
        self.multiplier_label.setText("倍率 x{:.1f}".format(multiplier))

    def show_status(self, text: str, level: str = "info") -> None:
        color_map = {
            "info": COLORS.muted,
            "pass": COLORS.success,
            "warn": COLORS.warning,
            "fail": COLORS.error,
        }
        self.status_label.setStyleSheet("color: %s;" % color_map.get(level, COLORS.muted))
        self.status_label.setText(text)

    def show_feedback(self, level: str, title: str, subtitle: str) -> None:
        self.feedback_popup.show_message(level, title, subtitle)
        if level == "pass":
            self.camera_panel.flash(COLORS.success, loops=3)
            self.video_panel.flash(COLORS.success, loops=2)
            self._pulse_score()
        elif level == "fail":
            self.camera_panel.flash(COLORS.error, loops=2)
            self._shake_widget(self.feedback_popup)
        elif level == "warn":
            self.camera_panel.flash(COLORS.warning, loops=1)
            self._shake_widget(self.feedback_popup)

    def _pulse_score(self) -> None:
        """pass 时让分数标签做缩放弹跳。"""
        anim = QPropertyAnimation(self.score_value, b"geometry", self)
        original = self.score_value.geometry()
        expanded = original.adjusted(-8, -6, 8, 6)
        anim.setDuration(300)
        anim.setKeyValueAt(0.0, original)
        anim.setKeyValueAt(0.4, expanded)
        anim.setKeyValueAt(1.0, original)
        anim.setEasingCurve(QEasingCurve.OutElastic)
        anim.start()
        self._pulse_anim = anim

    def _shake_widget(self, widget: QWidget) -> None:
        """warn/fail 时让控件左右快速抖动。"""
        pos = widget.pos()
        anim = QPropertyAnimation(widget, b"pos", self)
        anim.setDuration(300)
        anim.setKeyValueAt(0.0, pos)
        anim.setKeyValueAt(0.15, QPoint(pos.x() - 10, pos.y()))
        anim.setKeyValueAt(0.35, QPoint(pos.x() + 10, pos.y()))
        anim.setKeyValueAt(0.55, QPoint(pos.x() - 6, pos.y()))
        anim.setKeyValueAt(0.75, QPoint(pos.x() + 6, pos.y()))
        anim.setKeyValueAt(1.0, pos)
        anim.start()
        self._shake_anim = anim

    def show_combo(self, text: str) -> None:
        self.combo_banner.setText(text)
        self.combo_banner.show()
        start_pos = QPoint(0, self.combo_banner.y())
        left_pos = QPoint(-20, self.combo_banner.y())
        right_pos = QPoint(20, self.combo_banner.y())
        self.combo_banner.move(start_pos)
        self.combo_banner_effect.setOpacity(0.0)

        fade_in = QPropertyAnimation(self.combo_banner_effect, b"opacity", self)
        fade_in.setDuration(180)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        shake_1 = QPropertyAnimation(self.combo_banner, b"pos", self)
        shake_1.setDuration(90)
        shake_1.setStartValue(start_pos)
        shake_1.setEndValue(left_pos)
        shake_1.setEasingCurve(QEasingCurve.OutQuad)

        shake_2 = QPropertyAnimation(self.combo_banner, b"pos", self)
        shake_2.setDuration(90)
        shake_2.setStartValue(left_pos)
        shake_2.setEndValue(right_pos)

        shake_3 = QPropertyAnimation(self.combo_banner, b"pos", self)
        shake_3.setDuration(90)
        shake_3.setStartValue(right_pos)
        shake_3.setEndValue(start_pos)

        fade_out = QPropertyAnimation(self.combo_banner_effect, b"opacity", self)
        fade_out.setDuration(260)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)

        group = QSequentialAnimationGroup(self)
        group.addAnimation(fade_in)
        group.addAnimation(shake_1)
        group.addAnimation(shake_2)
        group.addAnimation(shake_3)
        group.addPause(400)
        group.addAnimation(fade_out)
        group.finished.connect(self.combo_banner.hide)
        group.start()
        self._combo_group = group

class ResultPage(GradientPage):
    """结算页。"""

    restart_clicked = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(140, 70, 140, 70)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        title = QLabel("本次训练结算", self)
        title.setAlignment(Qt.AlignCenter)
        title.setFont(build_font(42, bold=True))
        title.setStyleSheet("color: %s;" % COLORS.text)

        self.status_label = QLabel("视频完成，以下为本次成绩。", self)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFont(build_font(18))
        self.status_label.setStyleSheet("color: %s;" % COLORS.muted)

        self.score_label = AnimatedNumberLabel(parent=self)
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(build_font(96, bold=True))
        self.score_label.setStyleSheet("color: %s;" % COLORS.primary)

        self.badge_holder = QWidget(self)
        self.badge_holder.setFixedSize(260, 92)
        self.badge_label = QLabel("B", self.badge_holder)
        self.badge_label.setAlignment(Qt.AlignCenter)
        self.badge_label.setGeometry(40, 18, 180, 56)
        self.badge_label.setFont(build_font(34, bold=True))
        self.badge_label.setStyleSheet(
            "background: rgba(54, 211, 153, 0.12); color: %s; border-radius: 18px;" % COLORS.success
        )
        self.badge_effect = QGraphicsOpacityEffect(self.badge_label)
        self.badge_label.setGraphicsEffect(self.badge_effect)
        self.badge_effect.setOpacity(0.0)

        self.stats_panel = RoundedPanel(background_color=COLORS.white, border_color=COLORS.border)
        self.stats_panel.setFixedWidth(860)
        stats_layout = QGridLayout(self.stats_panel)
        stats_layout.setContentsMargins(36, 28, 36, 28)
        stats_layout.setHorizontalSpacing(18)
        stats_layout.setVerticalSpacing(18)

        self.pass_rate_label = QLabel("0%", self.stats_panel)
        self.max_combo_label = QLabel("0", self.stats_panel)
        self.total_actions_label = QLabel("0", self.stats_panel)

        for label in (self.pass_rate_label, self.max_combo_label, self.total_actions_label):
            label.setFont(build_font(34, bold=True))
            label.setStyleSheet("color: %s;" % COLORS.text)
            label.setAlignment(Qt.AlignCenter)

        captions = ["动作达标率", "最高连击", "总完成动作数"]
        values = [self.pass_rate_label, self.max_combo_label, self.total_actions_label]
        for column, (caption, value_label) in enumerate(zip(captions, values)):
            caption_label = QLabel(caption, self.stats_panel)
            caption_label.setFont(build_font(18, bold=True))
            caption_label.setStyleSheet("color: %s;" % COLORS.muted)
            caption_label.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(caption_label, 0, column)
            stats_layout.addWidget(value_label, 1, column)

        self.restart_button = AnimatedButton("再来一次", parent=self)
        self.restart_button.clicked.connect(self.restart_clicked.emit)

        layout.addWidget(title)
        layout.addWidget(self.status_label)
        layout.addSpacing(4)
        layout.addWidget(self.score_label)
        layout.addWidget(self.badge_holder, 0, Qt.AlignCenter)
        layout.addWidget(self.stats_panel, 0, Qt.AlignCenter)
        layout.addWidget(self.restart_button, 0, Qt.AlignCenter)

    def show_result(self, summary: Dict[str, Any]) -> None:
        self.status_label.setText(str(summary.get("status_text", "视频完成，以下为本次成绩。")))
        self.score_label.set_instant_value(0)
        self.score_label.animate_to(int(summary.get("final_score", 0)), duration=1200)

        self.pass_rate_label.setText(str(summary.get("pass_rate", "0%")))
        self.max_combo_label.setText(str(summary.get("max_combo", 0)))
        self.total_actions_label.setText(str(summary.get("completed_actions", 0)))

        badge_text = str(summary.get("rating_label", "B"))
        badge_color = str(summary.get("rating_color", COLORS.warning))
        self.badge_label.setText(badge_text)
        self.badge_label.setStyleSheet(
            "background: rgba(22, 93, 255, 0.10); color: %s; border-radius: 18px;" % badge_color
        )

        start_rect = QRect(80, 30, 100, 34)
        end_rect = QRect(40, 18, 180, 56)
        self.badge_label.setGeometry(start_rect)
        self.badge_effect.setOpacity(0.0)

        geometry_anim = QPropertyAnimation(self.badge_label, b"geometry", self)
        geometry_anim.setDuration(680)
        geometry_anim.setStartValue(start_rect)
        geometry_anim.setEndValue(end_rect)
        geometry_anim.setEasingCurve(QEasingCurve.OutBack)

        fade_anim = QPropertyAnimation(self.badge_effect, b"opacity", self)
        fade_anim.setDuration(420)
        fade_anim.setStartValue(0.0)
        fade_anim.setEndValue(1.0)
        fade_anim.setEasingCurve(QEasingCurve.OutCubic)

        group = QParallelAnimationGroup(self)
        group.addAnimation(geometry_anim)
        group.addAnimation(fade_anim)
        group.start()
        self._badge_group = group
