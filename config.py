"""
项目统一配置文件。

本文件只负责集中管理颜色、路径、评分阈值和默认元数据，
避免业务逻辑散落在多个文件中，方便后续调参与维护。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
GENERATED_DIR = DATA_DIR / "generated"


def _hex_to_rgb(color_hex: str) -> Tuple[int, int, int]:
    """将 #RRGGBB 颜色转换为 RGB 三元组。"""

    color = color_hex.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _hex_to_bgr(color_hex: str) -> Tuple[int, int, int]:
    """OpenCV 需要 BGR 顺序，这里做一次统一转换。"""

    r_value, g_value, b_value = _hex_to_rgb(color_hex)
    return b_value, g_value, r_value


@dataclass(frozen=True)
class ColorPalette:
    """项目固定配色方案。"""

    primary: str = "#165DFF"
    success: str = "#36D399"
    warning: str = "#FBBD23"
    error: str = "#F87272"
    background: str = "#F8F9FA"
    white: str = "#FFFFFF"
    text: str = "#343A40"
    muted: str = "#6C757D"
    border: str = "#DDE2E8"
    dark_overlay: str = "#101828"

    primary_bgr: Tuple[int, int, int] = field(default_factory=lambda: _hex_to_bgr("#165DFF"))
    success_bgr: Tuple[int, int, int] = field(default_factory=lambda: _hex_to_bgr("#36D399"))
    warning_bgr: Tuple[int, int, int] = field(default_factory=lambda: _hex_to_bgr("#FBBD23"))
    error_bgr: Tuple[int, int, int] = field(default_factory=lambda: _hex_to_bgr("#F87272"))


@dataclass(frozen=True)
class FontConfig:
    """统一字体与字号。"""

    family: str = "Source Han Sans CN"
    fallback_family: str = "Microsoft YaHei UI"
    title_px: int = 48
    subtitle_px: int = 24
    body_px: int = 16
    button_px: int = 20
    score_px: int = 64
    badge_px: int = 36


@dataclass(frozen=True)
class PathConfig:
    """项目关键路径。"""

    video_path: Path = BASE_DIR / "跟练视频.MP4"
    audio_path: Path = GENERATED_DIR / "audio.wav"
    legacy_script_path: Path = BASE_DIR / "跟练动作脚本.json"
    standard_template_path: Path = GENERATED_DIR / "standard_pose_templates.json"
    score_frame_path: Path = GENERATED_DIR / "score_frames.json"
    thumbnail_path: Path = GENERATED_DIR / "thumbnail.jpg"


@dataclass(frozen=True)
class ScoreConfig:
    """评分与同步相关参数。"""

    pass_threshold: int = 80
    warn_threshold: int = 60
    min_core_visibility: float = 0.60
    max_sample_delay_ms: int = 100
    score_trigger_lookahead_ms: int = 40
    combo_trigger_count: int = 2
    combo_bonus_step: float = 0.20
    combo_bonus_cap: float = 2.00


@dataclass(frozen=True)
class RuntimeConfig:
    """运行时参数。"""

    camera_id: int = 0
    camera_width: int = 960
    camera_height: int = 720
    camera_retry_limit: int = 60
    pose_model_complexity: int = 1
    detection_confidence: float = 0.55
    tracking_confidence: float = 0.55
    pose_smoothing_alpha: float = 0.65
    ui_fps_ms: int = 16
    countdown_seconds: int = 3
    window_width: int = 1920
    window_height: int = 1080
    video_panel_ratio: float = 1.0


@dataclass(frozen=True)
class LevelDefaults:
    """准备页与结算页在配置缺失时使用的默认信息。"""

    level_name: str = "单关卡健康跟练"
    level_subtitle: str = "跟随示范动作完成一次稳定、轻量、闭环的姿态训练"
    difficulty: str = "初级"
    coach_name: str = "标准示范视频"
    notes: List[str] = field(
        default_factory=lambda: [
            "保持全身尽量完整出现在镜头中",
            "动作以标准和稳定为主，不必追求速度",
            "到达关键得分帧时尽量保持动作到位",
        ]
    )


COLORS = ColorPalette()
FONTS = FontConfig()
PATHS = PathConfig()
SCORE_RULES = ScoreConfig()
RUNTIME = RuntimeConfig()
LEVEL_DEFAULTS = LevelDefaults()


ACTION_HINT_LIBRARY: Dict[str, str] = {
    "raise_arms": "手臂再抬高一点，让手腕高于肩线",
    "open_chest": "肩部再打开一点，保持躯干挺直",
    "squat": "再下蹲一些，注意膝盖朝向脚尖",
    "lunge": "弓步幅度可以再明确，保持重心稳定",
    "twist": "上半身扭转幅度再大一点，核心保持收紧",
    "throw": "手臂甩出去的幅度再大一点！",
    "pull_down": "从上往下拉的动作再用力一些！",
    "wrist_rotate": "手腕旋转的幅度再大一些！",
    "knock": "敲击动作再有力一些！",
    "clap": "鼓掌再响亮一些，双手拍合到位！",
    "wave": "挥手幅度再大一些！",
    "step": "踏步动作再有力，抬腿再高一点！",
    "body_shake": "全身摇摆幅度再大一些！",
    "jump": "跳得再高一些，双脚离地！",
    "default": "动作幅度再靠近示范姿态，保持稳定呼吸",
}


LEVEL_INFO_FALLBACK = {
    "action_name": LEVEL_DEFAULTS.level_name,
    "subtitle": LEVEL_DEFAULTS.level_subtitle,
    "difficulty": LEVEL_DEFAULTS.difficulty,
    "coach_name": LEVEL_DEFAULTS.coach_name,
    "notes": LEVEL_DEFAULTS.notes,
}


def ensure_runtime_directories() -> None:
    """确保运行和预处理需要的输出目录存在。"""

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
