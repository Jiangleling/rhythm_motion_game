"""
姿态检测、关键点归一化、模板读写和评分算法。

该模块被主程序与预处理脚本共用，尽量保持纯工具化，
避免和 PyQt 界面逻辑产生耦合。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mediapipe as mp
import numpy as np

from app.config import ACTION_HINT_LIBRARY, COLORS, LEVEL_INFO_FALLBACK, PATHS, SCORE_RULES


_MP_POSE = mp.solutions.pose

CORE_JOINTS: List[Tuple[str, int]] = [
    ("left_shoulder", _MP_POSE.PoseLandmark.LEFT_SHOULDER.value),
    ("right_shoulder", _MP_POSE.PoseLandmark.RIGHT_SHOULDER.value),
    ("left_elbow", _MP_POSE.PoseLandmark.LEFT_ELBOW.value),
    ("right_elbow", _MP_POSE.PoseLandmark.RIGHT_ELBOW.value),
    ("left_wrist", _MP_POSE.PoseLandmark.LEFT_WRIST.value),
    ("right_wrist", _MP_POSE.PoseLandmark.RIGHT_WRIST.value),
    ("left_hip", _MP_POSE.PoseLandmark.LEFT_HIP.value),
    ("right_hip", _MP_POSE.PoseLandmark.RIGHT_HIP.value),
    ("left_knee", _MP_POSE.PoseLandmark.LEFT_KNEE.value),
    ("right_knee", _MP_POSE.PoseLandmark.RIGHT_KNEE.value),
    ("left_ankle", _MP_POSE.PoseLandmark.LEFT_ANKLE.value),
    ("right_ankle", _MP_POSE.PoseLandmark.RIGHT_ANKLE.value),
]

_MIRROR_INDEX_MAP = np.array([1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10], dtype=np.int32)

DRAW_CONNECTIONS: List[Tuple[int, int]] = [
    (_MP_POSE.PoseLandmark.LEFT_SHOULDER.value, _MP_POSE.PoseLandmark.RIGHT_SHOULDER.value),
    (_MP_POSE.PoseLandmark.LEFT_SHOULDER.value, _MP_POSE.PoseLandmark.LEFT_ELBOW.value),
    (_MP_POSE.PoseLandmark.LEFT_ELBOW.value, _MP_POSE.PoseLandmark.LEFT_WRIST.value),
    (_MP_POSE.PoseLandmark.RIGHT_SHOULDER.value, _MP_POSE.PoseLandmark.RIGHT_ELBOW.value),
    (_MP_POSE.PoseLandmark.RIGHT_ELBOW.value, _MP_POSE.PoseLandmark.RIGHT_WRIST.value),
    (_MP_POSE.PoseLandmark.LEFT_SHOULDER.value, _MP_POSE.PoseLandmark.LEFT_HIP.value),
    (_MP_POSE.PoseLandmark.RIGHT_SHOULDER.value, _MP_POSE.PoseLandmark.RIGHT_HIP.value),
    (_MP_POSE.PoseLandmark.LEFT_HIP.value, _MP_POSE.PoseLandmark.RIGHT_HIP.value),
    (_MP_POSE.PoseLandmark.LEFT_HIP.value, _MP_POSE.PoseLandmark.LEFT_KNEE.value),
    (_MP_POSE.PoseLandmark.LEFT_KNEE.value, _MP_POSE.PoseLandmark.LEFT_ANKLE.value),
    (_MP_POSE.PoseLandmark.RIGHT_HIP.value, _MP_POSE.PoseLandmark.RIGHT_KNEE.value),
    (_MP_POSE.PoseLandmark.RIGHT_KNEE.value, _MP_POSE.PoseLandmark.RIGHT_ANKLE.value),
]


@dataclass
class PoseDetectionResult:
    """单帧姿态检测结果。"""

    visible: bool
    landmarks: List[Dict[str, float]] = field(default_factory=list)
    normalized_vector: Optional[np.ndarray] = None
    core_visibility: float = 0.0
    missing_core_joints: List[str] = field(default_factory=list)


@dataclass
class KeyframeRecord:
    """运行时使用的关键得分帧信息。"""

    template_id: str
    timestamp_ms: int
    frame_index: int
    label: str
    action: str
    correction_hint: str
    pass_threshold: int
    warn_threshold: int
    template_vector: np.ndarray


@dataclass
class LevelBundle:
    """标准模板与关键得分帧的合并结果。"""

    video_path: Path
    audio_path: Optional[Path]
    thumbnail_path: Path
    video_duration_ms: int
    action_name: str
    subtitle: str
    difficulty: str
    coach_name: str
    notes: List[str]
    keyframes: List[KeyframeRecord]


class PoseDetector:
    """MediaPipe Pose 的简单封装。"""

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
    ) -> None:
        self._pose = _MP_POSE.Pose(
            static_image_mode=static_image_mode,
            model_complexity=model_complexity,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> PoseDetectionResult:
        """对单帧进行姿态检测，并输出归一化后的核心向量。"""

        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb_frame)
        if not result.pose_landmarks:
            return PoseDetectionResult(visible=False)

        landmarks: List[Dict[str, float]] = []
        for landmark in result.pose_landmarks.landmark:
            landmarks.append(
                {
                    "x": float(landmark.x),
                    "y": float(landmark.y),
                    "z": float(landmark.z),
                    "visibility": float(getattr(landmark, "visibility", 1.0)),
                }
            )

        vector, core_visibility, missing = normalize_core_joints(landmarks)
        return PoseDetectionResult(
            visible=True,
            landmarks=landmarks,
            normalized_vector=vector,
            core_visibility=core_visibility,
            missing_core_joints=missing,
        )

    def close(self) -> None:
        """释放 MediaPipe 资源。"""

        self._pose.close()


class PoseSmoother:
    """使用指数平滑稳定关键点坐标，减少实时评分抖动。"""

    def __init__(self, alpha: float = 0.65) -> None:
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    def update(self, vector: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """输入新向量，返回平滑后的向量。"""

        if vector is None:
            self._state = None
            return None
        if self._state is None:
            self._state = vector.astype(np.float32)
            return self._state.copy()
        self._state = self.alpha * self._state + (1.0 - self.alpha) * vector.astype(np.float32)
        return self._state.copy()

    def reset(self) -> None:
        """清空平滑状态。"""

        self._state = None


def normalize_core_joints(
    landmarks: Sequence[Dict[str, float]],
    visibility_threshold: float = 0.45,
) -> Tuple[Optional[np.ndarray], float, List[str]]:
    """
    以肩部中点为原点、躯干长度为单位长度进行归一化。

    返回值:
    1. 24 维向量: 12 个核心关节，每个关节使用 (x, y)
    2. 核心关节可见性均值
    3. 可见性较差的核心关节名称列表
    """

    if len(landmarks) <= max(index for _, index in CORE_JOINTS):
        return None, 0.0, [name for name, _ in CORE_JOINTS]

    left_shoulder = landmarks[_MP_POSE.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks[_MP_POSE.PoseLandmark.RIGHT_SHOULDER.value]
    left_hip = landmarks[_MP_POSE.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[_MP_POSE.PoseLandmark.RIGHT_HIP.value]

    shoulder_mid = np.array(
        [
            (left_shoulder["x"] + right_shoulder["x"]) * 0.5,
            (left_shoulder["y"] + right_shoulder["y"]) * 0.5,
        ],
        dtype=np.float32,
    )
    hip_mid = np.array(
        [
            (left_hip["x"] + right_hip["x"]) * 0.5,
            (left_hip["y"] + right_hip["y"]) * 0.5,
        ],
        dtype=np.float32,
    )
    trunk_length = float(np.linalg.norm(hip_mid - shoulder_mid))
    if trunk_length < 1e-6:
        return None, 0.0, [name for name, _ in CORE_JOINTS]

    normalized_points: List[float] = []
    missing: List[str] = []
    visibility_scores: List[float] = []
    for joint_name, joint_index in CORE_JOINTS:
        landmark = landmarks[joint_index]
        visibility = float(landmark.get("visibility", 1.0))
        visibility_scores.append(visibility)
        if visibility < visibility_threshold:
            missing.append(joint_name)
        normalized_points.append((float(landmark["x"]) - float(shoulder_mid[0])) / trunk_length)
        normalized_points.append((float(landmark["y"]) - float(shoulder_mid[1])) / trunk_length)

    return np.array(normalized_points, dtype=np.float32), float(np.mean(visibility_scores)), missing


def mirror_normalized_vector(vector: np.ndarray) -> np.ndarray:
    """镜像归一化向量，用于兼容用户摄像头镜像和左右方向差异。"""

    matrix = np.asarray(vector, dtype=np.float32).reshape(len(CORE_JOINTS), 2)
    mirrored = matrix[_MIRROR_INDEX_MAP].copy()
    mirrored[:, 0] *= -1.0
    return mirrored.reshape(-1)


def _vector_cosine_to_score(first: np.ndarray, second: np.ndarray) -> float:
    """将余弦相似度转换为 0-100 分。"""

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm < 1e-8 or second_norm < 1e-8:
        return 0.0
    cosine_value = float(np.dot(first, second) / (first_norm * second_norm))
    cosine_value = max(-1.0, min(1.0, cosine_value))
    return max(0.0, min(100.0, cosine_value * 100.0))


def cosine_similarity_score(user_vector: np.ndarray, template_vector: np.ndarray) -> Tuple[float, bool]:
    """
    基于余弦相似度计算 0-100 分得分。

    返回值:
    1. 分数
    2. 是否使用了镜像比对结果
    """

    direct_score = _vector_cosine_to_score(user_vector, template_vector)
    mirrored_score = _vector_cosine_to_score(mirror_normalized_vector(user_vector), template_vector)
    if mirrored_score > direct_score:
        return mirrored_score, True
    return direct_score, False


def classify_score(score_value: float, pass_threshold: int, warn_threshold: int) -> str:
    """根据阈值返回评分等级。"""

    if score_value >= pass_threshold:
        return "pass"
    if score_value >= warn_threshold:
        return "warn"
    return "fail"


def combo_multiplier(combo_count: int) -> float:
    """连续达标后倍率逐步提升，第二次达标开始触发 1.2 倍。"""

    if combo_count < SCORE_RULES.combo_trigger_count:
        return 1.0
    bonus_steps = combo_count - SCORE_RULES.combo_trigger_count + 1
    return min(1.0 + bonus_steps * SCORE_RULES.combo_bonus_step, SCORE_RULES.combo_bonus_cap)


def infer_correction_hint(
    user_vector: Optional[np.ndarray],
    template_vector: np.ndarray,
    action_name: str,
    fallback_hint: str = "",
) -> str:
    """
    根据动作向量差异给出简洁纠错提示。

    这里不依赖复杂规则模型，只根据误差最大的关节组给出友好文案。
    """

    if user_vector is None:
        return "请保持全身入镜后重新完成动作"

    if fallback_hint:
        hint_text = fallback_hint.strip()
    else:
        hint_text = ACTION_HINT_LIBRARY.get(action_name, ACTION_HINT_LIBRARY["default"])

    user_matrix = np.asarray(user_vector, dtype=np.float32).reshape(len(CORE_JOINTS), 2)
    template_matrix = np.asarray(template_vector, dtype=np.float32).reshape(len(CORE_JOINTS), 2)
    joint_errors = np.linalg.norm(template_matrix - user_matrix, axis=1)
    worst_joint_index = int(np.argmax(joint_errors))
    joint_name = CORE_JOINTS[worst_joint_index][0]

    if "wrist" in joint_name or "elbow" in joint_name:
        return "手臂角度还可以再展开一些，尽量贴近示范动作"
    if "knee" in joint_name or "ankle" in joint_name:
        return "下肢动作再稳定一些，注意膝踝保持方向一致"
    if "hip" in joint_name:
        return "核心保持收紧，臀髋位置再接近示范姿态"
    if "shoulder" in joint_name:
        return "肩部姿态再打开一些，保持上半身挺拔"
    return hint_text


def draw_pose_overlay(frame_bgr: np.ndarray, landmarks: Sequence[Dict[str, float]]) -> np.ndarray:
    """在画面上绘制主色骨架和关键点。"""

    canvas = frame_bgr
    height, width = canvas.shape[:2]
    for start_index, end_index in DRAW_CONNECTIONS:
        start_lm = landmarks[start_index]
        end_lm = landmarks[end_index]
        if min(start_lm.get("visibility", 0.0), end_lm.get("visibility", 0.0)) < 0.35:
            continue
        start_point = (int(start_lm["x"] * width), int(start_lm["y"] * height))
        end_point = (int(end_lm["x"] * width), int(end_lm["y"] * height))
        cv2.line(canvas, start_point, end_point, COLORS.primary_bgr, 4, cv2.LINE_AA)

    for _, joint_index in CORE_JOINTS:
        joint = landmarks[joint_index]
        if joint.get("visibility", 0.0) < 0.35:
            continue
        point = (int(joint["x"] * width), int(joint["y"] * height))
        cv2.circle(canvas, point, 7, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 10, COLORS.primary_bgr, 2, cv2.LINE_AA)
    return canvas


def save_json(payload: Dict[str, Any], output_path: Path) -> None:
    """统一的 JSON 写入方法。"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def load_json(json_path: Path) -> Dict[str, Any]:
    """统一的 JSON 读取方法。"""

    with json_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_resource_path(raw_path: Any, default_path: Path) -> Path:
    """兼容旧 JSON 中的绝对路径，并优先回退到当前正式资源位置。"""

    if raw_path:
        candidate = Path(str(raw_path))
        if candidate.is_absolute():
            if candidate.exists():
                return candidate.resolve()
            return default_path.resolve()

        for base_dir in (default_path.parent, PATHS.video_path.parent, Path.cwd()):
            resolved = (base_dir / candidate).resolve()
            if resolved.exists():
                return resolved

    return default_path.resolve()


def build_template_payload(
    video_path: Path,
    thumbnail_path: Path,
    templates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """组装标准姿态模板 JSON。"""

    return {
        "video_path": str(video_path),
        "thumbnail_path": str(thumbnail_path),
        "joint_names": [joint_name for joint_name, _ in CORE_JOINTS],
        "templates": list(templates),
    }


def build_score_frame_payload(
    video_path: Path,
    thumbnail_path: Path,
    video_duration_ms: int,
    metadata: Dict[str, Any],
    keyframes: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """组装关键得分帧配置 JSON。"""

    level_info = dict(LEVEL_INFO_FALLBACK)
    level_info.update(metadata or {})
    return {
        "video_path": str(video_path),
        "thumbnail_path": str(thumbnail_path),
        "video_duration_ms": int(video_duration_ms),
        "action_name": level_info["action_name"],
        "subtitle": level_info["subtitle"],
        "difficulty": level_info["difficulty"],
        "coach_name": level_info["coach_name"],
        "notes": level_info["notes"],
        "keyframes": list(keyframes),
    }


def load_level_bundle(
    template_path: Path = PATHS.standard_template_path,
    score_frame_path: Path = PATHS.score_frame_path,
) -> LevelBundle:
    """读取运行时所需的完整关卡数据。"""

    template_payload = load_json(template_path)
    score_payload = load_json(score_frame_path)

    template_map: Dict[str, Dict[str, Any]] = {}
    for item in template_payload.get("templates", []):
        template_id = str(item.get("template_id", ""))
        if not template_id:
            continue
        template_map[template_id] = item

    keyframes: List[KeyframeRecord] = []
    for item in score_payload.get("keyframes", []):
        template_id = str(item.get("template_id", ""))
        template_data = template_map.get(template_id)
        if not template_data:
            continue

        raw_vector = template_data.get("template_vector", [])
        template_vector = np.array(raw_vector, dtype=np.float32)
        if template_vector.size != len(CORE_JOINTS) * 2:
            continue

        keyframes.append(
            KeyframeRecord(
                template_id=template_id,
                timestamp_ms=int(item.get("timestamp_ms", 0)),
                frame_index=int(item.get("frame_index", template_data.get("frame_index", 0))),
                label=str(item.get("label", "关键帧")),
                action=str(item.get("action", "default")),
                correction_hint=str(item.get("correction_hint", "")),
                pass_threshold=int(item.get("pass_threshold", SCORE_RULES.pass_threshold)),
                warn_threshold=int(item.get("warn_threshold", SCORE_RULES.warn_threshold)),
                template_vector=template_vector,
            )
        )

    if not keyframes:
        raise ValueError("未在关键帧配置中找到可用模板，请先运行 preprocess_video.py 生成数据。")

    keyframes.sort(key=lambda record: record.timestamp_ms)
    audio_path = PATHS.audio_path if PATHS.audio_path.exists() else None
    return LevelBundle(
        video_path=resolve_resource_path(
            score_payload.get("video_path", template_payload.get("video_path", "")),
            PATHS.video_path,
        ),
        audio_path=audio_path,
        thumbnail_path=resolve_resource_path(
            score_payload.get("thumbnail_path", template_payload.get("thumbnail_path", PATHS.thumbnail_path)),
            PATHS.thumbnail_path,
        ),
        video_duration_ms=int(score_payload.get("video_duration_ms", 0)),
        action_name=str(score_payload.get("action_name", LEVEL_INFO_FALLBACK["action_name"])),
        subtitle=str(score_payload.get("subtitle", LEVEL_INFO_FALLBACK["subtitle"])),
        difficulty=str(score_payload.get("difficulty", LEVEL_INFO_FALLBACK["difficulty"])),
        coach_name=str(score_payload.get("coach_name", LEVEL_INFO_FALLBACK["coach_name"])),
        notes=list(score_payload.get("notes", LEVEL_INFO_FALLBACK["notes"])),
        keyframes=keyframes,
    )


def extract_thumbnail(video_path: Path, output_path: Path, timestamp_ms: int = 0) -> Optional[Path]:
    """从视频中提取缩略图。"""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    if timestamp_ms > 0:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp_ms))
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return output_path


def read_video_metadata(video_path: Path) -> Dict[str, Any]:
    """读取视频基础元信息。"""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {
            "opened": False,
            "fps": 0.0,
            "frame_count": 0,
            "duration_ms": 0,
            "width": 0,
            "height": 0,
        }
    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    duration_ms = int((frame_count / fps_value) * 1000) if fps_value > 0 else 0
    return {
        "opened": True,
        "fps": fps_value,
        "frame_count": frame_count,
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
    }


def frame_index_to_ms(frame_index: int, fps_value: float) -> int:
    """将帧序号转换为毫秒时间戳。"""

    if fps_value <= 0:
        return 0
    return int(round(frame_index * 1000.0 / fps_value))


def ms_to_timestamp(ms_value: int) -> str:
    """用于界面显示的 mm:ss 时间格式。"""

    total_seconds = max(0, int(round(ms_value / 1000.0)))
    minutes, seconds = divmod(total_seconds, 60)
    return "{:02d}:{:02d}".format(minutes, seconds)
