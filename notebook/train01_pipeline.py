from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import shutil
import subprocess

import cv2
import mediapipe as mp
import numpy as np

_MP_POSE = mp.solutions.pose

JOINT_NAME_TO_INDEX: Dict[str, int] = {
    "left_shoulder": _MP_POSE.PoseLandmark.LEFT_SHOULDER.value,
    "right_shoulder": _MP_POSE.PoseLandmark.RIGHT_SHOULDER.value,
    "left_elbow": _MP_POSE.PoseLandmark.LEFT_ELBOW.value,
    "right_elbow": _MP_POSE.PoseLandmark.RIGHT_ELBOW.value,
    "left_wrist": _MP_POSE.PoseLandmark.LEFT_WRIST.value,
    "right_wrist": _MP_POSE.PoseLandmark.RIGHT_WRIST.value,
    "left_hip": _MP_POSE.PoseLandmark.LEFT_HIP.value,
    "right_hip": _MP_POSE.PoseLandmark.RIGHT_HIP.value,
    "left_knee": _MP_POSE.PoseLandmark.LEFT_KNEE.value,
    "right_knee": _MP_POSE.PoseLandmark.RIGHT_KNEE.value,
    "left_ankle": _MP_POSE.PoseLandmark.LEFT_ANKLE.value,
    "right_ankle": _MP_POSE.PoseLandmark.RIGHT_ANKLE.value,
}
DEFAULT_FEATURE_JOINTS: Tuple[str, ...] = tuple(JOINT_NAME_TO_INDEX.keys())

ACTION_HINTS: Dict[str, str] = {
    # train00 上半身动作
    "arms_clap_up": "双臂再往上抬高一些，手腕尽量超过头顶",
    "arms_swing_down": "双臂向下甩出再干脆一些，保持对称",
    "left_arm_swing": "左臂向下甩出幅度再大一些，右臂保持稳定",
    "right_arm_swing": "右臂向下甩出幅度再大一些，左臂保持稳定",
    "left_arm_side": "左臂侧敲动作再明确一些，肘部带动手腕",
    "right_arm_side": "右臂侧敲动作再明确一些，肘部带动手腕",
    "arms_back_swing": "双臂向后甩出时肩部放松，幅度再打开一些",
    # train01 全身动作
    "arms_up": "双臂上举时再抬高一点，手腕尽量超过肩线",
    "single_arm_reach": "单侧抬手动作再明确一些，另一侧保持稳定",
    "arms_open": "双臂展开再打开一些，肩部不要耸起",
    "cross_body": "交叉收臂时再收紧一些，注意肩髋稳定",
    "left_knee_up": "左侧抬膝再明显一些，核心保持收紧",
    "right_knee_up": "右侧抬膝再明显一些，核心保持收紧",
    "wide_stance": "下肢站距再稳一些，注意膝盖方向",
    "side_lean": "身体侧倾幅度再清晰一些，重心保持稳定",
    "reset_pose": "回到准备姿态时保持稳定，不要提前松掉动作",
    "default": "动作幅度再靠近示范视频，保持姿态稳定",
}

ACTION_LABELS: Dict[str, str] = {
    # train00 上半身动作
    "arms_clap_up": "举臂拍手",
    "arms_swing_down": "双臂下甩",
    "left_arm_swing": "左臂下甩",
    "right_arm_swing": "右臂下甩",
    "left_arm_side": "左臂侧敲",
    "right_arm_side": "右臂侧敲",
    "arms_back_swing": "双臂后甩",
    # train01 全身动作
    "arms_up": "双臂上举",
    "single_arm_reach": "单侧上举",
    "arms_open": "双臂展开",
    "cross_body": "交叉收臂",
    "left_knee_up": "左侧抬膝",
    "right_knee_up": "右侧抬膝",
    "wide_stance": "下肢跨步",
    "side_lean": "身体侧倾",
    "reset_pose": "准备姿态",
    "default": "动作片段",
}


@dataclass(frozen=True)
class Train01Paths:
    video_path: Path
    template_path: Path
    score_frame_path: Path
    thumbnail_path: Path
    audio_path: Path
    manifest_path: Path


@dataclass(frozen=True)
class Train01FeatureConfig:
    joint_names: Tuple[str, ...] = DEFAULT_FEATURE_JOINTS
    joint_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "left_shoulder": 0.9,
            "right_shoulder": 0.9,
            "left_elbow": 1.1,
            "right_elbow": 1.1,
            "left_wrist": 1.25,
            "right_wrist": 1.25,
            "left_hip": 0.9,
            "right_hip": 0.9,
            "left_knee": 1.05,
            "right_knee": 1.05,
            "left_ankle": 1.15,
            "right_ankle": 1.15,
        }
    )
    visibility_threshold: float = 0.45
    min_feature_visibility: float = 0.60


@dataclass(frozen=True)
class Train01GenerationConfig:
    analysis_stride: int = 4
    resize_width: int = 960
    target_keyframes: int = 39
    min_keyframes: int = 30
    max_keyframes: int = 45
    min_gap_seconds: float = 4.5
    candidate_window_seconds: float = 2.2
    smoothing_alpha: float = 0.40
    pose_model_complexity: int = 1
    detection_confidence: float = 0.55
    tracking_confidence: float = 0.55
    pass_threshold: int = 82
    warn_threshold: int = 62
    thumbnail_pick_index: int = 0
    generation_version: str = "train01_auto_v1"
    source_note: str = "train01.mp4 真人动作自动抽取检测点并生成模板，可通过配置继续微调。"


@dataclass
class PoseSample:
    frame_index: int
    timestamp_ms: int
    vector: Optional[np.ndarray]
    feature_visibility: float
    point_map: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    missing_joints: List[str] = field(default_factory=list)
    motion_score: float = 0.0
    turning_score: float = 0.0
    pose_score: float = 0.0
    composite_score: float = 0.0


def build_train01_paths(base_dir: Path) -> Train01Paths:
    """兼容旧调用，等价于 build_paths(base_dir, 'train01')。"""
    return build_paths(base_dir, "train01")


def build_paths(base_dir: Path, prefix: str) -> Train01Paths:
    """根据前缀构建资源路径，支持 train00/train01 等多训练源。"""
    generated_dir = base_dir / "data" / "generated"
    return Train01Paths(
        video_path=base_dir / f"{prefix}.mp4",
        template_path=generated_dir / f"{prefix}_pose_templates.json",
        score_frame_path=generated_dir / f"{prefix}_score_frames.json",
        thumbnail_path=generated_dir / f"{prefix}_thumbnail.jpg",
        audio_path=generated_dir / f"{prefix}_audio.wav",
        manifest_path=generated_dir / f"{prefix}_generation_manifest.json",
    )


def validate_joint_names(joint_names: Sequence[str]) -> List[str]:
    valid_names = [name for name in joint_names if name in JOINT_NAME_TO_INDEX]
    return valid_names or list(DEFAULT_FEATURE_JOINTS)


def normalize_named_joints(
    landmarks: Sequence[Dict[str, float]],
    joint_names: Sequence[str],
    visibility_threshold: float = 0.45,
    joint_weights: Optional[Dict[str, float]] = None,
) -> Tuple[Optional[np.ndarray], float, List[str], Dict[str, Tuple[float, float]]]:
    if len(landmarks) <= max(JOINT_NAME_TO_INDEX.values()):
        missing = [name for name in joint_names]
        return None, 0.0, missing, {}

    left_shoulder = landmarks[JOINT_NAME_TO_INDEX["left_shoulder"]]
    right_shoulder = landmarks[JOINT_NAME_TO_INDEX["right_shoulder"]]
    left_hip = landmarks[JOINT_NAME_TO_INDEX["left_hip"]]
    right_hip = landmarks[JOINT_NAME_TO_INDEX["right_hip"]]

    shoulder_mid = np.array(
        [
            (float(left_shoulder["x"]) + float(right_shoulder["x"])) * 0.5,
            (float(left_shoulder["y"]) + float(right_shoulder["y"])) * 0.5,
        ],
        dtype=np.float32,
    )
    hip_mid = np.array(
        [
            (float(left_hip["x"]) + float(right_hip["x"])) * 0.5,
            (float(left_hip["y"]) + float(right_hip["y"])) * 0.5,
        ],
        dtype=np.float32,
    )
    trunk_length = float(np.linalg.norm(hip_mid - shoulder_mid))
    if trunk_length < 1e-6:
        return None, 0.0, [name for name in joint_names], {}

    normalized_points: List[float] = []
    point_map: Dict[str, Tuple[float, float]] = {}
    missing: List[str] = []
    visibility_scores: List[float] = []
    weights = joint_weights or {}

    for joint_name in joint_names:
        landmark = landmarks[JOINT_NAME_TO_INDEX[joint_name]]
        visibility = float(landmark.get("visibility", 1.0))
        visibility_scores.append(visibility)
        if visibility < visibility_threshold:
            missing.append(joint_name)
        x_value = (float(landmark["x"]) - float(shoulder_mid[0])) / trunk_length
        y_value = (float(landmark["y"]) - float(shoulder_mid[1])) / trunk_length
        point_map[joint_name] = (x_value, y_value)
        weight = float(weights.get(joint_name, 1.0))
        scaled_weight = max(weight, 1e-6) ** 0.5
        normalized_points.append(x_value * scaled_weight)
        normalized_points.append(y_value * scaled_weight)

    vector = np.array(normalized_points, dtype=np.float32)
    return vector, float(np.mean(visibility_scores)), missing, point_map


def _serialize_dataclass(data: Any) -> Any:
    if hasattr(data, "__dataclass_fields__"):
        return {
            key: _serialize_dataclass(value)
            for key, value in asdict(data).items()
        }
    if isinstance(data, dict):
        return {str(key): _serialize_dataclass(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_serialize_dataclass(value) for value in data]
    if isinstance(data, Path):
        return str(data)
    return data


def _signature_for_generation(
    video_path: Path,
    feature_config: Train01FeatureConfig,
    generation_config: Train01GenerationConfig,
) -> str:
    video_stat = video_path.stat()
    payload = {
        "video_path": str(video_path.resolve()),
        "video_size": int(video_stat.st_size),
        "video_mtime_ns": int(video_stat.st_mtime_ns),
        "feature_config": _serialize_dataclass(feature_config),
        "generation_config": _serialize_dataclass(generation_config),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return sha1(raw).hexdigest()[:16]


def _robust_scale(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return array
    low = float(np.percentile(array, 10))
    high = float(np.percentile(array, 90))
    if high - low < 1e-6:
        return np.ones_like(array, dtype=np.float32)
    scaled = (array - low) / (high - low)
    return np.clip(scaled, 0.0, 1.0)


def _classify_action_upper_body(point_map: Dict[str, Tuple[float, float]]) -> str:
    """仅使用上半身 6 关节（肩/肘/腕）进行动作分类，供 train00 等上半身配置使用。

    动作类别（对应用户描述）：
      arms_clap_up   — 胳膊向上提并拍手：双腕高举（y < -0.7），双肘也抬起
      left_arm_side  — 左胳膊侧敲：左腕高（y < -0.25），右腕低（y > 0.1）
      right_arm_side — 右胳膊侧敲：右腕高（y < -0.25），左腕低（y > 0.1）
      left_arm_swing — 左胳膊垂直甩出：左腕低垂（y > 0.4），右腕相对高
      right_arm_swing— 右胳膊垂直甩出：右腕低垂（y > 0.4），左腕相对高
      arms_swing_down— 胳膊垂直甩出（双臂）：双腕均低垂（y > 0.55），双肘也低
      arms_back_swing— 胳膊向后甩出：双腕处于中位（-0.3 ~ +0.4），双肘展开
    """
    lw = point_map["left_wrist"]
    rw = point_map["right_wrist"]
    le = point_map["left_elbow"]
    re = point_map["right_elbow"]

    wrist_span = abs(lw[0] - rw[0])
    elbow_width = abs(le[0] - re[0])

    # --- arms_clap_up: 双腕高举（y < -0.50），双肘抬起（y < 0） ---
    if lw[1] < -0.50 and rw[1] < -0.50 and le[1] < 0.0 and re[1] < 0.0:
        return "arms_clap_up"

    # --- left_arm_side: 左腕高（y < -0.25），右腕低或中（y > -0.20） ---
    if lw[1] < -0.25 and rw[1] > -0.20:
        return "left_arm_side"

    # --- right_arm_side: 右腕高（y < -0.25），左腕低或中（y > -0.20） ---
    if rw[1] < -0.25 and lw[1] > -0.20:
        return "right_arm_side"

    # --- arms_swing_down: 双腕均低垂（y > 0.55），双肘也低（y > 0.4） ---
    if lw[1] > 0.55 and rw[1] > 0.55 and le[1] > 0.40 and re[1] > 0.40:
        return "arms_swing_down"

    # --- left_arm_swing: 左腕低垂（y > 0.4），右腕相对高（y < 0.55） ---
    if lw[1] > 0.40 and rw[1] < 0.55:
        return "left_arm_swing"

    # --- right_arm_swing: 右腕低垂（y > 0.4），左腕相对高（y < 0.55） ---
    if rw[1] > 0.40 and lw[1] < 0.55:
        return "right_arm_swing"

    # --- arms_back_swing: 双腕中位（-0.3 ~ +0.4），双肘展开（肘距 > 0.5） ---
    both_wrists_mid = -0.30 < lw[1] < 0.40 and -0.30 < rw[1] < 0.40
    if both_wrists_mid and elbow_width > 0.50:
        return "arms_back_swing"

    return "default"


def _classify_action(point_map: Dict[str, Tuple[float, float]]) -> str:
    """基于归一化关节坐标判断当前姿态所属动作类别。

    阈值经过收紧以减少误分类：只有动作幅度足够明显时才归入具体类别，
    否则归为 default，避免中性姿态被错误标记。
    当 point_map 只含上半身 6 关节时，自动切换到上半身专用分类逻辑。
    """
    upper_body_joints = {"left_wrist", "right_wrist", "left_elbow", "right_elbow",
                         "left_shoulder", "right_shoulder"}
    full_body_required = {
        "left_wrist", "right_wrist", "left_elbow", "right_elbow",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
        "left_hip", "right_hip", "left_shoulder", "right_shoulder",
    }

    # 上半身配置：只有 6 个关节，切换到上半身分类器
    if upper_body_joints.issubset(point_map.keys()) and not full_body_required.issubset(point_map.keys()):
        return _classify_action_upper_body(point_map)

    if not full_body_required.issubset(point_map.keys()):
        return "default"

    lw = point_map["left_wrist"]
    rw = point_map["right_wrist"]
    le = point_map["left_elbow"]
    re = point_map["right_elbow"]
    lk = point_map["left_knee"]
    rk = point_map["right_knee"]
    la = point_map["left_ankle"]
    ra = point_map["right_ankle"]
    lh = point_map["left_hip"]
    rh = point_map["right_hip"]
    ls = point_map["left_shoulder"]
    rs = point_map["right_shoulder"]

    # 基础特征
    wrist_span = abs(lw[0] - rw[0])
    elbow_width = abs(le[0] - re[0])
    ankle_span = abs(la[0] - ra[0])
    hip_cx = (lh[0] + rh[0]) * 0.5
    shoulder_cx = (ls[0] + rs[0]) * 0.5
    wrist_cy = (lw[1] + rw[1]) * 0.5

    # --- arms_up: 双腕都明显高于肩线（y < -0.25），且双肘也抬起 ---
    both_wrists_high = lw[1] < -0.25 and rw[1] < -0.25
    both_elbows_raised = le[1] < 0.0 and re[1] < 0.0
    if both_wrists_high and both_elbows_raised:
        return "arms_up"

    # --- single_arm_reach: 一侧腕明显高于肩线，另一侧明显低 ---
    left_high_right_low = lw[1] < -0.30 and rw[1] > 0.20
    right_high_left_low = rw[1] < -0.30 and lw[1] > 0.20
    if left_high_right_low or right_high_left_low:
        return "single_arm_reach"

    # --- arms_open: 双臂大幅展开，腕距和肘距都很大 ---
    if wrist_span > 1.30 and elbow_width > 0.80:
        return "arms_open"

    # --- cross_body: 双腕交叉（左腕在右侧、右腕在左侧），且腕距小 ---
    wrists_crossed = lw[0] > 0.10 and rw[0] < -0.10
    if wrists_crossed and wrist_span < 0.40:
        return "cross_body"

    # --- left_knee_up / right_knee_up: 膝盖明显高于同侧髋关节 ---
    if rh[1] - lk[1] > 0.50:
        return "left_knee_up"
    if lh[1] - rk[1] > 0.50:
        return "right_knee_up"

    # --- wide_stance: 踝距很大，且上半身相对居中 ---
    wrist_center_offset = abs((lw[0] + rw[0]) * 0.5)
    if ankle_span > 1.10 and wrist_center_offset < 0.20 and wrist_cy > 0.10:
        return "wide_stance"

    # --- side_lean: 肩髋中心明显偏移，或双肩高度差大 ---
    torso_offset = abs(shoulder_cx - hip_cx)
    shoulder_tilt = abs(ls[1] - rs[1])
    if torso_offset > 0.28 and shoulder_tilt > 0.12:
        return "side_lean"

    # --- reset_pose: 双腕低垂且靠近身体中线 ---
    if wrist_span < 0.35 and lw[1] > 0.35 and rw[1] > 0.35 and wrist_center_offset < 0.15:
        return "reset_pose"

    return "default"


def _extract_ffmpeg_path() -> Optional[str]:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    try:
        import imageio_ffmpeg  # type: ignore

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def extract_audio_to_wav(video_path: Path, audio_path: Path) -> Dict[str, Any]:
    ffmpeg_path = _extract_ffmpeg_path()
    if not ffmpeg_path:
        return {
            "ok": False,
            "message": "未找到 ffmpeg 或 imageio_ffmpeg，暂时无法自动提取音频。",
        }

    audio_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        "44100",
        str(audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not audio_path.exists() or audio_path.stat().st_size <= 0:
        message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "音频提取失败。"
        return {"ok": False, "message": message}
    return {"ok": True, "message": f"已提取 {audio_path.name}"}


def _pick_thumbnail(video_path: Path, thumbnail_path: Path, selected_samples: Sequence[PoseSample], pick_index: int) -> None:
    if not selected_samples:
        return
    sample_index = min(max(int(pick_index), 0), len(selected_samples) - 1)
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(selected_samples[sample_index].frame_index))
        success, frame = capture.read()
        if success and frame is not None:
            thumbnail_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(thumbnail_path), frame)
    finally:
        capture.release()


def _build_label(action_key: str, action_counter: Counter[str]) -> str:
    action_counter[action_key] += 1
    chinese_label = ACTION_LABELS.get(action_key, ACTION_LABELS["default"])
    return f"{chinese_label} {action_counter[action_key]}"


def analyze_train01_video(
    video_path: Path,
    feature_config: Train01FeatureConfig,
    generation_config: Train01GenerationConfig,
) -> Tuple[List[PoseSample], Dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开示范视频：{video_path}")

    fps_value = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    pose = _MP_POSE.Pose(
        static_image_mode=False,
        model_complexity=generation_config.pose_model_complexity,
        enable_segmentation=False,
        min_detection_confidence=generation_config.detection_confidence,
        min_tracking_confidence=generation_config.tracking_confidence,
    )

    samples: List[PoseSample] = []
    smoothed_state: Optional[np.ndarray] = None
    frame_index = -1

    try:
        while True:
            success, frame = capture.read()
            if not success or frame is None:
                break
            frame_index += 1
            if frame_index % max(1, generation_config.analysis_stride) != 0:
                continue

            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC) or round((frame_index / max(fps_value, 1.0)) * 1000.0))
            processing_frame = frame
            if generation_config.resize_width > 0 and frame.shape[1] > generation_config.resize_width:
                target_width = generation_config.resize_width
                target_height = int(frame.shape[0] * target_width / frame.shape[1])
                processing_frame = cv2.resize(frame, (target_width, target_height))

            rgb_frame = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)
            if not result.pose_landmarks:
                smoothed_state = None
                samples.append(PoseSample(frame_index=frame_index, timestamp_ms=timestamp_ms, vector=None, feature_visibility=0.0))
                continue

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

            vector, feature_visibility, missing, point_map = normalize_named_joints(
                landmarks,
                feature_config.joint_names,
                visibility_threshold=feature_config.visibility_threshold,
                joint_weights=feature_config.joint_weights,
            )

            if vector is None:
                smoothed_state = None
            elif smoothed_state is None:
                smoothed_state = vector.astype(np.float32)
            else:
                alpha = float(generation_config.smoothing_alpha)
                smoothed_state = alpha * smoothed_state + (1.0 - alpha) * vector.astype(np.float32)

            samples.append(
                PoseSample(
                    frame_index=frame_index,
                    timestamp_ms=timestamp_ms,
                    vector=None if smoothed_state is None else smoothed_state.copy(),
                    feature_visibility=feature_visibility,
                    point_map=point_map,
                    missing_joints=missing,
                )
            )
    finally:
        capture.release()
        pose.close()

    valid_samples = [sample for sample in samples if sample.vector is not None]
    if not valid_samples:
        raise RuntimeError("train01.mp4 中未检测到可用人体姿态，请先确认视频中人物完整入镜。")

    matrix = np.stack([sample.vector for sample in valid_samples if sample.vector is not None], axis=0)
    median_vector = np.median(matrix, axis=0)

    motion_values: List[float] = []
    turning_values: List[float] = []
    pose_values: List[float] = []
    visibility_values: List[float] = []

    for index, sample in enumerate(valid_samples):
        current_vector = np.asarray(sample.vector, dtype=np.float32)
        prev_vector = np.asarray(valid_samples[index - 1].vector, dtype=np.float32) if index > 0 else current_vector
        next_vector = np.asarray(valid_samples[index + 1].vector, dtype=np.float32) if index + 1 < len(valid_samples) else current_vector
        motion_score = float(np.linalg.norm(current_vector - prev_vector) + np.linalg.norm(next_vector - current_vector))
        turning_score = float(np.linalg.norm((current_vector - prev_vector) - (next_vector - current_vector)))
        pose_score = float(np.linalg.norm(current_vector - median_vector))
        sample.motion_score = motion_score
        sample.turning_score = turning_score
        sample.pose_score = pose_score
        motion_values.append(motion_score)
        turning_values.append(turning_score)
        pose_values.append(pose_score)
        visibility_values.append(float(sample.feature_visibility))

    motion_scaled = _robust_scale(motion_values)
    turning_scaled = _robust_scale(turning_values)
    pose_scaled = _robust_scale(pose_values)
    visibility_scaled = _robust_scale(visibility_values)

    for index, sample in enumerate(valid_samples):
        sample.composite_score = float(
            0.35 * motion_scaled[index]
            + 0.25 * turning_scaled[index]
            + 0.25 * pose_scaled[index]
            + 0.15 * visibility_scaled[index]
        )

    metadata = {
        "fps": fps_value,
        "frame_count": frame_count,
        "duration_ms": int(round(frame_count / max(fps_value, 1.0) * 1000.0)),
        "width": width,
        "height": height,
        "valid_sample_count": len(valid_samples),
        "analysis_stride": generation_config.analysis_stride,
    }
    return valid_samples, metadata


def select_detection_points(
    samples: Sequence[PoseSample],
    generation_config: Train01GenerationConfig,
) -> List[PoseSample]:
    """从分析样本中选取关键检测点。

    改进：在去重阶段加入动作多样性约束，连续相同动作类型的候选点
    会被替换为附近不同动作类型的候选点（如果存在），避免同类动作扎堆。
    """
    if not samples:
        return []

    # 预计算每个样本的动作分类
    sample_actions: List[str] = []
    for sample in samples:
        sample_actions.append(_classify_action(sample.point_map) if sample.point_map else "default")

    duration_ms = max(sample.timestamp_ms for sample in samples) if samples else 0
    target_count = max(generation_config.min_keyframes, min(generation_config.target_keyframes, generation_config.max_keyframes))
    min_gap_ms = int(generation_config.min_gap_seconds * 1000.0)

    scores = np.asarray([max(sample.composite_score, 0.01) for sample in samples], dtype=np.float32)
    cumulative = np.cumsum(scores)
    if cumulative.size == 0:
        return []

    if len(samples) > 1:
        sample_interval_ms = int(np.median(np.diff([sample.timestamp_ms for sample in samples])))
    else:
        sample_interval_ms = 100
    window_radius = max(1, int(round((generation_config.candidate_window_seconds * 1000.0) / max(sample_interval_ms, 1))))

    tentative_indices: List[int] = []
    targets = np.linspace(float(cumulative[0]), float(cumulative[-1]), target_count + 2)[1:-1]
    for target in targets:
        center = int(np.searchsorted(cumulative, target))
        start = max(0, center - window_radius)
        end = min(len(samples), center + window_radius + 1)
        best_index = max(
            range(start, end),
            key=lambda item: (
                samples[item].composite_score,
                samples[item].pose_score,
                samples[item].feature_visibility,
            ),
        )
        tentative_indices.append(best_index)

    # 去重 + 动作多样性约束
    MAX_CONSECUTIVE_SAME_ACTION = 2
    deduped_indices: List[int] = []
    for index in sorted(set(tentative_indices), key=lambda item: samples[item].timestamp_ms):
        if not deduped_indices:
            deduped_indices.append(index)
            continue
        previous_index = deduped_indices[-1]
        if samples[index].timestamp_ms - samples[previous_index].timestamp_ms < min_gap_ms:
            better_index = index if samples[index].composite_score > samples[previous_index].composite_score else previous_index
            deduped_indices[-1] = better_index
        else:
            # 检查连续相同动作：如果最近 N 个都是同一动作，尝试替换
            current_action = sample_actions[index]
            recent_actions = [sample_actions[i] for i in deduped_indices[-MAX_CONSECUTIVE_SAME_ACTION:]]
            if len(recent_actions) >= MAX_CONSECUTIVE_SAME_ACTION and all(a == current_action for a in recent_actions) and current_action != "default":
                # 在附近窗口找一个不同动作的候选
                search_start = max(0, index - window_radius * 2)
                search_end = min(len(samples), index + window_radius * 2 + 1)
                alt_candidates = [
                    j for j in range(search_start, search_end)
                    if sample_actions[j] != current_action
                    and samples[j].composite_score > 0.15
                    and all(abs(samples[j].timestamp_ms - samples[k].timestamp_ms) >= min_gap_ms for k in deduped_indices)
                ]
                if alt_candidates:
                    alt_index = max(alt_candidates, key=lambda j: samples[j].composite_score)
                    deduped_indices.append(alt_index)
                else:
                    deduped_indices.append(index)
            else:
                deduped_indices.append(index)

    if len(deduped_indices) < generation_config.min_keyframes:
        remaining = sorted(
            [index for index in range(len(samples)) if index not in deduped_indices],
            key=lambda item: samples[item].composite_score,
            reverse=True,
        )
        for index in remaining:
            if all(abs(samples[index].timestamp_ms - samples[kept].timestamp_ms) >= min_gap_ms * 0.75 for kept in deduped_indices):
                deduped_indices.append(index)
            if len(deduped_indices) >= generation_config.min_keyframes:
                break

    deduped_indices = sorted(deduped_indices, key=lambda item: samples[item].timestamp_ms)
    if len(deduped_indices) > generation_config.max_keyframes:
        ranked = sorted(deduped_indices, key=lambda item: samples[item].composite_score, reverse=True)[: generation_config.max_keyframes]
        deduped_indices = sorted(ranked, key=lambda item: samples[item].timestamp_ms)

    if len(deduped_indices) < generation_config.min_keyframes and duration_ms > 0:
        evenly_spaced = np.linspace(0.0, float(duration_ms), generation_config.min_keyframes + 2)[1:-1]
        for target_ms in evenly_spaced:
            best_index = min(range(len(samples)), key=lambda item: abs(samples[item].timestamp_ms - target_ms))
            if all(abs(samples[best_index].timestamp_ms - samples[kept].timestamp_ms) >= min_gap_ms * 0.5 for kept in deduped_indices):
                deduped_indices.append(best_index)
        deduped_indices = sorted(set(deduped_indices), key=lambda item: samples[item].timestamp_ms)

    return [samples[index] for index in deduped_indices]


def generate_train01_assets(
    paths: Train01Paths,
    feature_config: Train01FeatureConfig,
    generation_config: Train01GenerationConfig,
    prefix: str = "train01",
    display_name: str = "",
) -> Dict[str, Any]:
    """通用资源生成函数，prefix 决定模板 ID 前缀和文案中的名称。"""
    if not display_name:
        display_name = f"{prefix.capitalize()} 真人动作跟练"
    valid_joint_names = validate_joint_names(feature_config.joint_names)
    normalized_feature_config = Train01FeatureConfig(
        joint_names=tuple(valid_joint_names),
        joint_weights={name: float(feature_config.joint_weights.get(name, 1.0)) for name in valid_joint_names},
        visibility_threshold=feature_config.visibility_threshold,
        min_feature_visibility=feature_config.min_feature_visibility,
    )
    generation_signature = _signature_for_generation(paths.video_path, normalized_feature_config, generation_config)

    samples, video_meta = analyze_train01_video(paths.video_path, normalized_feature_config, generation_config)
    selected_samples = [
        sample
        for sample in select_detection_points(samples, generation_config)
        if sample.feature_visibility >= normalized_feature_config.min_feature_visibility
    ]
    if len(selected_samples) < generation_config.min_keyframes:
        additional_samples = sorted(samples, key=lambda item: item.composite_score, reverse=True)
        for sample in additional_samples:
            if sample.feature_visibility < normalized_feature_config.min_feature_visibility:
                continue
            if sample in selected_samples:
                continue
            if all(abs(sample.timestamp_ms - kept.timestamp_ms) >= int(generation_config.min_gap_seconds * 700) for kept in selected_samples):
                selected_samples.append(sample)
            if len(selected_samples) >= generation_config.min_keyframes:
                break

    selected_samples = sorted(selected_samples, key=lambda item: item.timestamp_ms)
    if not selected_samples:
        raise RuntimeError(f"自动抽点失败，未能从 {paths.video_path.name} 中筛出有效检测点。")

    action_counter: Counter[str] = Counter()
    templates: List[Dict[str, Any]] = []
    keyframes: List[Dict[str, Any]] = []
    manifest_keyframes: List[Dict[str, Any]] = []

    for index, sample in enumerate(selected_samples, start=1):
        action_key = _classify_action(sample.point_map)
        label = _build_label(action_key, action_counter)
        template_id = f"{prefix}_kf_{index:03d}"
        hint = ACTION_HINTS.get(action_key, ACTION_HINTS["default"])
        vector = np.asarray(sample.vector, dtype=np.float32) if sample.vector is not None else np.zeros(len(valid_joint_names) * 2, dtype=np.float32)
        template_vector = [round(float(value), 6) for value in vector.tolist()]
        templates.append(
            {
                "template_id": template_id,
                "frame_index": int(sample.frame_index),
                "timestamp_ms": int(sample.timestamp_ms),
                "action": action_key,
                "label": label,
                "template_vector": template_vector,
                "core_visibility": round(float(sample.feature_visibility), 4),
                "template_source": f"{prefix}_auto_motion",
                "template_source_frame_index": int(sample.frame_index),
                "template_source_timestamp_ms": int(sample.timestamp_ms),
                "template_source_note": generation_config.source_note,
                "motion_score": round(float(sample.motion_score), 6),
                "turning_score": round(float(sample.turning_score), 6),
                "pose_score": round(float(sample.pose_score), 6),
                "composite_score": round(float(sample.composite_score), 6),
                "missing_joints": list(sample.missing_joints),
            }
        )
        keyframes.append(
            {
                "template_id": template_id,
                "frame_index": int(sample.frame_index),
                "timestamp_ms": int(sample.timestamp_ms),
                "action": action_key,
                "label": label,
                "correction_hint": hint,
                "pass_threshold": int(generation_config.pass_threshold),
                "warn_threshold": int(generation_config.warn_threshold),
                "template_source": f"{prefix}_auto_motion",
                "template_source_note": generation_config.source_note,
            }
        )
        manifest_keyframes.append(
            {
                "template_id": template_id,
                "frame_index": int(sample.frame_index),
                "timestamp_ms": int(sample.timestamp_ms),
                "action": action_key,
                "label": label,
                "feature_visibility": round(float(sample.feature_visibility), 4),
                "motion_score": round(float(sample.motion_score), 6),
                "turning_score": round(float(sample.turning_score), 6),
                "pose_score": round(float(sample.pose_score), 6),
                "composite_score": round(float(sample.composite_score), 6),
            }
        )

    _pick_thumbnail(paths.video_path, paths.thumbnail_path, selected_samples, generation_config.thumbnail_pick_index)
    audio_result = extract_audio_to_wav(paths.video_path, paths.audio_path)

    video_name = paths.video_path.name
    action_name = display_name
    subtitle = f"检测点由 {video_name} 自动抽取，实时评分与示范视频共用同一时间轴"
    notes = [
        "检测点根据视频人体动作变化自动抽取，含动作多样性约束。",
        f"可调整 FeatureConfig 与 GenerationConfig 后重新生成模板。",
        f"实时跟练同步 {video_name}，评分采用余弦+欧氏混合方案。",
    ]

    template_payload = {
        "video_path": paths.video_path.name,
        "thumbnail_path": str(paths.thumbnail_path.relative_to(paths.video_path.parent)),
        "joint_names": list(valid_joint_names),
        "joint_weights": {name: round(float(normalized_feature_config.joint_weights.get(name, 1.0)), 4) for name in valid_joint_names},
        "generation_signature": generation_signature,
        "generation_config": _serialize_dataclass(generation_config),
        "feature_config": _serialize_dataclass(normalized_feature_config),
        "templates": templates,
    }
    score_payload = {
        "video_path": paths.video_path.name,
        "thumbnail_path": str(paths.thumbnail_path.relative_to(paths.video_path.parent)),
        "video_duration_ms": int(video_meta["duration_ms"]),
        "action_name": action_name,
        "subtitle": subtitle,
        "difficulty": "中等",
        "coach_name": f"{prefix} 自动模板",
        "notes": notes,
        "generation_signature": generation_signature,
        "generation_config": _serialize_dataclass(generation_config),
        "feature_config": _serialize_dataclass(normalized_feature_config),
        "keyframes": keyframes,
    }
    manifest_payload = {
        "video_path": paths.video_path.name,
        "template_path": paths.template_path.name,
        "score_frame_path": paths.score_frame_path.name,
        "thumbnail_path": paths.thumbnail_path.name,
        "audio_path": paths.audio_path.name,
        "generation_signature": generation_signature,
        "video_meta": video_meta,
        "generation_config": _serialize_dataclass(generation_config),
        "feature_config": _serialize_dataclass(normalized_feature_config),
        "keyframe_count": len(keyframes),
        "action_breakdown": dict(action_counter),
        "audio_status": audio_result,
        "keyframes": manifest_keyframes,
    }

    paths.template_path.parent.mkdir(parents=True, exist_ok=True)
    paths.template_path.write_text(json.dumps(template_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.score_frame_path.write_text(json.dumps(score_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "keyframe_count": len(keyframes),
        "generation_signature": generation_signature,
        "video_meta": video_meta,
        "joint_names": list(valid_joint_names),
        "joint_weights": {name: float(normalized_feature_config.joint_weights.get(name, 1.0)) for name in valid_joint_names},
        "audio_status": audio_result,
        "action_breakdown": dict(action_counter),
        "files": {
            "template_path": str(paths.template_path),
            "score_frame_path": str(paths.score_frame_path),
            "thumbnail_path": str(paths.thumbnail_path),
            "audio_path": str(paths.audio_path),
            "manifest_path": str(paths.manifest_path),
        },
    }


def ensure_train01_assets(
    base_dir: Path,
    feature_config: Train01FeatureConfig,
    generation_config: Train01GenerationConfig,
    force: bool = False,
    prefix: str = "train01",
    display_name: str = "",
) -> Dict[str, Any]:
    paths = build_paths(base_dir, prefix)
    if not paths.video_path.exists():
        raise RuntimeError(f"未找到 {prefix}.mp4：{paths.video_path}")

    expected_signature = _signature_for_generation(paths.video_path, feature_config, generation_config)
    manifest_payload: Dict[str, Any] = {}
    if paths.manifest_path.exists():
        try:
            manifest_payload = json.loads(paths.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest_payload = {}

    files_ready = all(
        path.exists()
        for path in [
            paths.template_path,
            paths.score_frame_path,
            paths.thumbnail_path,
            paths.manifest_path,
        ]
    )
    signature_match = manifest_payload.get("generation_signature") == expected_signature
    audio_ready = paths.audio_path.exists()

    if force or not files_ready or not signature_match:
        return generate_train01_assets(paths, feature_config, generation_config, prefix=prefix, display_name=display_name)

    if not audio_ready:
        audio_status = extract_audio_to_wav(paths.video_path, paths.audio_path)
        manifest_payload["audio_status"] = audio_status
        paths.manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "keyframe_count": int(manifest_payload.get("keyframe_count", 0)),
        "generation_signature": str(manifest_payload.get("generation_signature", "")),
        "video_meta": dict(manifest_payload.get("video_meta", {})),
        "joint_names": list(manifest_payload.get("feature_config", {}).get("joint_names", list(feature_config.joint_names))),
        "joint_weights": dict(manifest_payload.get("feature_config", {}).get("joint_weights", feature_config.joint_weights)),
        "audio_status": dict(manifest_payload.get("audio_status", {})),
        "action_breakdown": dict(manifest_payload.get("action_breakdown", {})),
        "files": {
            "template_path": str(paths.template_path),
            "score_frame_path": str(paths.score_frame_path),
            "thumbnail_path": str(paths.thumbnail_path),
            "audio_path": str(paths.audio_path),
            "manifest_path": str(paths.manifest_path),
        },
    }
