"""
教学视频预处理脚本。

功能:
1. 读取教学视频并提取标准姿态模板
2. 支持种子关键帧 + OpenCV 手动复核标记
3. 生成运行时直接可读的模板文件与关键帧配置文件

示例:
    python preprocess_video.py --video 跟练视频.MP4 --manual-review
    python preprocess_video.py --video 跟练视频.MP4 --seed-script 跟练动作脚本.json --accept-seed
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2

from config import ACTION_HINT_LIBRARY, LEVEL_INFO_FALLBACK, PATHS, SCORE_RULES, ensure_runtime_directories
from pose_utils import (
    PoseDetector,
    build_score_frame_payload,
    build_template_payload,
    extract_thumbnail,
    frame_index_to_ms,
    read_video_metadata,
    save_json,
)


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="为教学视频生成标准姿态模板与关键得分帧配置。")
    parser.add_argument("--video", type=Path, default=PATHS.video_path, help="教学视频路径")
    parser.add_argument(
        "--template-output",
        type=Path,
        default=PATHS.standard_template_path,
        help="标准姿态模板 JSON 输出路径",
    )
    parser.add_argument(
        "--score-output",
        type=Path,
        default=PATHS.score_frame_path,
        help="关键得分帧配置 JSON 输出路径",
    )
    parser.add_argument(
        "--thumbnail-output",
        type=Path,
        default=PATHS.thumbnail_path,
        help="准备页缩略图输出路径",
    )
    parser.add_argument("--action-name", type=str, default=LEVEL_INFO_FALLBACK["action_name"], help="动作名称")
    parser.add_argument("--subtitle", type=str, default=LEVEL_INFO_FALLBACK["subtitle"], help="动作副标题")
    parser.add_argument("--difficulty", type=str, default=LEVEL_INFO_FALLBACK["difficulty"], help="难度")
    parser.add_argument("--coach-name", type=str, default=LEVEL_INFO_FALLBACK["coach_name"], help="示范来源")
    parser.add_argument("--note", action="append", default=[], help="动作注意事项，可多次填写")
    parser.add_argument(
        "--seed-script",
        type=Path,
        default=None,
        help="可选，读取旧动作脚本并按区间中点生成关键帧种子",
    )
    parser.add_argument(
        "--auto-interval-sec",
        type=float,
        default=0.0,
        help="可选，按固定秒数间隔自动生成关键帧种子，例如 8 表示每 8 秒一个关键帧",
    )
    parser.add_argument(
        "--manual-review",
        action="store_true",
        help="使用 OpenCV 窗口手动复核关键帧，支持增删标记",
    )
    parser.add_argument(
        "--accept-seed",
        action="store_true",
        help="直接接受种子关键帧，不进入手动复核界面，便于自动化生成样例数据",
    )
    parser.add_argument(
        "--search-window-sec",
        type=float,
        default=12.0,
        help="在目标关键帧附近搜索可用标准姿态的窗口秒数，默认 12 秒",
    )
    parser.add_argument(
        "--search-step-frame",
        type=int,
        default=6,
        help="关键帧搜索时的步长帧数，默认 6 帧",
    )
    return parser.parse_args()


def load_seed_keyframes(seed_script_path: Path, fps_value: float) -> List[Dict[str, Any]]:
    """将旧动作区间脚本转换为关键帧种子。"""

    import json

    with seed_script_path.open("r", encoding="utf-8") as file:
        segments = json.load(file)

    seed_frames: List[Dict[str, Any]] = []
    for index, segment in enumerate(segments):
        start_sec = float(segment.get("start", 0.0))
        end_sec = float(segment.get("end", start_sec))
        midpoint_sec = max(start_sec, (start_sec + end_sec) * 0.5)
        frame_index = int(round(midpoint_sec * fps_value))
        action_name = str(segment.get("action", "default"))
        label = str(segment.get("label", "关键帧{}".format(index + 1)))
        seed_frames.append(
            {
                "frame_index": frame_index,
                "label": label,
                "action": action_name,
                "correction_hint": ACTION_HINT_LIBRARY.get(action_name, ACTION_HINT_LIBRARY["default"]),
            }
        )
    return seed_frames


def create_interval_seed_keyframes(
    video_duration_ms: int,
    fps_value: float,
    interval_seconds: float,
) -> List[Dict[str, Any]]:
    """按固定间隔自动生成关键帧种子。"""

    if interval_seconds <= 0 or fps_value <= 0:
        return []

    duration_seconds = video_duration_ms / 1000.0
    current_second = interval_seconds
    keyframes: List[Dict[str, Any]] = []
    index = 1
    while current_second < duration_seconds:
        keyframes.append(
            {
                "frame_index": int(round(current_second * fps_value)),
                "label": "关键帧 {:02d}".format(index),
                "action": "default",
                "correction_hint": ACTION_HINT_LIBRARY["default"],
            }
        )
        current_second += interval_seconds
        index += 1
    return keyframes


def manual_review_keyframes(
    video_path: Path,
    seed_keyframes: Sequence[Dict[str, Any]],
    fps_value: float,
) -> List[Dict[str, Any]]:
    """
    使用 OpenCV 窗口手动检查和修改关键帧。

    键位:
    A/D: 前后切帧
    J/L: 大步前后跳转
    M: 当前帧添加/删除关键帧
    S: 保存退出
    Q/ESC: 放弃退出
    """

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("无法打开视频，无法进行手动复核。")

    marked_map: Dict[int, Dict[str, Any]] = {int(item["frame_index"]): dict(item) for item in seed_keyframes}
    current_frame = min(marked_map.keys()) if marked_map else 0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while True:
        current_frame = max(0, min(frame_count - 1, current_frame))
        capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        success, frame = capture.read()
        if not success or frame is None:
            break

        display = frame.copy()
        is_marked = current_frame in marked_map
        timestamp_ms = frame_index_to_ms(current_frame, fps_value)
        overlay_lines = [
            "A/D: 单帧移动   J/L: 快速跳转   M: 标记/取消",
            "S: 保存并退出   ESC/Q: 放弃退出",
            "帧号: {}   时间: {:.2f}s".format(current_frame, timestamp_ms / 1000.0),
            "当前状态: {}".format("已标记关键帧" if is_marked else "未标记"),
        ]
        for line_index, text in enumerate(overlay_lines):
            cv2.putText(
                display,
                text,
                (28, 48 + line_index * 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (22, 93, 255),
                2,
                cv2.LINE_AA,
            )
        if is_marked:
            cv2.rectangle(display, (18, 18), (display.shape[1] - 18, display.shape[0] - 18), (54, 211, 153), 4)
        else:
            cv2.rectangle(display, (18, 18), (display.shape[1] - 18, display.shape[0] - 18), (248, 114, 114), 2)

        cv2.imshow("Preprocess Manual Review", display)
        key_value = cv2.waitKeyEx(0)

        if key_value in (ord("a"), ord("A"), 2424832):
            current_frame -= 1
        elif key_value in (ord("d"), ord("D"), 2555904):
            current_frame += 1
        elif key_value in (ord("j"), ord("J")):
            current_frame -= max(1, int(round(fps_value * 0.5)))
        elif key_value in (ord("l"), ord("L")):
            current_frame += max(1, int(round(fps_value * 0.5)))
        elif key_value in (ord("m"), ord("M")):
            if is_marked:
                marked_map.pop(current_frame, None)
            else:
                marker_index = len(marked_map) + 1
                marked_map[current_frame] = {
                    "frame_index": current_frame,
                    "label": "关键帧 {:02d}".format(marker_index),
                    "action": "default",
                    "correction_hint": ACTION_HINT_LIBRARY["default"],
                }
        elif key_value in (ord("s"), ord("S")):
            capture.release()
            cv2.destroyWindow("Preprocess Manual Review")
            return sorted(marked_map.values(), key=lambda item: int(item["frame_index"]))
        elif key_value in (27, ord("q"), ord("Q")):
            capture.release()
            cv2.destroyWindow("Preprocess Manual Review")
            raise RuntimeError("用户取消了手动关键帧标记。")

    capture.release()
    cv2.destroyAllWindows()
    return sorted(marked_map.values(), key=lambda item: int(item["frame_index"]))


def find_best_template_candidate(
    capture: cv2.VideoCapture,
    detector: PoseDetector,
    target_frame_index: int,
    fps_value: float,
    frame_count: int,
    search_window_sec: float,
    search_step_frame: int,
) -> Tuple[int, Any]:
    """在目标关键帧附近搜索最近且可用的标准姿态帧。"""

    search_radius = max(0, int(round(max(0.0, search_window_sec) * fps_value)))
    step_value = max(1, int(search_step_frame))
    candidate_offsets = [0]
    for offset in range(step_value, search_radius + step_value, step_value):
        candidate_offsets.extend([offset, -offset])

    for offset in candidate_offsets:
        candidate_index = max(0, min(frame_count - 1, target_frame_index + offset))
        capture.set(cv2.CAP_PROP_POS_FRAMES, candidate_index)
        success, frame = capture.read()
        if not success or frame is None:
            continue

        detection = detector.detect(frame)
        if not detection.visible or detection.normalized_vector is None:
            continue
        if detection.core_visibility < SCORE_RULES.min_core_visibility:
            continue
        return candidate_index, detection

    return -1, None


def extract_keyframe_templates(
    video_path: Path,
    keyframes: Sequence[Dict[str, Any]],
    fps_value: float,
    search_window_sec: float,
    search_step_frame: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """逐个关键帧提取标准姿态模板，并自动吸附到最近可用帧。"""

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("无法打开视频，不能提取标准姿态模板。")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    detector = PoseDetector(static_image_mode=True)
    templates: List[Dict[str, Any]] = []
    usable_keyframes: List[Dict[str, Any]] = []
    used_frame_map: Dict[int, str] = {}
    try:
        output_index = 0
        for keyframe in sorted(keyframes, key=lambda item: int(item["frame_index"])):
            frame_index = int(keyframe["frame_index"])
            matched_frame_index, detection = find_best_template_candidate(
                capture=capture,
                detector=detector,
                target_frame_index=frame_index,
                fps_value=fps_value,
                frame_count=frame_count,
                search_window_sec=search_window_sec,
                search_step_frame=search_step_frame,
            )
            if detection is None or matched_frame_index < 0:
                print(
                    "跳过关键帧 {}，原因: 在 +/- {:.1f} 秒范围内未搜索到可用标准姿态。".format(
                        frame_index,
                        search_window_sec,
                    )
                )
                continue

            if matched_frame_index in used_frame_map:
                print(
                    "跳过关键帧 {}，原因: 搜索命中了重复标准帧 {}。".format(
                        frame_index,
                        matched_frame_index,
                    )
                )
                continue

            output_index += 1
            used_frame_map[matched_frame_index] = str(keyframe.get("label", "关键帧"))
            template_id = "kf_{:03d}".format(output_index)
            if matched_frame_index != frame_index:
                print(
                    "关键帧 {} 自动吸附到可检测帧 {}。".format(
                        frame_index,
                        matched_frame_index,
                    )
                )

            templates.append(
                {
                    "template_id": template_id,
                    "frame_index": matched_frame_index,
                    "timestamp_ms": frame_index_to_ms(matched_frame_index, fps_value),
                    "action": keyframe.get("action", "default"),
                    "label": keyframe.get("label", "关键帧"),
                    "template_vector": [round(float(value), 6) for value in detection.normalized_vector.tolist()],
                    "core_visibility": round(float(detection.core_visibility), 4),
                }
            )
            usable_keyframes.append(
                {
                    "template_id": template_id,
                    "frame_index": matched_frame_index,
                    "timestamp_ms": frame_index_to_ms(matched_frame_index, fps_value),
                    "action": keyframe.get("action", "default"),
                    "label": keyframe.get("label", "关键帧"),
                    "correction_hint": keyframe.get(
                        "correction_hint",
                        ACTION_HINT_LIBRARY.get(str(keyframe.get("action", "default")), ACTION_HINT_LIBRARY["default"]),
                    ),
                    "pass_threshold": SCORE_RULES.pass_threshold,
                    "warn_threshold": SCORE_RULES.warn_threshold,
                }
            )
    finally:
        detector.close()
        capture.release()

    if not templates:
        raise RuntimeError("没有提取到可用模板，请重新标记关键帧或检查示范视频。")

    return templates, usable_keyframes


_FFMPEG_SEARCH_PATHS = [
    Path(r"D:\LosslessCut-win-x64\resources\ffmpeg.exe"),
    Path(r"D:\JianyingPro\8.1.1.12944\ffmpeg.exe"),
]


def _find_ffmpeg() -> str:
    """在系统 PATH 或已知路径中搜索 ffmpeg。"""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    for candidate in _FFMPEG_SEARCH_PATHS:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        "未找到 ffmpeg，请安装 ffmpeg 并加入 PATH，或编辑 preprocess_video.py 中的搜索路径。"
    )


def _extract_audio(video_path: Path, output_path: Path) -> None:
    """使用 ffmpeg 从视频提取 WAV 音频。"""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg, "-i", str(video_path),
        "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
        "-y", str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("警告: 音频提取失败，游戏将静音运行。\n" + result.stderr[-300:])
    else:
        print("音频已提取: {}".format(output_path))


def main() -> None:
    """预处理主流程。"""

    arguments = parse_arguments()
    ensure_runtime_directories()

    video_path = arguments.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError("未找到教学视频: {}".format(video_path))

    metadata = read_video_metadata(video_path)
    if not metadata["opened"]:
        raise RuntimeError("无法打开教学视频: {}".format(video_path))

    notes = arguments.note or list(LEVEL_INFO_FALLBACK["notes"])
    level_metadata = {
        "action_name": arguments.action_name,
        "subtitle": arguments.subtitle,
        "difficulty": arguments.difficulty,
        "coach_name": arguments.coach_name,
        "notes": notes,
    }

    fps_value = float(metadata["fps"])
    seed_keyframes: List[Dict[str, Any]] = []
    if arguments.seed_script:
        seed_keyframes.extend(load_seed_keyframes(arguments.seed_script.resolve(), fps_value))
    if arguments.auto_interval_sec > 0:
        seed_keyframes.extend(
            create_interval_seed_keyframes(
                video_duration_ms=int(metadata["duration_ms"]),
                fps_value=fps_value,
                interval_seconds=float(arguments.auto_interval_sec),
            )
        )

    unique_map: Dict[int, Dict[str, Any]] = {}
    for item in seed_keyframes:
        unique_map[int(item["frame_index"])] = item
    selected_keyframes = sorted(unique_map.values(), key=lambda item: int(item["frame_index"]))

    if arguments.manual_review and not arguments.accept_seed:
        selected_keyframes = manual_review_keyframes(video_path, selected_keyframes, fps_value)
    elif not selected_keyframes:
        raise RuntimeError("没有可用关键帧。请传入 --seed-script、--auto-interval-sec 或开启 --manual-review。")

    thumbnail_path = extract_thumbnail(video_path, arguments.thumbnail_output.resolve())
    if thumbnail_path is None:
        raise RuntimeError("无法从教学视频中提取缩略图。")

    templates, keyframes = extract_keyframe_templates(
        video_path,
        selected_keyframes,
        fps_value,
        search_window_sec=float(arguments.search_window_sec),
        search_step_frame=int(arguments.search_step_frame),
    )

    template_payload = build_template_payload(video_path, thumbnail_path, templates)
    score_payload = build_score_frame_payload(
        video_path=video_path,
        thumbnail_path=thumbnail_path,
        video_duration_ms=int(metadata["duration_ms"]),
        metadata=level_metadata,
        keyframes=keyframes,
    )

    save_json(template_payload, arguments.template_output.resolve())
    save_json(score_payload, arguments.score_output.resolve())

    audio_output = PATHS.audio_path
    _extract_audio(video_path, audio_output)

    print("预处理完成:")
    print("  标准模板: {}".format(arguments.template_output.resolve()))
    print("  关键帧配置: {}".format(arguments.score_output.resolve()))
    print("  缩略图: {}".format(thumbnail_path))
    print("  音频: {}".format(audio_output))
    print("  关键帧数量: {}".format(len(keyframes)))


if __name__ == "__main__":
    main()
