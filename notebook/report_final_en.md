# RhythmMotionCV: A Rhythm-Based Motion Follow-Along System Using Monocular RGB Video

## Abstract

RhythmMotionCV is a lightweight rhythm-based motion follow-along system designed for office workers who sit for long periods. It only requires a standard RGB camera to run. Users can upload any demo video and follow along in a game-like scoring mode, completing fun physical exercises during short breaks — no special equipment or dedicated space is needed. The system uses MediaPipe BlazePose for real-time human pose estimation. It follows a two-stage pipeline: an offline stage for automatic keyframe extraction, and an online stage for template matching and scoring. This creates a complete interactive loop where users follow a demo video and receive real-time feedback. In the offline stage, a four-dimensional composite score — based on motion magnitude, turning degree, pose deviation, and visibility — is used to automatically select high-quality keyframes as action templates. A motion diversity constraint is also applied to avoid selecting too many frames of the same action type. In the online scoring stage, a hybrid similarity metric (60% cosine similarity + 40% weighted Euclidean distance) is used, together with automatic mirror comparison and a combo multiplier, to produce a scoring system with good discrimination. The system supports switching between multiple training sources (6-joint upper body / 12-joint full body) and has been fully tested on two training sources: train00 (95.8s, 11 checkpoints, 6 action types) and train01 (329.5s, 31 checkpoints, 7 action types).

## 1. Introduction

Sitting for long hours is a common health problem for modern office workers. The high cost of gym memberships, fixed exercise schedules, and the need for special equipment make it hard for most people to exercise regularly. Popular fitness apps (such as Keep and Huawei Health) provide demo videos, but users can only watch passively. They have no way to know whether their own movements are correct, because there is no real-time pose detection or feedback. Professional motion capture systems can provide accurate feedback, but they are expensive and complex to set up, making them impractical for everyday use. As lightweight vision models and consumer-grade cameras become more widely available, monocular RGB video-based pose interaction systems offer a low-barrier alternative: with just a laptop camera, anyone can exercise at their desk, at home, or anywhere, and receive real-time pose comparison and scoring feedback. Compared to systems that rely on wearable sensors or depth cameras, a pure vision approach is cheaper to deploy and easier to use, making it suitable for a general audience.

The goal of this project is to build a complete rhythm-based motion follow-along system. The user faces a standard RGB camera and follows a demo video. The system detects the user's pose in real time and gives a score. To achieve this, the system needs to solve the following engineering problems: (1) how to automatically extract high-quality action templates from the demo video, rather than using arbitrary frames of varying quality; (2) how to design a scoring method that considers both overall pose direction and local joint errors; (3) how to handle the mirror effect when the user faces the camera; (4) how to synchronize video playback, audio playback, and score triggering.

RhythmMotionCV is built on three core technologies: MediaPipe BlazePose for pose estimation, template matching for scoring, and PyQt5 for the desktop interface. The main work includes: (1) designing a four-dimensional composite scoring algorithm for keyframe selection, based on motion magnitude, turning degree, pose deviation, and visibility, with a motion diversity constraint; (2) implementing a hybrid scoring method using cosine similarity and weighted Euclidean distance (6:4 weight), with automatic mirror comparison and a combo multiplier; (3) building a complete interactive system that supports multiple training sources (6-joint upper body / 12-joint full body), with a three-page flow: preparation page, live follow-along page, and results page.

## 2. Technical Background and Technology Selection

**Pose Estimation: MediaPipe BlazePose.** This project uses Google's MediaPipe BlazePose [3] as the pose estimation backend. BlazePose uses a two-stage detection-regression architecture that can run in real time on a CPU. It outputs 33 three-dimensional keypoints with visibility confidence scores, which balances accuracy and ease of deployment. Compared to OpenPose [1] (a bottom-up multi-person detector with high computational cost) and HRNet [2] (high-resolution feature representation that requires a GPU), BlazePose is better suited for this project's single-person, consumer-hardware setting.

**Action Scoring: Template Matching.** Common methods for action quality assessment include skeleton sequence modeling with graph convolutional networks (e.g., ST-GCN [4]) and frame-by-frame template matching. The former requires large amounts of labeled data and a training process. The latter computes a similarity score between the user's pose and a stored template, which is simple to implement and easy to interpret. This project uses template matching with a hybrid metric combining cosine similarity and weighted Euclidean distance, which keeps computation efficient while capturing both overall pose direction and local joint errors.

**User Interface: PyQt5 Desktop Application.** The system needs to handle video playback, camera capture, pose detection, and scoring feedback at the same time, which requires strong real-time rendering and multimedia support. PyQt5 provides a mature timer mechanism (QTimer), multimedia components (QSoundEffect), and a flexible layout system, making it well suited for building this kind of real-time interactive desktop application.

## 3. System Overview

RhythmMotionCV uses a two-stage architecture: offline preprocessing and online real-time interaction.

**Offline Stage (Pipeline).** Given a demo video (e.g., `train01.mp4`), the system samples frames with a stride of `analysis_stride=4`. It uses MediaPipe BlazePose (`model_complexity=1`, `detection_confidence=0.55`, `tracking_confidence=0.55`) to extract human poses, computes a normalized feature vector for each frame, and applies exponential moving average smoothing (`smoothing_alpha=0.40`). It then computes a four-dimensional composite score (motion magnitude 0.35, turning degree 0.25, pose deviation 0.25, visibility 0.15), and selects keyframe checkpoints through uniform sampling on the cumulative score distribution, local-best candidate window selection, and a motion diversity constraint. The final outputs are a pose template JSON, a score frame config JSON, a thumbnail, and an audio file.

**Online Stage (Notebook PyQt UI).** After launch, the system enters a three-page flow. The preparation page shows training source information and a cover thumbnail, with a dropdown to switch between training sources (train00/train01). The live follow-along page plays the demo video and camera feed side by side, and automatically triggers scoring at each keyframe time. The results page shows the final score, grade, and statistics. The online stage uses `time.perf_counter()` to maintain a unified timeline. Video playback, audio playback, and score triggering all share the same time reference. Score triggering fires `score_trigger_lookahead_ms=40ms` before the keyframe time.

**Multiple Training Source Support.** The system manages training sources through a `TRAIN_SOURCES` registry. Each source carries its own feature configuration (`FeatureConfig`) and generation parameter overrides. train00 uses a 6-joint upper-body configuration (both shoulders, elbows, and wrists). train01 uses a 12-joint full-body configuration (adding both hips, knees, and ankles). When the user switches sources, the system automatically reloads resources and updates scoring parameters.

## 4. Method

### 4.1 Keyframe Selection Algorithm

Many frames in a demo video are in transition between actions — the pose is unstable or joint visibility is low, making them unsuitable as scoring templates. The system therefore uses motion analysis to automatically filter for high-quality pose frames, using a four-dimensional composite score for keyframe selection.

**Four-Dimensional Composite Score.** For each valid sampled frame, the system computes scores on four dimensions:

- **Motion magnitude (motion_score, weight 0.35)**: the sum of vector difference norms between the current frame and its neighbors, reflecting how much movement is happening around this frame.
- **Turning degree (turning_score, weight 0.25)**: the norm of the change in motion direction between neighboring frames, capturing action turning points.
- **Pose deviation (pose_score, weight 0.25)**: the distance between the current frame's pose and the global median pose, favoring frames with distinctive, non-neutral poses.
- **Visibility (feature_visibility, weight 0.15)**: the average visibility confidence of the feature joints, ensuring template quality.

Each dimension is mapped to [0, 1] using robust scaling (10–90 percentile normalization), then combined by weighted sum to get `composite_score`.

**Candidate Window Selection.** The system uniformly samples `target_keyframes` target points on the cumulative distribution of composite scores (train00: 15, train01: 39). For each target point, it selects the frame with the highest composite score within a `candidate_window_seconds` window (train00: 2.5s, train01: 2.2s) as the candidate.

**Deduplication and Diversity Constraint.** After sorting candidates by time, if two adjacent candidates are closer than `min_gap_seconds` (train00: 5.0s, train01: 4.5s), only the one with the higher composite score is kept. In addition, a motion diversity constraint (`MAX_CONSECUTIVE_SAME_ACTION=2`) is applied: if the last two consecutive candidates belong to the same action category, the system searches a nearby window for a candidate from a different category, avoiding repetitive action sequences.

**Action Classification.** The system classifies each frame into one of 9 action categories using a threshold-based decision tree on normalized joint coordinates: arms raised (`arms_up`, both wrists y < -0.25 and both elbows y < 0), single arm reach (`single_arm_reach`, one wrist y < -0.30 and the other > 0.20), arms open (`arms_open`, wrist distance > 1.30 and elbow distance > 0.80), cross body (`cross_body`, wrists crossed and distance < 0.40), left/right knee up (`left/right_knee_up`, knee-hip height difference > 0.50), wide stance (`wide_stance`, ankle distance > 1.10), side lean (`side_lean`, torso offset > 0.28 and shoulder tilt > 0.12), reset pose (`reset_pose`, both wrists low and near the center line), and default (`default`).

### 4.2 Pose Representation and Normalization

The system selects a subset of feature joints from the 33 three-dimensional keypoints output by MediaPipe BlazePose, based on the training source configuration. train00 uses 6 upper-body joints (left/right shoulder, left/right elbow, left/right wrist). train01 uses 12 full-body joints (adding left/right hip, left/right knee, left/right ankle).

**Trunk Length Normalization.** The midpoint of the two shoulders is used as the coordinate origin. The Euclidean distance from the shoulder midpoint to the hip midpoint (trunk_length) is used as the scale factor to normalize each feature joint's (x, y) coordinates into relative coordinates. If trunk_length < 1e-6, the frame is marked as invalid.

**Joint Weight Scaling.** The normalized coordinates are multiplied by the square root of each joint's weight (`weight^0.5`), giving more importance to end-effector joints. For train00: shoulder 0.9, elbow 1.15, wrist 1.3. For train01: shoulder 0.9, elbow 1.1, wrist 1.25, hip 0.9, knee 1.05, ankle 1.15. All weighted (x, y) values are concatenated into a one-dimensional feature vector (train00: 12-dim, train01: 24-dim).

**Visibility Filtering.** Joints with visibility below `visibility_threshold=0.45` are marked as missing. If the average visibility of the feature joints falls below `min_feature_visibility` (train00: 0.55, train01: 0.60), the frame is excluded from scoring.

**Temporal Smoothing.** In the online stage, exponential moving average smoothing (`pose_smoothing_alpha=0.65`) is applied to the user's pose vector to reduce single-frame jitter. In the offline stage, a stronger smoothing of `smoothing_alpha=0.40` is used.

### 4.3 Template Matching and Scoring

**Hybrid Similarity Score.** The system uses a hybrid of cosine similarity and weighted Euclidean distance:

$$S_{hybrid} = 0.60 \times S_{cosine} + 0.40 \times S_{euclidean}$$

The cosine component is $S_{cosine} = \max(0, \cos(\mathbf{u}, \mathbf{t}) \times 100)$. The Euclidean component computes per-joint deviations and mixes the mean and maximum:

$$S_{euclidean} = \max\left(0, \left(1 - \frac{0.7 \times \bar{d} + 0.3 \times d_{max}}{1.5}\right) \times 100\right)$$

where $\bar{d}$ is the mean per-joint deviation and $d_{max}$ is the distance of the most deviated joint. The 30% weight on the maximum deviation joint means the score reflects obvious errors in individual joints.

**Automatic Mirror Comparison.** Since the user may face the camera (mirrored relative to the demo video), the system computes both a direct comparison score and a mirrored comparison score, and takes the higher one. Mirroring is done through a dynamically built left-right joint index mapping, compatible with both 6-joint and 12-joint configurations.

**Score Grading and Combo Multiplier.** Scores are divided into three levels: pass (≥ 82), warn (≥ 62), and fail (< 62). Consecutive passes trigger a combo multiplier: after `combo_trigger_count=2` consecutive passes, each additional pass adds 0.20 to the multiplier, up to a cap of `combo_bonus_cap=2.00`. Final score = similarity score × combo multiplier.

**Real-Time Correction Hints.** When a score does not pass, the system finds the joint with the largest deviation and generates a targeted hint based on the joint type (e.g., "Try to open your arms wider" or "Keep your lower body more stable"). Each action category also has its own hint text (e.g., `arms_up`: "Raise your arms higher — try to get your wrists above shoulder level").

### 4.4 User Interface Design

The system uses PyQt5 to build a desktop interface with a three-page layout:

**Preparation Page (PreparePage).** Shows the training source cover thumbnail, resource information (video file, template file, audio status), challenge information (action name, difficulty, source), and usage notes. A dropdown (QComboBox) lets the user switch training sources; switching automatically regenerates or reloads resources.

**Live Follow-Along Page (GamePage).** The top area shows a timeline component (TimelineWidget) with dots marking keyframe positions (green for completed, orange for upcoming). The middle area shows the demo video and the camera feed with skeleton overlay side by side. The bottom area shows real-time feedback: current total score, combo count, completed keyframes, and correction hints. The main loop runs at `timer_interval_ms ≈ 20ms`, reading video and camera frames, detecting poses, and checking whether to trigger scoring each cycle.

**Results Page (ResultPage).** Shows the final score, grade (S/A/B/C, based on pass rate and total score), pass rate, highest combo, and audio status.

## 5. Experiments and Results

### 5.1 Experimental Setup

The system was tested on two training sources:

| Source | Duration | Keyframes | Feature Joints | Action Types | Avg. Visibility | Avg. Composite Score |
|--------|----------|-----------|----------------|--------------|-----------------|----------------------|
| train00 | 95.8s | 11 | 6 (upper body) | 6 (arms_clap_up, right_arm_side, left_arm_side, arms_swing_down, right_arm_swing, default) | 0.9507 | 0.7723 |
| train01 | 329.5s | 31 | 12 (full body) | 7 | 0.9678 | 0.6705 |

**train00 Checkpoint Details:**

| No. | Label | Action | Time | frame_index | Template Visibility | Template Mode | Status |
|-----|-------|--------|------|-------------|---------------------|---------------|--------|
| 1 | Arm Clap Up 1 | arms_clap_up | 00:06 | 180 | 0.9897 | train00_auto_motion | ok |
| 2 | Arm Clap Up 2 | arms_clap_up | 00:15 | 448 | 0.9288 | train00_auto_motion | ok |
| 3 | Right Arm Side Strike 1 | right_arm_side | 00:23 | 704 | 0.9387 | train00_auto_motion | ok |
| 4 | Motion Clip 1 | default | 00:31 | 940 | 0.9462 | train00_auto_motion | ok |
| 5 | Left Arm Side Strike 1 | left_arm_side | 00:40 | 1192 | 0.9242 | train00_auto_motion | ok |
| 6 | Arm Clap Up 3 | arms_clap_up | 00:49 | 1468 | 0.9373 | train00_auto_motion | ok |
| 7 | Arm Clap Up 4 | arms_clap_up | 00:59 | 1764 | 0.9875 | train00_auto_motion | ok |
| 8 | Arms Swing Down 1 | arms_swing_down | 01:08 | 2036 | 0.9172 | train00_auto_motion | ok |
| 9 | Arms Swing Down 2 | arms_swing_down | 01:13 | 2192 | 0.9587 | train00_auto_motion | ok |
| 10 | Arms Swing Down 3 | arms_swing_down | 01:24 | 2528 | 0.9672 | train00_auto_motion | ok |
| 11 | Right Arm Swing Down 1 | right_arm_swing | 01:30 | 2692 | 0.9668 | train00_auto_motion | ok |

All 11 checkpoints were automatically extracted from train00.mp4. Template visibility is above 0.91 for all checkpoints, and all have status ok. The action distribution covers 6 types: arms_clap_up (4), arms_swing_down (3), right_arm_side (1), left_arm_side (1), right_arm_swing (1), default (1).

Hardware: a standard consumer laptop with a built-in RGB camera (640×480), no GPU. Software: Python 3.10+, MediaPipe 0.10.14, OpenCV 4.8+, PyQt5 5.15+.

### 5.2 Offline Template Quality Verification

To verify the quality of the automatically extracted keyframes, each keyframe's corresponding video frame was re-fed into the pose detector, and the match score between the detection result and the stored template was computed. All 11 checkpoints in train00 passed (score ≥ 82). In train01, 30 out of 31 checkpoints passed and 1 was at warn level. This shows that the keyframes selected by the four-dimensional composite score have good template quality and internal consistency.

### 5.3 Effect of the Hybrid Scoring Method

Compared to using cosine similarity alone, the hybrid scoring method (60% cosine + 40% Euclidean) performs better in two ways: (1) it is more sensitive to local joint errors — pure cosine can still give a high score when the overall direction is correct but one joint is clearly wrong, while the hybrid method reduces the score in such cases through the maximum deviation penalty (30% weight); (2) it has better discrimination across different action types — full-body actions (such as knee raises and wide stances) involve more joints, and the Euclidean component captures spatial distribution differences more effectively.

### 5.4 Effect of the Motion Diversity Constraint

On train01, after enabling the motion diversity constraint (`MAX_CONSECUTIVE_SAME_ACTION=2`), the final 31 checkpoints cover 7 action types (arms_up, single_arm_reach, arms_open, cross_body, left_knee_up, right_knee_up, default). Compared to the version without the constraint, sequences of 3 or more consecutive same-type actions were reduced, making the follow-along experience more varied.

## 6. Limitations and Future Work

The current system has the following limitations: (1) it relies on MediaPipe BlazePose's single-person detection, so a multi-person scene would need an additional person selection mechanism; (2) action classification is based on a hand-crafted threshold decision tree, which has limited ability to generalize to new action types — a lightweight classification network could be considered in the future; (3) scoring is based on single-frame pose matching and does not model the temporal continuity of actions — DTW or sliding window matching could be introduced; (4) audio synchronization relies on QSoundEffect, which may have latency in some environments — a more precise audio backend could be used; (5) the system currently supports only single-user follow-along — a multi-user competitive mode would require extending camera management and score isolation.

## 7. Conclusion

This project built RhythmMotionCV, a rhythm-based motion follow-along system, using MediaPipe BlazePose, template matching scoring, and a PyQt5 desktop interface. The system automatically selects high-quality keyframes from demo videos using a four-dimensional composite score, performs real-time scoring with a hybrid cosine and weighted Euclidean distance method, and improves the user experience through a motion diversity constraint, automatic mirror comparison, and a combo multiplier. The system supports switching between multiple training sources (6-joint upper body / 12-joint full body) and has been fully tested on train00 and train01, covering the complete pipeline from demo video input to real-time user scoring.

## Personal Work

This project was completed independently. The main work includes:

1. **System Architecture**: Designed the two-stage offline/online architecture and the three-page interaction flow (preparation → follow-along → results).
2. **Keyframe Selection Algorithm**: Designed and implemented the four-dimensional composite scoring algorithm with a motion diversity constraint and candidate window selection strategy.
3. **Hybrid Scoring Method**: Implemented the hybrid cosine and weighted Euclidean scoring method (6:4 weight), automatic mirror comparison, and the combo multiplier.
4. **Multiple Training Source Support**: Implemented the TRAIN_SOURCES registry and dynamic feature configuration switching, supporting 6-joint upper body and 12-joint full body modes.
5. **Pipeline Module**: Implemented `train01_pipeline.py`, covering the full pipeline of video analysis, keyframe selection, template generation, audio extraction, and resource management.
6. **PyQt5 Interface**: Implemented the complete desktop interface including the timeline component, dual-view preview, real-time feedback, and training source switching.

## References

[1] Cao, Z., Simon, T., Wei, S. E., & Sheikh, Y. (2017). Realtime multi-person 2D pose estimation using part affinity fields. In CVPR.

[2] Sun, K., Xiao, B., Liu, D., & Wang, J. (2019). Deep high-resolution representation learning for visual recognition. In CVPR.

[3] Bazarevsky, V., Grishchenko, I., Raveendran, K., Zhu, T., Zhang, F., & Grundmann, M. (2020). BlazePose: On-device real-time body pose tracking. In CVPR Workshop.

[4] Yan, S., Xiong, Y., & Lin, D. (2018). Spatial temporal graph convolutional networks for skeleton-based action recognition. In AAAI.
