# RhythmMotionCV

基于计算机视觉的节奏运动跟随小游戏，使用 **MediaPipe** + **OpenCV** 实时捕捉人体姿态，与示范视频中的动作模板进行评分比对。

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Jiangleling/rhythm_motion_game)

## 项目结构

```
RhythmMotionCV/
├── notebook/                          # Jupyter Notebook 演示版（主入口）
│   ├── RhythmMotionCV_节奏运动跟随小游戏.ipynb  # 完整演示 notebook
│   ├── train01_pipeline.py            # 训练数据处理流水线
│   ├── data/generated/                # 自动生成的姿态模板、关键帧、音频
│   ├── train00.mp4 / train01.mp4      # 示范跟练视频
│   └── requirements.txt               # 依赖
├── .gitignore
└── README.md
```

## 功能特点

- **实时姿态检测**：基于 MediaPipe Pose，支持全身 12 关节跟踪
- **混合评分算法**：余弦相似度（60%）+ 加权欧氏距离（40%），提高不同动作区分度
- **连击倍率系统**：连续达标触发分数加成，上限 2.0 倍
- **镜像自动适配**：自动尝试镜像比对，左右手不分方向
- **离线验证**：支持对示范视频关键帧进行自动打分验证
- **PyQt5 界面**：准备页 → 实时跟练页 → 结果页，完整流程

## Colab 在线运行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jiangleling/rhythm_motion_game/blob/main/notebook/RhythmMotionCV_%E8%8A%82%E5%A5%8F%E8%BF%90%E5%8A%A8%E8%B7%9F%E9%9A%8F%E5%B0%8F%E6%B8%B8%E6%88%8F.ipynb)

## 环境要求

- Python 3.10+
- 摄像头（实时跟练需要）

## 快速开始

```bash
# 安装依赖
pip install -r notebook/requirements.txt

# 启动 Jupyter
jupyter notebook notebook/RhythmMotionCV_节奏运动跟随小游戏.ipynb
```

或使用 VS Code 直接打开 notebook 运行。

## 训练源

| 训练源 | 时长 | 关节数 | 说明 |
|--------|------|--------|------|
| train00 | ~96s | 6 (上半身) | 基础双臂动作，适合入门体验 |
| train01 | ~330s | 12 (全身) | 完整真人跟练，动作更丰富 |

Notebook 内置了自动模板生成管线。首次运行时会从示范视频中抽取关键帧，生成姿态模板和评分数据。

train00:https://drive.google.com/file/d/1eSHB2fVwfOz7-c7lUTI14hhPNbFio8Nz/view?usp=drive_link (仅供学习使用，如造成困扰请联系删除)
train01:https://drive.google.com/file/d/17WHlvbWna5CZX1fa3OtI1X33NQiSF69v/view?usp=drive_link (仅供学习使用，如造成困扰请联系删除)

## 技术栈

- **MediaPipe** — 姿态关键点检测
- **OpenCV** — 视频处理与骨架绘制
- **PyQt5** — 桌面界面
- **NumPy / Pandas** — 数据处理与评分计算
- **Matplotlib** — 离线验证可视化

## 评分机制

评分基于 60% 余弦相似度 + 40% 加权欧氏距离的混合方案，对关节位置偏差较大的动作进行额外惩罚。连续达标触发连击倍率，鼓励动作的稳定性与一致性。

- **pass**: ≥ 82 分 — 动作达标
- **warn**: ≥ 62 分 — 需要调整
- **fail**: < 62 分 — 动作偏差较大
