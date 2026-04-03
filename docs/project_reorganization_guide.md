# 项目整理说明

本次整理的目标是把“应用源码”“预处理工具”“Notebook 成品与资源”拆开，避免根目录继续混放代码、素材和生成文件。

## 当前正式结构

- `app/`：桌面应用正式源码
- `tools/`：预处理工具实现
- `notebook/`：Notebook 正式目录，也是 Notebook 相关资源的唯一正式位置
- 根目录：只保留 README、依赖文件和兼容启动入口

## Notebook 目录规则

- `notebook/RhythmMotionCV_节奏运动跟随小游戏.ipynb` 是唯一正式 Notebook 文件
- `notebook/跟练视频.MP4` 是 Notebook 与应用共用的视频资源
- `notebook/跟练动作脚本.json` 是预处理可复用的动作脚本
- `notebook/data/generated/` 保存模板、关键帧、缩略图和音频

根目录不再保留这套 Notebook 资源副本。

## 兼容入口

为了不打断原有使用方式，根目录保留了轻量入口：

- `python main.py`
- `python preprocess_video.py`
- `python rhythm_motion_game.py`

这些入口只做转发，正式实现分别位于 `app/` 和 `tools/`。

## 运行路径原则

- Python 应用统一通过 `app/config.py` 指向 `notebook/` 下的正式资源
- 预处理脚本输出统一写入 `notebook/data/generated/`
- Notebook 支持从项目根目录或 `notebook/` 目录启动 Jupyter

## 为什么这样整理

- 降低根目录噪音，项目职责更清晰
- Notebook 成品和它依赖的资源集中存放，提交和演示更稳定
- 后续替换视频、重新生成 JSON、调试 Notebook 时，路径更统一
- 根目录兼容入口保留后，原来常用命令不需要全部重学