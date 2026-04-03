# RhythmMotionCV

单关卡 AI 动作匹配健康训练系统。

项目使用 `OpenCV + MediaPipe Pose + NumPy + PyQt5` 构建，流程闭环为：

`启动页 -> 准备页 -> 3 秒倒计时 -> 关卡核心页 -> 结算页`

系统会先通过预处理脚本对教学视频提取标准姿态模板与关键得分帧；运行时左侧播放示范视频，右侧实时显示摄像头骨架，视频到达关键得分帧时自动执行姿态匹配评分，并在页面中触发分数、边框闪烁、提示语和连击动效。

## 项目结构

- `main.py`：主程序入口与流程控制器
- `rhythm_motion_game.py`：兼容入口，等价于运行 `main.py`
- `ui_pages.py`：四个页面、切换动画、计分显示、反馈动效
- `pose_utils.py`：姿态检测、关键点归一化、余弦相似度评分、模板读写
- `preprocess_video.py`：教学视频预处理脚本，生成模板与关键帧配置
- `config.py`：统一管理路径、阈值、配色、字体与运行参数
- `requirements.txt`：依赖列表
- `data/generated/standard_pose_templates.json`：标准姿态模板输出
- `data/generated/score_frames.json`：关键得分帧配置输出
- `data/generated/thumbnail.jpg`：准备页缩略图输出
- `跟练视频.MP4`：样例教学视频

## 环境要求

- Windows
- Python 3.8+
- 推荐 Python 3.10 / 3.11 安装依赖，MediaPipe 与 PyQt5 在这两个版本上更稳
- 摄像头可正常被系统调用

## 一键安装依赖

```powershell
pip install -r requirements.txt
```

## 第一步：预处理教学视频

首次运行前，必须先生成标准模板和关键得分帧配置。

### 方式 1：基于旧动作脚本快速生成样例数据

```powershell
python preprocess_video.py --video 跟练视频.MP4 --seed-script 跟练动作脚本.json --accept-seed
```

### 方式 2：按固定时间间隔自动抽取关键帧

```powershell
python preprocess_video.py --video 跟练视频.MP4 --auto-interval-sec 8 --accept-seed
```

### 方式 3：手动标记关键得分帧

```powershell
python preprocess_video.py --video 跟练视频.MP4 --auto-interval-sec 8 --manual-review
```

手动标记界面快捷键：

- `A / D`：前后切帧
- `J / L`：快速前后跳转
- `M`：将当前帧标记为关键帧或取消关键帧
- `S`：保存并退出
- `ESC / Q`：取消退出

预处理完成后，会自动生成以下文件：

- `data/generated/standard_pose_templates.json`
- `data/generated/score_frames.json`
- `data/generated/thumbnail.jpg`

## 第二步：启动系统

```powershell
python main.py
```

或者继续使用兼容入口：

```powershell
python rhythm_motion_game.py
```

## 运行说明

1. 进入启动页后点击“开始挑战”
2. 在准备页确认缩略图、动作信息与注意事项
3. 点击“准备开始”，系统进入 3 秒倒计时
4. 关卡页左侧播放示范视频，右侧显示摄像头骨架
5. 视频到达关键得分帧时，系统自动抓取最近 100ms 内的姿态样本并执行评分
6. 视频播放完毕后自动进入结算页
7. 点击“再来一次”回到启动页

## 评分规则

- 检测模型：`MediaPipe Pose` 默认预训练权重
- 核心关节：左右肩、肘、腕、髋、膝、踝，共 12 个
- 归一化方式：以肩部中点为原点，以肩髋中点距离作为单位长度
- 相似度：对归一化后的 24 维向量做余弦相似度匹配，并转换为 0-100 分
- 判定阈值：
  - `>= 80`：动作达标，计分
  - `60 - 79`：待改进，不计分
  - `< 60`：不达标，提示纠错
- 连击倍率：连续 2 次达标后触发连击，倍率从 `x1.2` 起逐步提升

## 交互与异常处理

- `ESC`：在关卡页提前退出并进入结算页
- 摄像头打不开：程序不会崩溃，会返回准备页并给出提示
- 视频播放失败：程序会提示视频路径或编码异常
- 未检测到人体：关键帧按未达标处理，同时提示“保持全身入镜”
- 关键点可见性不足：不计分并提示用户调整站位
- 模板文件缺失：准备页按钮会禁用，并提示先运行预处理脚本

## 可调配置

如需修改默认路径、评分阈值、颜色、窗口尺寸、摄像头编号，可在 `config.py` 中统一调整。
