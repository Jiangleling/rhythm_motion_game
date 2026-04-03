# standard_pose_templates.json 字段说明与下拉动作排查

这份说明对应文件 data/generated/standard_pose_templates.json。

先给结论：

1. 这个文件主要决定“拿什么姿态做相似度比对”。
2. 真正决定“什么时候判分、及格线是多少”的文件是 data/generated/score_frames.json。
3. 如果下拉动作总不过，最常见的原因通常不是 template_id 写错，而是以下三类问题：
   - 下拉时你的手腕、手肘在摄像头里可见性不够。
   - pull_down 的 template_vector 和你实际做的姿态差得太大。
   - 判分时间点或阈值过严，导致动作还没做到位就已经判分。

## 1. 这个文件在项目里到底起什么作用

可以把两个 JSON 文件拆开理解：

- standard_pose_templates.json：定义“标准姿态长什么样”。
- score_frames.json：定义“在哪个时间点用哪个模板判分”。

当前 Notebook 的运行逻辑是：

1. 从 score_frames.json 里读取每个关键帧的 template_id、timestamp_ms、frame_index、pass_threshold、warn_threshold、correction_hint。
2. 再去 standard_pose_templates.json 里按 template_id 找到对应的 template_vector。
3. 运行时把摄像头检测到的人体姿态归一化成 24 维向量。
4. 用余弦相似度把“用户向量”和“模板向量”做比较。
5. 如果分数大于等于 pass_threshold，就判定通过。

这意味着：

- 你在 standard_pose_templates.json 里改的核心是模板姿态本身。
- 你在 score_frames.json 里改的是判分时机、提示语、通过线。

## 2. 顶层字段含义

standard_pose_templates.json 顶层现在有这几个字段：

### video_path

- 含义：关联的视频路径，主要是元数据。
- 作用：方便你知道这批模板对应哪一个视频素材。
- 对实时判分的直接影响：基本没有。

### thumbnail_path

- 含义：缩略图路径，主要用于展示。
- 对实时判分的直接影响：没有。

### joint_names

- 含义：模板向量里一共用了哪些关节点，以及默认顺序。
- 当前顺序是：
  - left_shoulder
  - right_shoulder
  - left_elbow
  - right_elbow
  - left_wrist
  - right_wrist
  - left_hip
  - right_hip
  - left_knee
  - right_knee
  - left_ankle
  - right_ankle
- 作用：告诉你 template_vector 的 24 个数字分别对应谁。

### templates

- 含义：所有模板条目的数组。
- 每一个对象就是一个可被关键帧引用的标准姿态。

## 3. 单个模板条目的字段含义

下面以 pull_down 条目为例说明。

```json
{
  "template_id": "kf_005",
  "frame_index": 714,
  "timestamp_ms": 23800,
  "action": "pull_down",
  "label": "下拉动作 2",
  "template_vector": [ ... 24 个数 ... ],
  "core_visibility": 0.6801,
  "template_source": "manual_action_hint",
  "template_source_frame_index": 693,
  "template_source_timestamp_ms": 23100,
  "template_source_note": "示范视频为节奏提示素材，模板按动作提示设计并复用动作原型。"
}
```

### template_id

- 含义：模板唯一 ID。
- 用途：让 score_frames.json 通过 template_id 找到这个模板。
- 注意：这个字段必须和 score_frames.json 中对应关键帧的 template_id 完全一致。
- 如果不一致：这个关键帧就可能直接找不到模板，或者引用到错误姿态。

### frame_index

- 含义：这个模板对应的帧序号。
- 当前用途：主要是记录和排查用的元数据。
- 重要说明：在你现在这套 Notebook 逻辑里，实时判分真正采用的 frame_index 优先来自 score_frames.json，不是这里。
- 所以：单独改这里的 frame_index，通常不会直接修复“下拉动作不过”的问题。

### timestamp_ms

- 含义：这个模板记录的时间戳，单位毫秒。
- 当前用途：同样主要是模板元数据。
- 重要说明：实时判分真正触发的时间点优先看 score_frames.json 里的 timestamp_ms。
- 所以：如果你怀疑判分时机太早或太晚，应该优先改 score_frames.json，而不是这里。

### action

- 含义：动作类别名称。
- 用途：
  - 让界面显示动作类型。
  - 在评分失败时，辅助生成 correction_hint。
- 约束：最好和 score_frames.json 里的 action 一致。

### label

- 含义：给人看的动作标签。
- 用途：界面显示、配置检查表显示、排查时识别用。
- 对判分数值的直接影响：没有。

### template_vector

- 含义：真正参与相似度计算的姿态向量。
- 这是这个文件里最重要的字段。
- 长度：24 个数。
- 解释方式：12 个关节点，每个关节点 2 个值，依次是 x 和 y。

具体展开顺序如下：

1. left_shoulder.x
2. left_shoulder.y
3. right_shoulder.x
4. right_shoulder.y
5. left_elbow.x
6. left_elbow.y
7. right_elbow.x
8. right_elbow.y
9. left_wrist.x
10. left_wrist.y
11. right_wrist.x
12. right_wrist.y
13. left_hip.x
14. left_hip.y
15. right_hip.x
16. right_hip.y
17. left_knee.x
18. left_knee.y
19. right_knee.x
20. right_knee.y
21. left_ankle.x
22. left_ankle.y
23. right_ankle.x
24. right_ankle.y

这些值不是原始像素坐标，而是归一化后的相对位置：

- 原点：双肩中点。
- 缩放基准：躯干长度，具体是双肩中点到双髋中点的距离。
- x 含义：相对双肩中点的水平偏移。
- y 含义：相对双肩中点的垂直偏移。

因为图像坐标系里 y 轴向下，所以：

- y 为负：说明该点在肩中点上方。
- y 为正：说明该点在肩中点下方。

这也是你判断“下拉动作模板是否合理”的关键依据。

例如：

- 如果你希望是“先把手抬高，再往下拉”，那么抬高手阶段的 wrist.y 通常应该更偏负。
- 如果模板里 wrist.y 很大且为正，说明手腕位置已经明显低于肩部，更像拉下完成态。

### core_visibility

- 含义：生成该模板时，12 个核心点的平均可见性。
- 当前用途：主要是模板质量参考，不是实时判分的硬门槛。
- 非常重要：你当前 Notebook 的实时判分并不会直接读取这个字段来决定 pass 或 fail。

实时判分真正检查的是：

- 用户当前摄像头采样的 core_visibility 是否大于等于 0.60。

所以要特别注意：

- 模板里的 core_visibility 偏低，不一定导致你不过。
- 你自己做动作时，摄像头那一帧的 core_visibility 偏低，才会直接导致失败或提示“关键点可见性不足”。

### template_source

- 含义：模板来源类型。
- 当前值 manual_action_hint 表示：
  - 这不是从示范视频里自动精确提取的人体姿态。
  - 而是按动作提示手工设计并复用原型模板。
- 作用：帮助你理解模板的可信来源。

### template_source_frame_index

- 含义：模板来源参考帧号。
- 当前用途：记录模板大致参考了哪一帧附近的动作语义。
- 对实时判分的直接影响：没有。

### template_source_timestamp_ms

- 含义：模板来源参考时间点，单位毫秒。
- 当前用途：同上，主要用于追溯和检查。
- 对实时判分的直接影响：没有。

### template_source_note

- 含义：来源备注说明。
- 当前用途：告诉你这批模板是怎么来的。
- 对实时判分的直接影响：没有。

## 4. pull_down 现在为什么容易不过

你当前的 pull_down 有 4 个关键点，对应模板是：

- kf_004：下拉动作 1
- kf_005：下拉动作 2
- kf_006：下拉动作 3
- kf_007：下拉动作 4

但它们实际只复用了两套姿态原型：

- kf_004 和 kf_006 使用同一套 template_vector。
- kf_005 和 kf_007 使用同一套 template_vector。

这说明当前 pull_down 不是连续 4 个不同姿态，而是“两个姿态交替使用”。

这会带来几个问题：

### 问题 1：模板过于刚性

如果你真实做动作时，下拉轨迹更连续、更自然，而模板只有两个离散姿态，余弦相似度就容易不够高。

### 问题 2：及格线偏高

当前 pull_down 的 pass_threshold 在 score_frames.json 里是 80。

80 对于“手臂类动作”通常偏严格，尤其是：

- 手肘角度变化快。
- 手腕摆动大。
- 左右手不完全同步。
- 人和摄像头距离不稳定。

### 问题 3：下拉动作天然更容易掉可见性

下拉时常见情况是：

- 手腕速度快，导致识别抖动。
- 手肘靠近身体，关键点被遮挡。
- 上半身前倾，肩和腕的相对位置变化大。
- 手臂有时候出镜不完整。

在这套逻辑下，只要你那一帧用户采样的 core_visibility 小于 0.60，就会先被挡在相似度比较之前。

### 问题 4：判分时间可能和你的动作节奏不一致

实时判分触发时间看的是 score_frames.json 里的 timestamp_ms。

如果你实际习惯在 23.8s 的后 200ms 才把手拉到位，但系统已经在 23.8s 附近判了，那你会感觉“动作明明做了，但总不过”。

这类问题应该改 score_frames.json，不是改 standard_pose_templates.json。

## 5. 先改哪里，最有效

建议按这个顺序排查：

### 第一步：先改 score_frames.json 的阈值

先只改 pull_down 对应的 4 个关键帧：

- pass_threshold：80 改成 72 或 75
- warn_threshold：60 改成 50 或 55

这样可以先确认问题到底是“模板不准”，还是“阈值太严”。

### 第二步：如果还是不过，再改 template_vector

这时再去改 standard_pose_templates.json 里 kf_004 到 kf_007 的 template_vector。

原则是：

- 不要先追求动作漂亮。
- 先追求和你摄像头里真实容易做出来的姿态一致。

### 第三步：如果总是早判或晚判，再改 score_frames.json 的时间点

重点改：

- timestamp_ms
- frame_index

不要误以为改 standard_pose_templates.json 里的 timestamp_ms 就能修复时机问题。

## 6. 我对你当前 pull_down 的具体判断

从当前数据看，pull_down 里更需要你优先关注的是这几件事：

1. 下拉动作 1 和 3 完全复用同一套向量，下拉动作 2 和 4 也完全复用同一套向量，离散度偏低。
2. 其中一套 pull_down 原型的 core_visibility 只有 0.4909，这说明它本身更像“参考草图”，不太像高质量真实采样模板。
3. 但更关键的是，运行时真正拦住你的，很可能还是用户采样的可见性门槛 0.60 和 pass_threshold 80。

所以我的建议不是先大改结构，而是：

1. 先把 pull_down 的 pass_threshold 降到 72 到 75。
2. 再观察是不是从“经常 fail”变成“能到 warn 或偶尔 pass”。
3. 如果改善明显，说明主要矛盾是阈值和时机，不是模板字段写错。
4. 如果几乎没改善，再替换 pull_down 的 template_vector。

## 7. 你后面改模板时，最容易改错的点

### 容易改错 1：只改 standard_pose_templates.json，不改 score_frames.json

如果你想调通过率，很多时候你真正该改的是 score_frames.json 的：

- pass_threshold
- warn_threshold
- timestamp_ms
- frame_index

### 容易改错 2：template_id 没对齐

只要 template_id 对不上，score_frames.json 就找不到正确模板。

### 容易改错 3：向量顺序改乱

template_vector 必须严格按 joint_names 顺序展开，不能自行调换左右手、上下肢顺序。

### 容易改错 4：误把 template_source_* 当成判分字段

template_source_frame_index 和 template_source_timestamp_ms 只是说明来源，不参与实时打分。

## 8. 建议你的最小实验方案

如果你现在就想验证“到底是不是检测问题”，我建议你先只做这一组最小改动：

1. 在 score_frames.json 里，把 kf_004 到 kf_007 的 pass_threshold 从 80 改成 74。
2. 把 warn_threshold 从 60 改成 55。
3. 保持 standard_pose_templates.json 不动。
4. 跑几次实际跟练。

如果通过率明显提升，说明模板不是主要问题。

如果仍然经常不过，再改 standard_pose_templates.json 的 pull_down 模板向量。

## 9. 配套模板文件

我另外给你放了一份可复制的调参模板文件：

- docs/pull_down_tuning_template.jsonc

这个文件里包含：

- standard_pose_templates.json 的 pull_down 模板示例
- score_frames.json 的 pull_down 关键帧示例
- 每个字段该怎么改的注释

你后面如果要微调，直接照着那份改最快。