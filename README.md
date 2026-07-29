# 🚪 门图识别 Android APP

基于 TensorFlow Lite + 纯 Kotlin 双引擎的麻将门图识别工具。

## 项目结构

```
DoorMatcher/
├── app/
│   └── src/main/
│       ├── java/com/doormatcher/
│       │   ├── DoorMatcherApp.kt           # Application
│       │   ├── data/
│       │   │   └── DoorDatabase.kt         # 门图数据库管理
│       │   ├── native/
│       │   │   ├── DoorMatcherNative.kt    # 核心比对算法（Kotlin 实现）
│       │   │   └── DoorMatcherPureKotlin.kt # 备用实现
│       │   ├── tflite/
│       │   │   └── DoorMatcherTFLite.kt    # TFLite 推理层
│       │   └── ui/
│       │       ├── MainActivity.kt         # 主界面
│       │       ├── ResultActivity.kt        # 结果页
│       │       ├── SettingsActivity.kt      # 参数设置
│       │       └── FloatingService.kt       # 悬浮窗 + 实时识别
│       ├── res/                            # 布局/资源
│       └── assets/
│           ├── doors/                      # ← 29张门图放这里
│           └── models/                     # ← TFLite 模型放这里
└── model_train/
    └── train_tflite_model.py               # 模型训练脚本
```

## 功能特性

| 功能 | 说明 |
|------|------|
| 🎮 屏幕录制 | MediaProjection API，截取游戏画面 |
| 🧠 双引擎 | TFLite（深度学习）/ 纯 Kotlin（完全离线）|
| 📊 三指标比对 | Cosine + Histogram + SSIM，可调权重 |
| 🎯 防抖逻辑 | 连续 3 帧匹配同一门图才确认 |
| ⚙️ 参数可调 | 区域比例、权重全可在 APP 内调整 |
| 🔄 热更新 | TFLite 模型从 filesDir 加载，支持网络更新 |

## 编译

### 前置条件
- Android Studio Hedgehog 或更高
- Android SDK 34
- Kotlin 1.9+
- Python 3.8+（用于训练模型）

### 步骤

```bash
# 1. 克隆/复制项目到本地
cd DoorMatcher

# 2. 放入门图（29张）
cp /path/to/your/doors/*.jpg app/src/main/assets/doors/
cp /path/to/your/doors/*.png app/src/main/assets/doors/

# 3. 训练 TFLite 模型（可选，纯 Kotlin 版本无需此步）
pip install tensorflow pillow numpy
python model_train/train_tflite_model.py
# 生成: app/src/main/assets/models/door_model_int8.tflite
#       app/src/main/assets/models/door_embeddings.json

# 4. Android Studio 打开项目，Sync & Run
```

### Gradle 命令行编译
```bash
./gradlew assembleDebug      # 调试 APK
./gradlew assembleRelease     # 发布 APK
```

## 比对算法说明

### 区域提取
- 截取截图和门图的 **x=[25%:75%] y=[25%:75%]** 中央区域
- 缩放到统一尺寸 **450×750** 后比对

### 三指标综合分
```
综合分 = Cosine×35% + Histogram×15% + SSIM×50%
```

- **Cosine**: 像素向量余弦相似度，对整体色调敏感
- **Histogram**: 颜色直方图匹配，对颜色分布好
- **SSIM**: 结构相似度，对纹理图案最重要（权重最高）

### 防抖机制
- 连续 3 帧匹配同一门图才更新显示
- 避免画面晃动导致的误判闪烁

## TFLite 模型说明

训练脚本输出两种模型：
- `door_model_fp32.tflite` — FP32 精度 (~2MB)
- `door_model_int8.tflite` — INT8 量化 (~500KB)，推荐手机使用

嵌入向量 JSON（door_embeddings.json）包含所有 29 张门图的预提取特征，APP 启动时加载，运行时只需做向量点积，速度极快。

## 隐私说明

- 全部离线运行，无数据上传
- 屏幕录制仅限本 APP 使用
- 网络权限仅用于检查 APP 更新

## 权限清单

```
RECORD_AUDIO          — 屏幕录制（必须）
FOREGROUND_SERVICE    — 前台服务（必须）
FOREGROUND_SERVICE_MEDIA_PROJECTION — 同上
INTERNET              — 检查更新（可选）
SYSTEM_ALERT_WINDOW   — 悬浮窗
READ_EXTERNAL_STORAGE — 读取门图（可选）
```

## 开发备忘

- 主引擎：`DoorMatcherNative.kt`（纯 Kotlin，直接可用）
- TFLite 引擎：`DoorMatcherTFLite.kt`（需要模型文件）
- 两种引擎可并行存在，优先用 TFLite，不可用时 fallback 到 Kotlin
- 截图方向：横屏右半部分，需要根据实际游戏做微调
