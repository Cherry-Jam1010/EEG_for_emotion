# 🧠 EEG 情绪智能分析实时系统
### (EEG Emotion Intelligence Analysis Real-time System)

这是一个结合了**深度学习 (CNN)** 与 **神经科学理论** 的跨学科项目。系统能够实时解析脑电信号（EEG），利用卷积神经网络预测被试者的情绪状态（积极、消极、中性），并通过交互式可视化界面展示脑波频段能量分布及大脑皮层热力拓扑图。

---

## 🚀 核心特性

* **实时情绪诊断**：基于训练好的 CNN 模型，实现高置信度的情绪状态预测。
* **10-20 系统热力拓扑**：动态绘制大脑简笔轮廓，并将电极点能量映射为热力分布，支持悬停查看特定脑区（如 Fp1, Oz）的心理学功能。
* **频段能量解析 (Radar)**：实时计算并展示 **Delta, Theta, Alpha, Beta, Gamma** 五大频段的能量占比。
* **神经科学深度解读**：系统自动生成基于“前额叶不对称理论”及“Alpha/Beta 比率”的专业分析报告。
* **内置百科指南**：专为新手设计的侧边栏百科，科普脑电导联命名法及生理唤醒含义。

---

## 🛠️ 技术栈

| 模块 | 技术实现 |
| :--- | :--- |
| **后端 (Backend)** | Python / Flask |
| **深度学习 (AI)** | TensorFlow / Keras (CNN), Scikit-learn, Joblib |
| **数据处理** | Pandas, Numpy (FFT 变换与方差分析) |
| **前端 (Frontend)** | HTML5, CSS3 (Flex/Grid Layout), JavaScript |
| **可视化 (Viz)** | Apache ECharts (5.4.3) |

---

## 📂 项目结构

```text
.
├── app.py                # Flask 后端服务，包含 AI 推理逻辑
├── index.html            # 前端交互界面与 ECharts 可视化配置
├── emotions.csv          # 脑电信号原始数据集 (需自行准备)
├── emotion_cnn_model.h5  # 预训练好的 CNN 模型文件
├── scaler.pkl            # 数据标准化模型
└── label_encoder.pkl     # 标签编码器
```

---

## 📋 快速启动

### 1. 环境准备
确保你的环境中已安装以下 Python 库：
```bash
pip install flask flask-cors pandas numpy tensorflow joblib scikit-learn
```

### 2. 启动后端服务
在项目根目录下运行服务：
```bash
python app.py
```
*服务将默认运行在 `http://127.0.0.1:5000`。*

### 3. 运行前端界面
由于浏览器安全策略（CORS），建议通过简单的静态服务器运行 HTML：
```bash
# 在另一个终端中运行
python -m http.server 8000
```
然后在浏览器中访问 `http://127.0.0.1:8000`。

---

## 🔬 神经科学背景说明

本系统在分析过程中参考了以下理论指标：
* **前额叶不对称性 (Frontal Asymmetry)**：左侧前额叶 (Fp1) 活跃通常关联趋近动机（积极情绪）；右侧 (Fp2) 活跃通常关联回避动机（消极情绪）。
* **Beta/Alpha 比率**：用于衡量心理负荷与放松程度。
* **10-20 系统**：国际标准的电极放置法，确保空间定位的科学性。

