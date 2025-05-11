# 花卉识别系统 (Flower Recognition System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.4+](https://img.shields.io/badge/tensorflow-2.4+-orange.svg)](https://www.tensorflow.org/)

基于深度学习的花卉识别系统，能够识别5种常见花卉：雏菊、蒲公英、玫瑰、向日葵和郁金香。通过简单的Web界面，用户可以上传图片获取识别结果。

![花卉识别示例](examples/demo.png)

## 项目特点

- 🌸 **多类别识别**：可以识别5种常见花卉
- 🧠 **迁移学习**：基于MobileNetV2的高效架构
- 📊 **可视化界面**：提供Streamlit和Gradio两种界面选择
- 📱 **轻量级模型**：适合移动设备和边缘计算部署
- 🔍 **详细分析**：提供识别结果的置信度和概率分布

## 项目概述

本项目利用迁移学习和MobileNetV2卷积神经网络架构，构建了一个高效且准确的花卉识别模型。通过优化训练过程和数据增强，模型能够在各种光照条件和角度下识别花卉。

## 快速开始

### 安装

1. 克隆仓库：
```bash
git clone https://github.com/yourusername/flowers-recognition.git
cd flowers-recognition
```

2. 安装依赖项：
```bash
pip install -r requirements.txt
```

### 使用流程

1. **训练模型**（可选，已提供预训练模型）：
```bash
python train.py
```

2. **启动Web应用**（选择一种）：

- Streamlit界面（推荐）：
```bash
streamlit run app.py
```

- Gradio界面：
```bash
python gradio_app.py
```

3. 在浏览器中上传花卉图片进行识别

## 数据集

数据集包含5种花卉的图像，直接存放在项目根目录下：
- `daisy/` - 雏菊图像
- `dandelion/` - 蒲公英图像
- `rose/` - 玫瑰图像
- `sunflower/` - 向日葵图像
- `tulip/` - 郁金香图像

## 项目结构

```
flowers-recognition/
│
├── daisy/                # 雏菊图像目录
├── dandelion/            # 蒲公英图像目录
├── rose/                 # 玫瑰图像目录
├── sunflower/            # 向日葵图像目录
├── tulip/                # 郁金香图像目录
│
├── examples/             # 示例图像目录
│   ├── daisy.jpg
│   ├── dandelion.jpg
│   ├── rose.jpg
│   ├── sunflower.jpg
│   └── tulip.jpg
│
├── train.py              # 模型训练脚本
├── app.py                # Streamlit Web应用
├── gradio_app.py         # Gradio Web应用
├── requirements.txt      # 项目依赖
├── README.md             # 项目说明
├── 实验报告.md            # 实验记录和分析
├── flower_model.h5       # 预训练模型（可通过Git LFS下载）
└── training_history.png  # 训练历史图表
```

## 模型架构

本项目使用MobileNetV2作为基础模型，并通过迁移学习进行微调：

1. 加载预训练的MobileNetV2模型（基于ImageNet权重）
2. 冻结基础模型的所有层
3. 添加全局平均池化层和密集分类层
4. 使用Adam优化器和分类交叉熵损失函数进行训练

```
模型结构：
MobileNetV2 (预训练) → GlobalAveragePooling2D → Dense(128, ReLU) → Dense(5, Softmax)
```

## 性能评估

| 类别 | 精确率 | 召回率 | F1分数 |
|------|-------|-------|-------|
| 雏菊 | 94.5% | 95.2% | 94.8% |
| 蒲公英 | 92.7% | 93.8% | 93.2% |
| 玫瑰 | 91.2% | 90.5% | 90.8% |
| 向日葵 | 96.3% | 96.1% | 96.2% |
| 郁金香 | 91.0% | 89.7% | 90.3% |

*注：以上数据为样例，实际性能可能因训练条件而异*

## 贡献指南

1. Fork本仓库
2. 创建您的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打开Pull Request

## 许可证

本项目采用MIT许可证 - 详情请参见 [LICENSE](LICENSE) 文件

## 致谢

- 感谢TensorFlow和Keras团队提供的优秀深度学习框架
- 感谢Streamlit提供的简便部署工具
