# Semi-UNet3+-CBAM-IC-SEM-Defect-Segmentation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7%2B-green.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.8-orange.svg)](https://pytorch.org/)

## 项目简介

本项目实现了基于Semi-UNet3+-CBAM的集成电路SEM图像半监督缺陷分割方法。随着半导体工艺向纳米尺度发展，集成电路器件尺寸持续缩小、集成度不断提高，制造过程中产生的微小缺陷（如金属层断裂、颗粒污染、光刻残留、形貌异常等）对芯片良率和可靠性构成严重威胁。

传统的图像处理方法在面对高分辨率、高噪声、对比度不均匀的SEM图像时存在局限性，特别是对于在规则纹理背景中占比极小的缺陷区域，难以准确识别和定位。

## 研究目标

1. 在UNet3+全尺度特征融合结构基础上引入CBAM注意力机制，构建Semi-UNet3+-CBAM分割网络
2. 设计半监督学习策略，充分利用少量带标注样本和大量未标注样本进行联合训练
3. 在降低标注成本的同时，提升微小缺陷区域的表征和定位能力
4. 验证所提方法在IC SEM图像缺陷分割场景中的有效性和优势

## 项目结构

```
Semi-UNet3+-CBAM-IC-SEM-Defect-Segmentation/
├── README.md
├── requirements.txt
├── .gitignore
├── setup.py
├── src/                      # 源代码目录
│   ├── data/                 # 数据处理相关代码
│   │   └── data_loader.py    # 数据加载器
│   ├── models/               # 模型定义
│   │   ├── __init__.py
│   │   └── unet3p_cbam.py    # Semi-UNet3+-CBAM模型定义
│   ├── utils/                # 工具函数
│   │   └── utils.py          # 评估指标等工具函数
│   ├── configs/              # 配置文件
│   │   └── default_config.yaml # 默认配置
│   └── cli/                  # 命令行接口
│       ├── train.py          # 训练脚本
│       ├── run_training.py   # 启动训练的入口脚本
│       ├── preprocess_data.py # 数据预处理脚本
│       └── split_dataset.py  # 数据集划分脚本
├── datasets/                 # 数据集目录
│   ├── Anomaly_train/        # 异常检测训练集
│   ├── Anomaly_test/         # 异常检测测试集
│   │   ├── abnormal_bbox/    # 异常边界框标注
│   │   ├── abnormal_mask/    # 异常像素级掩码
│   │   └── normal_img/       # 正常图像
│   ├── Inpainting_train/     # 图像修复训练集
│   └── Inpainting_test/      # 图像修复测试集
└── checkpoints/              # 模型检查点（训练时生成）
```

## 环境配置

1. 克隆项目：
```bash
git clone <repository-url>
cd Semi-UNet3+-CBAM-IC-SEM-Defect-Segmentation
```

2. 创建虚拟环境并安装依赖：
```bash
conda create -n semi_unet python=3.8
conda activate semi_unet
pip install -r requirements.txt
# 或者使用setup.py安装
pip install -e .
```

## 使用方法

### 训练模型
```bash
python -m src.cli.run_training --config src/configs/default_config.yaml --device cuda
```

或者直接运行训练脚本：
```bash
python -m src.cli.train --config src/configs/default_config.yaml
```

### 使用命令行工具（如果已安装包）
```bash
# 训练模型
semi-unet3p-train --config src/configs/default_config.yaml

# 预处理数据
semi-unet3p-preprocess --data-dir ./datasets --output-dir ./preprocessed_data
```

## 技术方案

### 网络架构
- **基础架构**: UNet3+ 全尺度特征融合结构
- **注意力机制**: CBAM (Convolutional Block Attention Module)
- **学习范式**: 半监督学习框架

### 核心创新点
- 针对IC SEM图像中小目标缺陷的优化网络设计
- CBAM注意力机制增强微小异常区域感知能力
- 半监督学习策略降低像素级标注成本
- 全尺度特征融合提升多尺度缺陷检测效果

## 方法详解

### Semi-UNet3+-CBAM网络结构
1. **编码器**: 多层次特征提取
2. **全尺度跳跃连接**: 不同层级特征融合
3. **CBAM模块**: 通道注意力和空间注意力机制
4. **解码器**: 特征重构与缺陷分割

### 半监督学习策略
- 利用少量像素级标注数据进行有监督学习
- 利用大量未标注数据进行一致性正则化
- 结合伪标签技术提升模型泛化能力

## 数据集

本项目使用真实的集成电路SEM图像数据，包含：
- 高分辨率SEM图像
- 像素级缺陷标注
- 规则纹理背景中的多种缺陷类型

数据集结构：
- Anomaly_train: 包含25,160张正常图像和116张异常图像
- Anomaly_test: 包含1,272张正常图像和116张异常图像
- Inpainting_train: 1,312对合成异常图像及其对应正常图像
- Inpainting_test: 135对合成异常图像及其对应正常图像

## 评估指标

- IoU (Intersection over Union)
- Dice系数
- Pixel Accuracy
- Precision, Recall, F1 Score

## 未来工作

1. 扩展数据集规模，提高模型泛化能力
2. 优化网络结构，减少计算复杂度
3. 添加更多半监督学习策略
4. 集成模型压缩技术以适应边缘部署

## 许可证

本项目采用MIT许可证 - 详情请参阅[LICENSE](LICENSE)文件

## 参考文献

L. Huang, D. Cheng, X. Yang, T. Lin, Y. Shi, K. Yang, B.-H. Gwee, B. Wen, "Joint Anomaly Detection and Inpainting for Microscopy Images via Deep Self-Supervised Learning," in Proc. IEEE Int. Conf. Image Processing (ICIP), 2021.