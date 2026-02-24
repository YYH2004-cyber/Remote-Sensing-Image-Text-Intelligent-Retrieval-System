# 遥感图像-文本智能检索系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.48.1-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

一个基于深度学习的多模态遥感图像智能检索系统，支持文本检索图像、图像检索图像、图像生成文本描述等多种检索模式。

[功能特性](#功能特性) • [快速开始](#快速开始) • [安装说明](#安装说明) • [使用教程](#使用教程) • [项目结构](#项目结构)

</div>

---

## 项目简介

遥感图像-文本智能检索系统是一个基于深度学习的多模态检索平台，采用 CLIP（Contrastive Language-Image Pre-training）模型作为核心算法，结合 FAISS 向量检索引擎，提供高效、准确的跨模态检索服务。系统支持遥感图像与文本之间的智能检索，适用于遥感图像库管理、智能标注、图像相似度分析等场景。

## 功能特性

### 核心功能

- **文本 → 图像检索**：输入文本描述，系统自动检索最相似的遥感图像
- **图像 → 图像检索**：上传一张遥感图像，系统在图库中查找相似的图片
- **图像 → 文本描述**：上传遥感图像，系统自动生成文本描述
- **历史记录管理**：自动记录检索历史，支持查询和管理
- **数据集管理**：支持动态添加、删除和管理图像数据集
- **高性能检索**：基于 FAISS 的向量索引，支持大规模数据快速检索

### 技术亮点

- 🚀 **多模态学习**：基于 CLIP 模型实现图像与文本的语义对齐
- ⚡ **高性能检索**：FAISS 向量索引，支持毫秒级检索响应
- 🎨 **用户友好界面**：基于 Streamlit 的现代化 Web 界面
- 🔧 **灵活配置**：支持 Top-K 参数调整，满足不同检索需求
- 💾 **历史记录**：自动保存检索历史，便于回溯和分析

## 技术栈

| 技术组件 | 版本 | 用途 |
|---------|------|------|
| Python | 3.10+ | 开发语言 |
| PyTorch | 2.7.0+cu118 | 深度学习框架 |
| OpenCLIP | 2.32.0 | 多模态模型 |
| FAISS-GPU | 1.7.2 | 向量检索引擎 |
| Streamlit | 1.48.1 | Web 应用框架 |
| NumPy | 1.26.4 | 数值计算 |
| PIL | 11.0.0 | 图像处理 |

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- CUDA 11.8+（可选，用于 GPU 加速）
- 8GB+ 内存（推荐 16GB+）
- 20GB+ 可用磁盘空间

### 一键安装

```bash
# 克隆仓库
git clone https://github.com/YYH2004-cyber/Remote-Sensing-Image-Text-Intelligent-Retrieval-System.git
cd Remote-Sensing-Image-Text-Intelligent-Retrieval-System

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run main.py
```

### 访问应用

启动成功后，在浏览器中访问：
```
Local URL: http://localhost:8501
Network URL: http://YOUR_IP:8501
```

## 安装说明

### 1. 环境配置

#### 安装 Python

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# macOS
brew install python@3.10

# Windows
# 从 https://www.python.org/downloads/ 下载安装
```

#### 创建虚拟环境

```bash
# 使用 venv
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 使用 conda（可选）
conda create -n rsicd python=3.10
conda activate rsicd
```

### 2. 依赖安装

```bash
# 升级 pip
pip install --upgrade pip

# 安装 PyTorch（根据你的 CUDA 版本选择）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

### 3. 数据准备

#### 下载预训练模型

将预训练模型文件 `RS5M_ViT-B-32_RSICD.pt` 放置到 `demo/data/` 目录下。

#### 准备图像数据集

将遥感图像放置到 `demo/rsicd_imgs/` 目录下。

#### 构建索引

```bash
# 构建图像索引
cd demo/preparation
python build_faiss_image.py

# 构建文本索引
python build_faiss_text.py
```

### 4. 验证安装

```bash
# 验证 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 验证 CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证 FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# 验证 OpenCLIP
python -c "import open_clip; print(f'OpenCLIP: {open_clip.__version__}')"
```

## 使用教程

### 文本 → 图像检索

1. 在左侧导航栏选择"文本 → 图像检索"
2. 在输入框中输入检索文本（如："机场跑道"、"建筑物群"）
3. 调整 Top-K 参数（返回结果数量）
4. 点击"开始检索"按钮
5. 查看检索结果和相似度评分

### 图像 → 图像检索

1. 在左侧导航栏选择"图像 → 图像检索"
2. 点击"上传图像"按钮，选择一张遥感图像
3. 调整 Top-K 参数
4. 点击"开始检索"按钮
5. 查看相似图像结果

### 图像 → 文本描述

1. 在左侧导航栏选择"图像 → 文本描述"
2. 上传一张遥感图像
3. 系统自动生成文本描述
4. 查看生成的描述内容

### 历史记录

1. 在左侧导航栏选择"历史记录"
2. 查看所有检索历史记录
3. 可以删除单条记录或清除所有记录
4. 点击记录可以查看详细信息

### 数据集管理

1. 在左侧导航栏选择"数据集管理"
2. 查看当前数据集统计信息
3. 上传新图像添加到数据集
4. 删除不需要的图像
5. 重建索引以更新数据集

## 项目结构

```
demo/
├── main.py                      # 主应用入口
├── utils.py                     # 工具函数模块
├── requirements.txt             # 依赖包列表
├── README.md                    # 项目说明文档
├── COMPLETE_GUIDE.md           # 完整指南文档
├── SETUP_GUIDE.md             # 环境配置指南
├── data/                        # 数据目录
│   ├── faiss_index.bin         # 图像向量索引
│   ├── faiss_index_text.bin    # 文本向量索引
│   ├── meta.json               # 图像元数据
│   ├── meta_text.json          # 文本元数据
│   ├── RS5M_ViT-B-32_RSICD.pt  # 预训练模型权重
│   └── search_history.json     # 检索历史记录
├── rsicd_imgs/                  # 遥感图像库
├── preparation/                 # 数据预处理脚本
│   ├── build_faiss_image.py    # 构建图像索引
│   └── build_faiss_text.py     # 构建文本索引
└── tools/                       # 辅助工具
    ├── dataset_rsicd.json      # RSICD数据集
    ├── dataset_RSITMD.json     # RSITMD数据集
    ├── captions.txt            # 图像描述文本
    ├── get_captions.py         # 描述提取工具
    └── img_name.py             # 图像命名工具
```

## 常见问题

### Q: 模型加载失败怎么办？

A: 确保已下载预训练模型文件 `RS5M_ViT-B-32_RSICD.pt` 并放置在正确的目录下。如果仍然失败，可以尝试使用 OpenAI 预训练模型。

### Q: 检索速度慢怎么办？

A: 检查是否使用了 GPU 加速。如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch 和 FAISS-GPU 可以大幅提升检索速度。

### Q: 如何添加新的图像？

A: 将新图像放置到 `demo/rsicd_imgs/` 目录下，然后运行 `build_faiss_image.py` 重建索引。

### Q: 内存不足怎么办？

A: 可以减少 Top-K 参数值，或者使用 CPU 模式运行。在 `utils.py` 中设置 `DEVICE="cpu"`。

## 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 致谢

- [CLIP](https://openai.com/research/clip) - OpenAI 的对比语言-图像预训练模型
- [FAISS](https://github.com/facebookresearch/faiss) - Facebook 的向量相似度搜索库
- [Streamlit](https://streamlit.io/) - 用于构建数据应用的 Python 框架

## 联系方式

- 项目主页: [https://github.com/YYH2004-cyber/Remote-Sensing-Image-Text-Intelligent-Retrieval-System](https://github.com/YYH2004-cyber/Remote-Sensing-Image-Text-Intelligent-Retrieval-System)
- 问题反馈: [Issues](https://github.com/YYH2004-cyber/Remote-Sensing-Image-Text-Intelligent-Retrieval-System/issues)

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐️ Star 支持一下！**

Made with ❤️ by YYH2004-cyber

</div>
