# 遥感图像-文本智能检索系统 - VSCode + WSL 配置指南

## 目录

1. [环境要求](#1-环境要求)
2. [WSL环境配置](#2-wsl环境配置)
3. [Python环境设置](#3-python环境设置)
4. [依赖安装](#4-依赖安装)
5. [VSCode配置](#5-vscode配置)
6. [项目运行](#6-项目运行)
7. [常见问题](#7-常见问题)

---

## 1. 环境要求

### 1.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 4核 | 8核+ |
| 内存 | 8GB | 16GB+ |
| GPU | 无 | NVIDIA RTX 3060+ (6GB+ VRAM) |
| 存储 | 20GB | 50GB+ SSD |

### 1.2 软件要求

| 软件 | 版本要求 | 说明 |
|------|----------|------|
| Windows | 10/11 | WSL2支持 |
| WSL | 2.0+ | 推荐使用Ubuntu 22.04 |
| Python | 3.10+ | 推荐使用3.10或3.11 |
| VSCode | 最新版 | 推荐安装Python扩展 |
| CUDA | 11.8+ | GPU加速必需 |

---

## 2. WSL环境配置

### 2.1 安装WSL2

#### 在Windows上安装WSL2

**方法一：使用PowerShell命令（推荐）**

```powershell
# 以管理员身份打开PowerShell
wsl --install

# 重启计算机
```

**方法二：手动安装**

1. 下载WSL2内核更新包：[WSL2 Linux kernel update](https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi)
2. 运行安装包
3. 在PowerShell中运行：
   ```powershell
   wsl --set-default-version 2
   ```

### 2.2 安装Ubuntu

```powershell
# 在PowerShell中运行
wsl --install -d Ubuntu-22.04

# 设置默认用户名和密码
```

### 2.3 更新WSL Ubuntu系统

```bash
# 进入WSL Ubuntu
wsl

# 更新系统包
sudo apt update && sudo apt upgrade -y

# 安装基础工具
sudo apt install -y build-essential git curl wget vim
```

### 2.4 配置WSL网络（可选）

如果遇到网络问题，可以配置WSL使用镜像模式：

**在Windows上创建 `.wslconfig` 文件**：

```ini
# 位置: C:\Users\<你的用户名>\.wslconfig

[wsl2]
memory=16GB
processors=8
swap=8GB
localhostForwarding=true
```

**应用配置**：

```powershell
# 在PowerShell中运行
wsl --shutdown
wsl
```

---

## 3. Python环境设置

### 3.1 安装Python和pip

```bash
# 检查Python版本
python3 --version

# 如果Python版本过低，安装Python 3.10
sudo apt install -y python3.10 python3.10-venv python3-pip

# 设置默认Python版本
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### 3.2 创建虚拟环境

**推荐使用conda（如果已安装）**：

```bash
# 创建虚拟环境
conda create -n rsicd python=3.10 -y

# 激活环境
conda activate rsicd
```

**使用venv（推荐）**：

```bash
# 进入项目目录
cd /home/yyh2004/demo

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate
```

### 3.3 验证Python环境

```bash
# 检查Python版本
python --version

# 检查pip版本
pip --version

# 检查虚拟环境
which python
```

---

## 4. 依赖安装

### 4.1 创建requirements.txt

在项目根目录创建 `requirements.txt` 文件：

```txt
# 深度学习框架
torch==2.0.1
torchvision==0.15.2

# FAISS向量检索
faiss-cpu==1.7.4

# 多模态模型
open-clip-torch==2.23.0
ftfy==6.1.3
regex==2022.10.31

# Web框架
streamlit==1.28.1
altair==4.2.0

# 图像处理
Pillow==10.0.0
opencv-python==4.8.0.76

# 数据处理
numpy==1.24.3
pandas==2.0.3

# 工具库
tqdm==4.65.0
```

### 4.2 安装依赖

**使用pip安装**：

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt
```

**使用conda安装（如果使用conda）**：

```bash
# 激活conda环境
conda activate rsicd

# 安装PyTorch（CPU版本）
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装其他依赖
pip install open-clip-torch streamlit faiss-cpu
```

### 4.3 安装CUDA版本（如果有GPU）

```bash
# 检查CUDA版本
nvidia-smi

# 安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装CUDA版本的FAISS
pip install faiss-gpu
```

### 4.4 验证安装

```bash
# 验证PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# 验证CUDA支持
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# 验证FAISS
python -c "import faiss; print(f'FAISS: {faiss.__version__}')"

# 验证OpenCLIP
python -c "import open_clip; print(f'OpenCLIP: {open_clip.__version__}')"

# 验证Streamlit
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
```

---

## 5. VSCode配置

### 5.1 安装VSCode

**在Windows上安装**：

1. 下载VSCode：[https://code.visualstudio.com/](https://code.visualstudio.com/)
2. 运行安装程序
3. 选择"添加到PATH"选项

### 5.2 安装VSCode扩展

#### 必需扩展

| 扩展名称 | 用途 | 安装命令 |
|----------|------|---------|
| Python | Python语言支持 | `code --install-extension ms-python.python` |
| Remote - WSL | WSL远程开发 | `code --install-extension ms-vscode-remote.remote-wsl` |

#### 推荐扩展

| 扩展名称 | 用途 | 安装命令 |
|----------|------|---------|
| Pylance | Python智能提示 | `code --install-extension ms-python.pylance` |
| Jupyter | Notebook支持 | `code --install-extension ms-toolsaijupyter` |
| GitLens | Git增强 | `code --install-extension eamodio.gitlens` |
| Chinese (Simplified) | 中文界面 | `code --install-extension ms-ceintlvscode-language-pack-zh-hans` |

### 5.3 在WSL中打开项目

**方法一：使用VSCode命令（推荐）**

```bash
# 在WSL中进入项目目录
cd /home/yyh2004/demo

# 使用VSCode打开项目
code .
```

**方法二：从Windows资源管理器**

1. 打开Windows资源管理器
2. 在地址栏输入：`\\wsl$\Ubuntu\home\yyh2004\demo`
3. 右键点击文件夹，选择"通过VSCode打开"

**方法三：使用VSCode Remote-WSL扩展**

1. 打开VSCode
2. 点击左下角的绿色图标（><）
3. 选择"Connect to WSL"
4. 选择"Ubuntu-22.04"
5. 在WSL中打开项目文件夹

### 5.4 配置VSCode Python解释器

1. 打开VSCode
2. 按 `Ctrl+Shift+P` 打开命令面板
3. 输入"Python: Select Interpreter"
4. 选择虚拟环境中的Python解释器：
   - `/home/yyh2004/demo/venv/bin/python` (venv)
   - 或 `~/miniconda3/envs/rsicd/bin/python` (conda)

### 5.5 配置VSCode设置

**创建 `.vscode/settings.json` 文件**：

```json
{
    "python.defaultInterpreterPath": "/home/yyh2004/demo/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.cwd": "${workspaceFolder}",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": true
    }
}
```

### 5.6 配置VSCode任务

**创建 `.vscode/tasks.json` 文件**：

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "激活虚拟环境",
            "type": "shell",
            "command": "source venv/bin/activate",
            "problemMatcher": []
        },
        {
            "label": "运行Streamlit应用",
            "type": "shell",
            "command": "streamlit run main.py",
            "problemMatcher": [],
            "group": {
                "kind": "build",
                "isDefault": true
            }
        },
        {
            "label": "构建图像索引",
            "type": "shell",
            "command": "cd demo/preparation && python build_faiss_image.py",
            "problemMatcher": []
        },
        {
            "label": "构建文本索引",
            "type": "shell",
            "command": "cd demo/preparation && python build_faiss_text.py",
            "problemMatcher": []
        }
    ]
}
```

### 5.7 配置VSCode启动配置

**创建 `.vscode/launch.json` 文件**：

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Streamlit",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}",
            "env": {
                "PYTHONPATH": "${workspaceFolder}"
            }
        }
    ]
}
```

---

## 6. 项目运行

### 6.1 准备数据文件

#### 6.1.1 创建目录结构

```bash
# 进入项目目录
cd /home/yyh2004/demo

# 创建必要目录
mkdir -p demo/data
mkdir -p demo/rsicd_imgs
mkdir -p demo/preparation
mkdir -p demo/tools
```

#### 6.1.2 下载预训练模型

**方法一：从官方渠道下载**

```bash
# 下载RS5M模型权重
cd demo/data
wget https://github.com/your-repo/RS5M_ViT-B-32_RSICD.pt

# 或使用其他预训练模型
# 注意：如果没有RS5M模型，可以使用OpenAI预训练模型
```

**方法二：使用OpenAI预训练模型（临时方案）**

修改 `utils.py` 中的模型加载代码：

```python
@st.cache_resource
def load_model():
    # 使用OpenAI预训练模型
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    model.to(DEVICE)
    model.eval()

    return model, preprocess, tokenizer
```

#### 6.1.3 准备图像数据

```bash
# 将图像复制到数据集目录
cp /path/to/your/images/* demo/rsicd_imgs/

# 或使用符号链接
ln -s /path/to/your/images demo/rsicd_imgs
```

### 6.2 构建索引

#### 6.2.1 构建图像索引

```bash
# 确保虚拟环境已激活
source venv/bin/activate

# 进入预处理目录
cd demo/preparation

# 运行索引构建脚本
python build_faiss_image.py
```

**预期输出**：

```
Loading CLIP model...
Model loaded on cuda
Found 1000 images. Building index...
正在处理 1/1000: image1.jpg
正在处理 2/1000: image2.jpg
...
Index built successfully with 1000 images
Index saved to faiss_index.bin
Metadata saved to meta.json
```

#### 6.2.2 构建文本索引

```bash
# 确保文本描述文件存在
ls demo/tools/dataset_RSITMD.json

# 运行文本索引构建脚本
python build_faiss_text.py
```

### 6.3 运行应用

#### 6.3.1 在VSCode中运行

**方法一：使用集成终端**

1. 在VSCode中按 `Ctrl+~` 打开集成终端
2. 激活虚拟环境：
   ```bash
   source venv/bin/activate
   ```
3. 运行应用：
   ```bash
   streamlit run main.py
   ```

**方法二：使用任务**

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入"Tasks: Run Task"
3. 选择"运行Streamlit应用"

**方法三：使用调试器**

1. 打开 `main.py` 文件
2. 按 `F5` 或点击"Run and Debug"
3. 选择"Python: Streamlit"配置

#### 6.3.2 在浏览器中访问

应用启动后，VSCode会自动打开浏览器，或手动访问：

```
Local URL: http://localhost:8501
Network URL: http://172.23.248.1:8501
```

### 6.4 常用命令速查

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行应用
streamlit run main.py

# 指定端口运行
streamlit run main.py --server.port 8502

# 允许外部访问
streamlit run main.py --server.address 0.0.0.0

# 查看Streamlit日志
streamlit run main.py --logger.level debug

# 退出虚拟环境
deactivate
```

---

## 7. 常见问题

### 7.1 WSL相关问题

#### Q1: WSL无法访问Windows文件

**问题**：无法访问Windows的C盘或其他盘符

**解决方案**：

```bash
# Windows盘符在WSL中的挂载点
cd /mnt/c
cd /mnt/d

# 或使用\\wsl$路径从Windows访问
# 在Windows资源管理器中输入：\\wsl$\Ubuntu
```

#### Q2: WSL网络无法访问外网

**问题**：WSL中无法ping通外网或下载速度慢

**解决方案**：

```bash
# 临时解决方案：重启WSL网络服务
sudo service network-manager restart

# 或重启WSL
# 在Windows PowerShell中运行：
wsl --shutdown
wsl
```

#### Q3: WSL磁盘空间不足

**问题**：WSL虚拟磁盘空间不足

**解决方案**：

```powershell
# 在Windows PowerShell中运行
wsl --shutdown

# 压缩虚拟磁盘
Optimize-VHD -Path "\\wsl$\Ubuntu\ext4.vhdx" -Mode Full

# 或扩展虚拟磁盘大小
wsl --shutdown
wsl --set-default-version 2
# 在WSL中运行：
sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 7.2 Python环境问题

#### Q4: pip安装依赖失败

**问题**：pip install时出现编译错误或超时

**解决方案**：

```bash
# 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或使用阿里云镜像
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

# 升级pip和setuptools
pip install --upgrade pip setuptools wheel
```

#### Q5: CUDA版本不匹配

**问题**：PyTorch无法识别CUDA

**解决方案**：

```bash
# 检查CUDA版本
nvidia-smi

# 安装匹配的PyTorch版本
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Q6: 虚拟环境激活失败

**问题**：source venv/bin/activate报错

**解决方案**：

```bash
# 检查虚拟环境是否存在
ls -la venv/

# 重新创建虚拟环境
python3 -m venv venv --clear

# 或使用绝对路径
source /home/yyh2004/demo/venv/bin/activate
```

### 7.3 VSCode相关问题

#### Q7: VSCode无法识别Python解释器

**问题**：VSCode提示"Python extension not loaded"

**解决方案**：

1. 确保已安装Python扩展
2. 按 `Ctrl+Shift+P` 打开命令面板
3. 输入"Python: Select Interpreter"
4. 手动选择虚拟环境中的Python：
   ```
   /home/yyh2004/demo/venv/bin/python
   ```

#### Q8: VSCode终端无法激活虚拟环境

**问题**：在VSCode终端中运行source命令无效

**解决方案**：

**修改VSCode终端配置**：

在 `.vscode/settings.json` 中添加：

```json
{
    "terminal.integrated.profiles.linux": {
        "bash": {
            "path": "bash",
            "icon": "terminal-bash",
            "args": ["--login"]
        }
    }
}
```

或在终端中手动运行：

```bash
# 使用bash登录shell
bash -l

# 然后激活虚拟环境
source venv/bin/activate
```

### 7.4 项目运行问题

#### Q9: Streamlit无法启动

**问题**：运行streamlit run main.py报错

**解决方案**：

```bash
# 检查依赖是否完整
pip list | grep streamlit

# 重新安装Streamlit
pip uninstall streamlit
pip install streamlit

# 检查端口是否被占用
lsof -i :8501

# 使用其他端口
streamlit run main.py --server.port 8502
```

#### Q10: 模型权重文件缺失

**问题**：启动时提示"No pretrained weights loaded"

**解决方案**：

```bash
# 检查模型文件是否存在
ls -la demo/data/RS5M_ViT-B-32_RSICD.pt

# 如果不存在，使用OpenAI预训练模型
# 修改utils.py中的load_model函数
```

#### Q11: 索引文件缺失

**问题**：运行时提示"索引文件不存在"

**解决方案**：

```bash
# 检查索引文件
ls -la demo/data/faiss_index.bin
ls -la demo/data/meta.json

# 如果不存在，重新构建索引
cd demo/preparation
python build_faiss_image.py
python build_faiss_text.py
```

### 7.5 性能优化

#### Q12: 应用运行缓慢

**问题**：检索响应时间过长

**解决方案**：

```bash
# 检查GPU是否被使用
nvidia-smi

# 确保使用GPU版本的FAISS
pip uninstall faiss-cpu
pip install faiss-gpu

# 减少Top-K值
# 在应用界面中选择较小的Top-K值
```

#### Q13: 内存不足

**问题**：运行时出现CUDA out of memory错误

**解决方案**：

```bash
# 减少批处理大小
# 在代码中调整batch_size参数

# 使用CPU模式
# 设置DEVICE="cpu"在utils.py中

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache()"
```

---

## 8. 快速开始指南

### 8.1 一键配置脚本

创建 `setup.sh` 脚本：

```bash
#!/bin/bash

# 遥感图像-文本智能检索系统 - 快速配置脚本

echo "=== 开始配置环境 ==="

# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Python和pip
sudo apt install -y python3.10 python3.10-venv python3-pip

# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 升级pip
pip install --upgrade pip setuptools wheel

# 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install open-clip-torch streamlit faiss-gpu Pillow opencv-python numpy pandas

echo "=== 环境配置完成 ==="
echo "请运行: source venv/bin/activate"
echo "然后运行: streamlit run main.py"
```

**使用方法**：

```bash
# 赋予执行权限
chmod +x setup.sh

# 运行脚本
./setup.sh
```

### 8.2 验证配置

```bash
# 激活虚拟环境
source venv/bin/activate

# 运行验证脚本
python -c "
import torch
import faiss
import open_clip
import streamlit
import numpy as import numpy as np
import PIL

print('✅ 所有依赖安装成功！')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'FAISS: {faiss.__version__}')
print(f'OpenCLIP: {open_clip.__version__}')
print(f'Streamlit: {streamlit.__version__}')
"
```

---

## 9. 附录

### 9.1 文件结构

```
demo/
├── .vscode/                    # VSCode配置
│   ├── settings.json          # VSCode设置
│   ├── tasks.json            # 任务配置
│   └── launch.json           # 启动配置
├── venv/                       # Python虚拟环境
├── main.py                     # 主应用入口
├── utils.py                    # 工具函数模块
├── requirements.txt             # 依赖列表
├── SETUP_GUIDE.md            # 本配置指南
├── TECHNICAL_DOCUMENTATION.md  # 技术文档
├── data/                        # 数据目录
│   ├── faiss_index.bin       # 图像向量索引
│   ├── faiss_index_text.bin  # 文本向量索引
│   ├── meta.json            # 图像元数据
│   ├── meta_text.json       # 文本元数据
│   ├── RS5M_ViT-B-32_RSICD.pt  # 预训练模型权重
│   └── search_history.json   # 检索历史记录
├── rsicd_imgs/                  # 遥感图像库
├── preparation/                 # 数据预处理脚本
│   ├── build_faiss_image.py # 构建图像索引
│   └── build_faiss_text.py  # 构建文本索引
└── tools/                       # 辅助工具
    ├── dataset_rsicd.json   # RSICD数据集
    ├── dataset_RSITMD.json  # RSITMD数据集
    ├── captions.txt         # 图像描述文本
    ├── get_captions.py     # 描述提取工具
    └── img_name.py         # 图像命名工具
```

### 9.2 环境变量

在 `.vscode/settings.json` 或 `.env` 文件中配置：

```bash
# 设备配置
DEVICE=cuda  # 或 cpu

# 数据目录
IMAGES_DIR=/home/yyh2004/demo/rsicd_imgs
DATA_DIR=/home/yyh2004/demo/data

# 模型配置
MODEL_WEIGHTS=/home/yyh2004/demo/data/RS5M_ViT-B-32_RSICD.pt

# Streamlit配置
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### 9.3 参考资源

**官方文档**:
- [WSL文档](https://docs.microsoft.com/en-us/windows/wsl/)
- [VSCode文档](https://code.visualstudio.com/docs)
- [Python文档](https://docs.python.org/3/)
- [PyTorch文档](https://pytorch.org/docs/)
- [Streamlit文档](https://docs.streamlit.io/)

**社区资源**:
- [WSL GitHub](https://github.com/microsoft/WSL)
- [VSCode Python扩展](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [PyTorch论坛](https://discuss.pytorch.org/)

---

**文档版本**: v1.0  
**最后更新**: 2024-02-19  
**维护者**: Development Team
