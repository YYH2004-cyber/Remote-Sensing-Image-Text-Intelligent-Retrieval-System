#!/bin/bash

# 遥感图像-文本智能检索系统 - 快速配置脚本
# 版本: v1.0
# 更新日期: 2024-02-19

set -e  # 遇到错误立即退出

echo "========================================"
echo "  遥感图像-文本智能检索系统"
echo "  快速配置脚本"
echo "========================================"
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查是否在WSL中
check_wsl() {
    if grep -q Microsoft /proc/version; then
        echo -e "${GREEN}✓ 检测到WSL环境${NC}"
        return 0
    else
        echo -e "${RED}✗ 未检测到WSL环境${NC}"
        echo -e "${YELLOW}请在WSL中运行此脚本${NC}"
        exit 1
    fi
}

# 检查Python版本
check_python() {
    echo -n "检查Python版本... "
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        echo -e "${GREEN}✓ Python $PYTHON_VERSION${NC}"
        
        if [ "$(echo "$PYTHON_VERSION < 3.10" | bc)" -eq 1 ]; then
            echo -e "${YELLOW}⚠ Python版本过低，建议升级到3.10+${NC}"
        fi
        return 0
    else
        echo -e "${RED}✗ 未找到Python3${NC}"
        return 1
    fi
}

# 更新系统包
update_system() {
    echo ""
    echo "========================================"
    echo "  步骤 1/6: 更新系统包"
    echo "========================================"
    echo ""
    
    echo -n "更新apt包列表... "
    sudo apt update > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
    
    echo -n "升级已安装的包... "
    sudo apt upgrade -y > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
}

# 安装Python和pip
install_python() {
    echo ""
    echo "========================================"
    echo "  步骤 2/6: 安装Python和pip"
    echo "========================================"
    echo ""
    
    sudo apt install -y python3.10 python3.10-venv python3-pip python3-dev > /dev/null 2>&1
    echo -e "${GREEN}✓ Python 3.10 安装完成${NC}"
    
    # 设置默认Python版本
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 > /dev/null 2>&1
    echo -e "${GREEN}✓ Python 3.10 已设置为默认版本${NC}"
}

# 创建虚拟环境
create_venv() {
    echo ""
    echo "========================================"
    echo "  步骤 3/6: 创建虚拟环境"
    echo "========================================"
    echo ""
    
    if [ -d "venv" ]; then
        echo -e "${YELLOW}⚠ 虚拟环境已存在${NC}"
        read -p "是否删除并重新创建？(y/n) " -n 1 -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf venv
            echo -e "${GREEN}✓ 已删除旧虚拟环境${NC}"
        else
            echo -e "${GREEN}✓ 使用现有虚拟环境${NC}"
            return 0
        fi
    fi
    
    echo -n "创建虚拟环境... "
    python3 -m venv venv > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
    
    echo -e "${GREEN}✓ 虚拟环境创建完成${NC}"
    echo ""
    echo -e "${YELLOW}激活虚拟环境命令: source venv/bin/activate${NC}"
}

# 升级pip
upgrade_pip() {
    echo ""
    echo "========================================"
    echo "  步骤 4/6: 升级pip"
    echo "========================================"
    echo ""
    
    source venv/bin/activate
    
    echo -n "升级pip... "
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
    
    deactivate
}

# 安装依赖
install_dependencies() {
    echo ""
    echo "========================================"
    echo "  步骤 5/6: 安装项目依赖"
    echo "========================================"
    echo ""
    
    source venv/bin/activate
    
    # 检查是否有GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo -e "${GREEN}✓ 检测到GPU: $GPU_INFO${NC}"
        echo ""
        read -p "是否安装GPU版本的依赖？(y/n) " -n 1 -r
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo -n "安装PyTorch (CUDA 11.8)... "
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 > /dev/null 2>&1
            echo -e "${GREEN}✓ 完成${NC}"
            
            echo -n "安装FAISS-GPU... "
            pip install faiss-gpu > /dev/null 2>&1
            echo -e "${GREEN}✓ 完成${NC}"
        else
            install_cpu_dependencies
        fi
    else
        echo -e "${YELLOW}⚠ 未检测到GPU，将安装CPU版本${NC}"
        install_cpu_dependencies
    fi
    
    deactivate
}

install_cpu_dependencies() {
    echo -n "安装PyTorch (CPU)... "
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
    
    echo -n "安装FAISS-CPU... "
    pip install faiss-cpu > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
    
    echo -n "安装其他依赖... "
    pip install open-clip-torch streamlit Pillow opencv-python numpy pandas > /dev/null 2>&1
    echo -e "${GREEN}✓ 完成${NC}"
}

# 创建目录结构
create_directories() {
    echo ""
    echo "========================================"
    echo "  步骤 6/6: 创建目录结构"
    echo "========================================"
    echo ""
    
    mkdir -p demo/data
    mkdir -p demo/rsicd_imgs
    mkdir -p demo/preparation
    mkdir -p demo/tools
    mkdir -p .vscode
    
    echo -e "${GREEN}✓ 目录结构创建完成${NC}"
    echo ""
    echo "创建的目录:"
    echo "  - demo/data/"
    echo "  - demo/rsicd_imgs/"
    echo "  - demo/preparation/"
    echo "  - demo/tools/"
    echo "  - .vscode/"
}

# 创建VSCode配置
create_vscode_config() {
    echo ""
    echo "========================================"
    echo "  创建VSCode配置"
    echo "========================================"
    echo ""
    
    # 创建settings.json
    cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "/home/yyh2004/demo/venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.cwd": "\${workspaceFolder}",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.fixAll": true
    }
}
EOF
    echo -e "${GREEN}✓ .vscode/settings.json 创建完成${NC}"
    
    # 创建tasks.json
    cat > .vscode/tasks.json << EOF
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
EOF
    echo -e "${GREEN}✓ .vscode/tasks.json 创建完成${NC}"
    
    # 创建launch.json
    cat > .vscode/launch.json << EOF
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Streamlit",
            "type": "python",
            "request": "launch",
            "program": "\${workspaceFolder}/main.py",
            "console": "integratedTerminal",
            "cwd": "\${workspaceFolder}",
            "env": {
                "PYTHONPATH": "\${workspaceFolder}"
            }
        }
    ]
}
EOF
    echo -e "${GREEN}✓ .vscode/launch.json 创建完成${NC}"
}

# 验证安装
verify_installation() {
    echo ""
    echo "========================================"
    echo "  验证安装"
    echo "========================================"
    echo ""
    
    source venv/bin/activate
    
    echo -n "验证PyTorch... "
    python -c "import torch; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    echo -n "验证FAISS... "
    python -c "import faiss; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    echo -n "验证OpenCLIP... "
    python -c "import open_clip; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    echo -n "验证Streamlit... "
    python -c "import streamlit; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    echo -n "验证NumPy... "
    python -c "import numpy; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    echo -n "验证PIL... "
    python -c "from PIL import Image; print('OK')" 2>/dev/null && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"
    
    deactivate
}

# 显示完成信息
show_completion() {
    echo ""
    echo "========================================"
    echo "  配置完成！"
    echo "========================================"
    echo ""
    echo -e "${GREEN}✓ 环境配置完成${NC}"
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 激活虚拟环境:"
    echo -e "   ${YELLOW}source venv/bin/activate${NC}"
    echo ""
    echo "2. 在VSCode中打开项目:"
    echo -e "   ${YELLOW}code .${NC}"
    echo ""
    echo "3. 运行应用:"
    echo -e "   ${YELLOW}streamlit run main.py${NC}"
    echo ""
    echo "4. 在浏览器中访问:"
    echo -e "   ${YELLOW}http://localhost:8501${NC}"
    echo ""
    echo "详细配置说明请查看: SETUP_GUIDE.md"
    echo ""
}

# 主函数
main() {
    echo ""
    echo -e "${GREEN}开始配置环境...${NC}"
    echo ""
    
    check_wsl
    check_python
    
    read -p "是否继续配置？(y/n) " -n 1 -r
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "配置已取消"
        exit 0
    fi
    
    update_system
    install_python
    create_venv
    upgrade_pip
    install_dependencies
    create_directories
    create_vscode_config
    verify_installation
    show_completion
}

# 运行主函数
main
