#!/bin/bash

# Word2Vec环境一键安装脚本
# 使用方法: bash install_word2vec_env.sh

set -e

ENV_NAME="word2vec"

echo "=== Word2Vec环境安装脚本 ==="

# 检查conda是否可用
if ! command -v conda &> /dev/null; then
    echo "错误: 未找到conda，请先安装miniconda或anaconda"
    exit 1
fi

# 初始化conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh

# 检查环境是否已存在
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "警告: 环境 ${ENV_NAME} 已存在"
    read -p "是否删除并重新创建? (y/n): " choice
    if [ "$choice" = "y" ]; then
        conda env remove -n ${ENV_NAME} -y
    else
        echo "跳过安装"
        exit 0
    fi
fi

echo "正在创建conda环境 ${ENV_NAME} (Python 3.13)..."
conda create -n ${ENV_NAME} python=3.13 -y

echo "激活环境..."
conda activate ${ENV_NAME}

echo "安装核心依赖..."
# 安装PyTorch (使用清华镜像)
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装NLP相关包..."
pip install gensim jieba transformers datasets tokenizers accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装数据科学包..."
pip install numpy pandas scipy scikit-learn matplotlib seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装工具包..."
pip install tqdm pyyaml requests huggingface_hub -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "安装GPU监控工具..."
pip install nvitop -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=== 安装完成 ==="
echo "使用以下命令激活环境:"
echo "  conda activate ${ENV_NAME}"
echo ""
