#!/bin/bash
export HF_TOKEN=YOUR_HF_TOKEN_HERE
PYTHON=/home/user/miniconda3/envs/vllm/bin/python

echo '[1/3] Downloading EXAONE-3.5-7.8B-Instruct...'
$PYTHON -c "
from huggingface_hub import snapshot_download, login
login(token='$HF_TOKEN')
snapshot_download('LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct', local_dir='/home/user/models/EXAONE-3.5-7.8B-Instruct')
print('EXAONE-3.5-7.8B done')
" >> /home/user/CBNU/logs/download.log 2>&1

echo '[2/3] Downloading Llama-3.1-8B-Instruct...'
$PYTHON -c "
from huggingface_hub import snapshot_download, login
login(token='$HF_TOKEN')
snapshot_download('meta-llama/Llama-3.1-8B-Instruct', local_dir='/home/user/models/Llama-3.1-8B-Instruct')
print('Llama-3.1-8B done')
" >> /home/user/CBNU/logs/download.log 2>&1

echo '[3/3] Downloading Qwen3-8B...'
$PYTHON -c "
from huggingface_hub import snapshot_download, login
login(token='$HF_TOKEN')
snapshot_download('Qwen/Qwen3-8B', local_dir='/home/user/models/Qwen3-8B')
print('Qwen3-8B done')
" >> /home/user/CBNU/logs/download.log 2>&1

echo 'ALL DOWNLOADS COMPLETE' >> /home/user/CBNU/logs/download.log
