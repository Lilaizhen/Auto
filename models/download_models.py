import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# 模型名称和保存路径
model_name = "meta-llama/Llama-2-7b-chat-hf"
base_model_path = "./llama2/llama-2-7b-chat-hf"

# 检查目标目录是否存在，不存在则创建
if not os.path.exists(base_model_path):
    os.makedirs(base_model_path)

# 使用您的 Hugging Face 访问令牌
your_token = "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"

# 加载分词器和模型，使用认证令牌
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, use_auth_token=your_token)
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16,
                                             low_cpu_mem_usage=True,
                                             use_cache=False,
                                             use_auth_token=your_token)

# 保存模型和分词器到本地
model.save_pretrained(base_model_path, from_pt=True)
tokenizer.save_pretrained(base_model_path, from_pt=True)
