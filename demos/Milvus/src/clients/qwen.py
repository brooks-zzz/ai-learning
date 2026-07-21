import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"

client = OpenAI(
    api_key = os.getenv("DASHSCOPE_API_KEY"),
    base_url = os.getenv("DASHSCOPE_BASE_URL")
)


def send_message(messages: List[dict] = None):
    """
    调用大模型对话
    :param messages:一个字典列表，每个字典包含 "role" 和 "content" 键。
                 例如: [{"role": "system", "content": "你是个诗人"},
    :return:
    """

    if messages== None:
        messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "地球是"},
    ]
    else:
        messages=messages
    result = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-plus",
        messages=messages,
    )
    return result.model_dump_json()

