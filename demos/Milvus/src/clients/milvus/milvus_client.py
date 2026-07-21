import os
import yaml
from pymilvus import MilvusClient

# 全局变量
_client = None


def get_milvus_client(config_path="src/config.yaml") -> MilvusClient:
    """
    获取全局唯一的 MilvusClient 实例。
    不需要定义类，直接用函数和全局变量实现单例。
    """
    global _client  # 声明使用全局变量

    if _client is None:
        # 处理路径偏离问题
        if not os.path.exists(config_path) and os.path.exists("config.yaml"):
            config_path = "config.yaml"

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        milvus_config = config.get("milvus", {})
        host = milvus_config.get("host", "localhost")
        port = milvus_config.get("port", 19530)

        # 实例化官方的原生客户端
        _client = MilvusClient(uri=f"http://{host}:{port}")
        print(f"[Milvus] 成功初始化全局连接: http://{host}:{port}")

    return _client