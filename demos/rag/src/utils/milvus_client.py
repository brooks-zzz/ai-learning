import os
from typing import List

import yaml
from pymilvus import MilvusClient, DataType

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

def insert_collection(collection_name: str, data: List[dict]):
    if not (_client.has_collection(collection_name)):
        create_collection(collection_name)
    value = [ item.__dict__ for item in data]
    _client.insert(collection_name, value)

def create_collection(collection_name: str):
    #
    schema = _client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id",datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector",datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text",datatype=DataType.VARCHAR, max_length=512)
    #因为是做demo，元数据相关字段就只加入一个页码，其他先不加了，以后集成时在留出逻辑外键做关系型数据库的关联
    schema.add_field(field_name="page",datatype=DataType.INT32)
    schema.add_field(field_name="title",datatype=DataType.VARCHAR, max_length=100)

    index_params = _client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    _client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        metric_type="COSINE"
    )


#本来不想在demo里把这个类搞的太复杂，没办法要修改text字段长度，只能再加一个collection的schema更新方法了，入参就不控制封装了，先完成demo为主
#感觉以后的milvus支持类可以直接拿这个文件当模板里
def update_collection(collection_name: str):
    #先释放集合，不然直接修改会报错
    _client.release_collection(collection_name)
    # 3. 修改字段长度
    # 假设要修改一个名为 "varchar_field" 的 VarChar 字段，将最大长度设为 1024
    field_params = {"max_length": 4096}  # [citation:9][citation:11]

    _client.alter_collection_field(
        collection_name=collection_name,
        field_name="text",
        field_params=field_params
    )