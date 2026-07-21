from pymilvus import DataType

from clients.milvus.milvus_client import get_milvus_client

def create_collection(collection_name: str):
    client = get_milvus_client()
    #
    schema = client.create_schema(
        auto_id=True,
        enable_dynamic_field=True,
    )
    schema.add_field(field_name="id",datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="vector",datatype=DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field(field_name="text",datatype=DataType.VARCHAR, max_length=512)
    #
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="vector",
        index_type="FLAT",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200}
    )

    client.create_collection(
        collection_name=collection_name,
        schema=schema,
        index_params=index_params,
        metric_type="COSINE"
    )


def set_collection(collection_name: str):
    client = get_milvus_client()

    # 控制自增时是否使用入参主键
    # client.alter_collection_properties(
    #     collection_name=collection_name,
    #     properties={"allow_insert_auto_id": "false"}
    # )

    # 删除集合
    client.drop_collection(
        collection_name=collection_name
    )

