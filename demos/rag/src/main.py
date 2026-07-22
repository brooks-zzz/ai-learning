from dotenv import load_dotenv

import model.doc_model
import os
from model.doc_model import DocModel
from utils.milvus_client import *
from utils.pdf_util import get_file_data
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter



def insert_base_data():
    # 构思，第一步 读取静态文档
    pdf_data = get_file_data("resource/员工手册.pdf")
    # 处理数据，插入向量库 #电脑没内存里先舍弃使用beg-m3，优先使用阿里在线向量模型
    # 必须在创建 embeddings 对象之前设置环境变量
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "请设置 DASHSCOPE_API_KEY"
        )
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=api_key
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128
    )
    datas = []
    # 切片
    chunks = splitter.split_documents(pdf_data)
    # 取出所有的片
    texts = [
        chunk.page_content
        for chunk in chunks
    ]
    vectors = embeddings.embed_documents(texts)
    for chunk, vector in zip(chunks, vectors):
        data = DocModel()
        data.vector = vector
        data.text = chunk.page_content
        data.title = chunk.metadata["source"]
        data.page = chunk.metadata["page"]
        datas.append(data)
    # 插入向量库
    insert_collection("im_collection", datas)

def get_data(param: str):
    print("start search"+param)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v4",
        dashscope_api_key=api_key
    )
    query_vector = embeddings.embed_query(param)
    client = get_milvus_client()
    collection_name = "im_collection"
    search_params = {
        "metric_type": "COSINE",
        "params": {"drop_ratio_search": 0.8},  # the ratio of small vector values to be dropped during search.
    }
    # 加载 Collection 到内存
    client.load_collection("im_collection")
    print("Collection 已加载")

    results = client.search(
        collection_name=collection_name,
        data=[query_vector],
        limit=5,
        anns_field="vector",
        output_fields=["text", "id"],
        search_params=search_params,
    )
    for hits in results:
        for hit in hits:
            print(f"匹配到的文本: {hit['entity']['text']}")
            print(f"相似度得分(Score): {hit['distance']}")
            print("-" * 20)
    return results



if __name__ == '__main__':
    load_dotenv()
    get_milvus_client()
    # 第一步准备基础数据
    #update_collection("im_collection")
    #insert_base_data()
    # 第二步，使用langchain框架
    get_data("我做什么事会被开除")

# 第三部发出message后加载问答链转向量去向量库查看相关数据

    # 对接大模型-----暂时去掉

    # 返回时携带数据来源
