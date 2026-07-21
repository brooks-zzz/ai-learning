from joblib.externals.loky.backend import context
import json
from clients.milvus.milvus_client import get_milvus_client
from clients.milvus.collection_opreate import create_collection
from FlagEmbedding import BGEM3FlagModel
from clients.qwen import send_message

def insert_data():
    print("start insert data...")
    client = get_milvus_client()
    if not (client.has_collection(collection_name="demo_collection")):
        create_collection("demo_collection")
    documents = [
        "狗",
        "猫",
        "兔子","老鼠",
        "水稻",
        "小麦",
        "玉米","鲫鱼",
        "bk喜欢吃水里的",
        "水里的",
        "鲤鱼",
        "小麦兔子狗老鼠在陆地上",
        "水稻、鲫鱼、鲤鱼在水里"
        "犬类",
        "鱼类生活在水里"
        "哺乳动物",
        "无脊椎动物",
        "章鱼"
    ]
    print("生成文檔向量......")

    embeddings = model.encode(documents, return_dense=True)['dense_vecs']

    # 插入數據到milvus
    data = [
        {"vector": embeddings[i].tolist(), "text": documents[i]}
        for i in range(len(documents))
    ]
    client.insert(collection_name="demo_collection", data=data)
    print(f"成功插入{len(data)}條數據")

def get_data(param: str):
    print("start search")
    # model = BGEM3FlagModel("BAAI/bge-m3")
    model = BGEM3FlagModel("/home/brooks/models/bge-m3", use_fp16=False)
    query_vector = model.encode([param], return_dense=True)['dense_vecs']
    client = get_milvus_client()
    collection_name = "demo_collection"
    search_params = {
        "metric_type": "COSINE",
        "params": {"drop_ratio_search": 0.8},  # the ratio of small vector values to be dropped during search.
    }
    # 加载 Collection 到内存
    client.load_collection("demo_collection")
    print("Collection 已加载")

    results = client.search(
        collection_name=collection_name,
        data=query_vector,
        limit=10,
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
    # # 3. 解析结果
    # print("咬人")
    # for hits in results:
    #     for hit in hits:
    #         print(f"匹配到的文本: {hit['entity']['text']}")
    #         print(f"相似度得分(Score): {hit['distance']}")
    #         print("-" * 20)

def rag_search(user_query: str):
    print("start rag search...")
    #获取milvus关联都较高内容
    print("get relation information...")
    results = get_data(user_query)
    #进行内容过滤
    filtered_results = list(filter(lambda item: item["distance"]>0.3, results[0]))
    #拼接内容
    final_context= [item["text"] for item in filtered_results]
    #构造提示词
    system_prompt = {"role": "system", "content": "你是一个百科老师"}
    user_prompt = {"role": "user", "content": f"""参考资料:"{final_context}
                   "问题:"{user_query}
                   """}
    #调用大模型生成答案
    message = [
        {
            "role": "system",
            # 严格约束模型：只允许基于传入的 context 回答，防带偏
            "content": (
                "你是一个严谨的知识库回答助手。\n"
                "1. 请完全基于提供的【参考资料】回答【问题】。\n"
                "2. 如果资料中有相关推导，请给出合理的解答。\n"
                "3. 如果参考资料中完全没有提及相关信息，请直接回答：'根据已知资料无法回答'，严禁凭空捏造！"
            )
        },
        {
            "role": "user",
            # 注意这里的键名必须是 content
            "content": f"【参考资料】:\n{final_context}\n\n【问题】:\n{user_query}"
        }
    ]
    result = send_message(message)
    data = json.loads(result)
    print("Q:"+data['choices'][0]['message']['content'])

if __name__ == "__main__":
    #
    rag_search("bk喜欢吃什么")
    # set_collection("demo_collection")
    #
    #test insert
    # insert_data()
    # # 測試查詢
    # get_data()

