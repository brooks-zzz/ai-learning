from typing import Any
from langchain_core.documents import Document
from pypdf import PdfReader
from pathlib import Path


def get_file_data(path:str) -> list[Any]:
    """
    使用pypdf的demo
    获取pdf文字信息
    :return: 返回字典{页码：每页内容}
    """

    #该位置加个方法获取文件名称，正式环境不需要，因为会在关系型数据库用冗余字段存储
    filename = Path(path).name

    reader = PdfReader(path)
    # 获取页数
    page_count = len(reader.pages)  # [citation:1][citation:7][citation:12]
    # 提取页面文本
    docs = []

    for page in reader.pages:
        text = page.extract_text()
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "page": page.page_number,
                    "source": filename
                }
            )
        )
    return docs