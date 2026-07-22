import datetime
from typing import List, Optional

from pydantic import Field


class DocModel:
    """
            文档元数据
            """
    id: str = Field(..., description="文档唯一ID")

    vector: List[float] = Field(..., description="文件名")
    text: str = Field(..., description="文件类型，例如 pdf/docx/ppt")

    page: Optional[str] = Field(None, description="作者")
    title: Optional[str] = Field(None, description="所属部门")