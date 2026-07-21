# AI Engineering Learning Repository

一个用于学习和实践 AI 应用开发的工程化仓库。

本项目主要围绕 **LLM 应用开发、RAG（Retrieval-Augmented Generation）、Embedding、向量数据库、Agent 等方向**进行学习和实践。

仓库采用 **Demo 验证 → 技术沉淀 → 项目集成** 的方式组织代码，记录 AI 技术探索过程，并逐步形成可复用的 AI 应用开发基础框架。

---

## 项目结构

```text
ai-learning/
│
├── projects/                 # 正式项目
│   └── rag-platform/         # AI应用主项目
│
├── demos/                    # 技术验证Demo
│   ├── 01_embedding/         # Embedding模型实践
│   ├── 02_milvus/            # 向量数据库实践
│   ├── 03_rag/               # RAG流程实践
│   └── 04_agent/             # Agent实践
│
├── docs/                     # 技术学习文档
│   ├── embedding.md
│   ├── milvus.md
│   └── rag.md
│
├── requirements.txt          # Python依赖
├── environment.yml           # Conda环境配置
└── README.md
```

---

# 学习路线

## 1. 基础模型能力

### Embedding

学习内容：

* 文本向量化
* Embedding模型使用
* 相似度计算
* 向量检索基础

实践：

* BGE系列模型
* 文本向量生成
* 向量相似度搜索

---

## 2. 向量数据库

### Milvus

学习内容：

* Collection设计
* Schema设计
* Index建立
* 向量插入
* 向量检索

实践：

* 本地Milvus部署
* 文档向量存储
* 相似问题查询

---

## 3. RAG应用开发

学习内容：

* 文档加载
* 文本切割
* Embedding生成
* 向量检索
* Prompt构造
* LLM生成答案

目标：

构建企业知识库问答系统。

应用场景：

* 企业制度查询
* 技术文档助手
* MES/WMS/LIMS知识助手
* 内部知识库问答

---

## 4. Agent应用

学习内容：

* Tool调用
* Function Calling
* 工作流编排
* 多Agent协作

目标：

探索AI Agent在企业业务系统中的落地。

---

# 开发环境

## Python

```text
Python >= 3.10
```

## Conda环境

创建环境：

```bash
conda create -n ai-learning python=3.10
```

进入环境：

```bash
conda activate ai-learning
```

安装依赖：

```bash
pip install -r requirements.txt
```

---

# 技术栈

## AI模型

* BGE-M3
* OpenAI API
* 本地LLM模型

## 向量数据库

* Milvus

## Python生态

* LangChain
* LlamaIndex
* FastAPI
* Pydantic

## 工程能力

* Docker
* Git
* Linux
* Conda

---

# 项目开发流程

本仓库采用以下开发模式：

```
技术探索
    |
    ↓
Demo验证
    |
    ↓
代码优化
    |
    ↓
抽象封装
    |
    ↓
集成主项目
```

Demo主要用于：

* 验证技术可行性
* 快速学习新组件
* 记录实验过程

Projects主要用于：

* 集成稳定代码
* 模拟真实业务项目
* 构建工程化AI应用

---

# 目标

通过持续实践，掌握：

* AI应用后端开发能力
* RAG系统设计能力
* 向量数据库应用能力
* LLM工程化落地能力

最终能够独立完成：

```
用户请求
    |
    ↓
业务服务
    |
    ↓
知识检索(RAG)
    |
    ↓
LLM推理
    |
    ↓
业务结果输出
```

---

# Learning Notes

学习过程中会持续记录：

* 技术原理
* 使用方式
* 遇到的问题
* 解决方案
* 工程实践经验

---

# License

仅用于个人学习和技术研究。

