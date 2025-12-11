# Task07 大模型应用

## 1 LLM 的评测

- LLM 的评测数据集：通用评测集（MMLU）、工具使用评测集（BFCL V2、Nexus）、数学评测集（GSM8K、MATH）、推理评测集（ARC Challenge、GPQA、HellaSwag）、长文本理解评测集（InfiniteBench/En.MC、NIH/Multi-needle）、多语言评测集（MGSM）
- 主流评测榜单：Open LLM Leaderboard、Lmsys Chatbot Arena Leaderboard、OpenCompass。
- 垂类评测榜单：金融榜（基于 CFBenchmark 评测集）、安全榜（基于 Flames 评测集）、通识榜（基于 BotChat 评测集）、法律榜（基于 LawBench 评测集）、医疗榜（基于 MedBench 评测集）

## 2 RAG

### 2.1 RAG 原理

将“检索”与“生成”结合，当用户提出查询时，系统首先通过检索模块找到与问题相关的文本片段，然后将这些片段作为附加信息传递给语言模型，模型据此生成更为精准和可靠的回答。

### 2.2 搭建一个 RAG 框架

- RAG 基本结构：向量化模块、文档加载和切分模块、数据库、检索模块、大模型模块。
- RAG 主要流程：索引、检索、生成。

1. 加载 python 依赖库

```python
import json
import os
import re
from typing import List

import PyPDF2
import markdown
import numpy as np
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv, find_dotenv
from tqdm import tqdm
```

2. 加载环境变量，用于加载 API_KEY。

```python
loaded = load_dotenv(find_dotenv(), override=True)
```

3. 实现 RAG 向量化

```python
class BaseEmbeddings:
    """
    向量化基类
    """

    def __init__(self, path: str, is_api: bool) -> None:
        self.path = path
        self.is_api = is_api

    def get_embedding(self, text: str, model: str=None) -> List[float]:
        raise NotImplementedError

    @classmethod
    def cosine_similarity(cls, vector1: List[float], vector2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        :param vector1: 向量1
        :param vector2: 向量2
        :return: 两个向量的相似度
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
```

```python
class SiliconFlowEmbedding(BaseEmbeddings):
    """
    基于硅基流动的向量化类
    """

    def __init__(self, path: str = '', is_api: bool = True) -> None:
        super().__init__(path, is_api)
        if self.is_api:
            from openai import OpenAI
            API_KEY = os.getenv("SiliconFlow_API_KEY")
            BASE_URL = os.getenv("SiliconFlow_BASE_URL")
            self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def get_embedding(self, text: str, model: str = "BAAI/bge-m3") -> List[float]:
        if self.is_api:
            text = text.replace("\n", " ")
            return self.client.embeddings.create(input=[text], model=model).data[0].embedding
        else:
            raise NotImplementedError
```

4. 实现文档加载和切分

```python
class ReadFiles:
    """
    class to read files
    """
    enc = tiktoken.get_encoding("cl100k_base")

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # args：dir_path，目标文件夹路径
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk 函数将递归遍历指定文件夹
            for filename in filenames:
                # 通过后缀名判断文件类型是否满足要求
                if filename.endswith(".md"):
                    # 如果满足要求，将其绝对路径加入到结果列表
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # 读取文件内容
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def read_file_content(cls, file_path: str):
        """
        根据文件扩展名选择读取方法
        """
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        chunk_text = []

        curr_len = 0
        curr_chunk = ''

        token_len = max_token_len - cover_content
        lines = text.splitlines()  # 假设以换行符分割文本为行

        for line in lines:
            line = line.replace(' ', '')
            line_len = len(cls.enc.encode(line))
            if line_len > max_token_len:
                # 如果单行长度就超过限制，则将其分割成多个块
                num_chunks = (line_len + token_len - 1) // token_len
                for i in range(num_chunks):
                    start = i * token_len
                    end = start + token_len
                    # 避免跨单词分割
                    while not line[start:end].rstrip().isspace():
                        start += 1
                        end += 1
                        if start >= line_len:
                            break
                    curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                    chunk_text.append(curr_chunk)
                # 处理最后一个块
                start = (num_chunks - 1) * token_len
                curr_chunk = curr_chunk[-cover_content:] + line[start:end]
                chunk_text.append(curr_chunk)

            if curr_len + line_len <= token_len:
                curr_chunk += line
                curr_chunk += '\n'
                curr_len += line_len
                curr_len += 1
            else:
                chunk_text.append(curr_chunk)
                curr_chunk = curr_chunk[-cover_content:] + line
                curr_len = line_len + cover_content

        if curr_chunk:
            chunk_text.append(curr_chunk)

        return chunk_text

    @classmethod
    def read_pdf(cls, file_path: str):
        # 读取PDF文件
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # 读取Markdown文件
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # 使用BeautifulSoup从HTML中提取纯文本
            soup = BeautifulSoup(html_text, 'html.parser')
            plain_text = soup.get_text()
            # 使用正则表达式移除网址链接
            text = re.sub(r'http\S+', '', plain_text)
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # 读取文本文件
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
```

5. 实现数据库与向量检索

```python
class VectorStore:
    def __init__(self, document=None) -> None:
        self.vectors = None
        if document is None:
            document = ['']
        self.document = document

    def get_vector(self, EmbeddingModel: BaseEmbeddings) -> List[List[float]]:
        self.vectors = []
        for doc in tqdm(self.document, desc="Calculating embeddings"):
            self.vectors.append(EmbeddingModel.get_embedding(doc))
        return self.vectors

    def persist(self, path: str = '../../storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/doecment.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/doecment.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        return BaseEmbeddings.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel: BaseEmbeddings, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.get_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector)
                           for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()
```

6. 实现大模型模块

```python
class BaseModel:
    def __init__(self, path: str = '') -> None:
        self.path = path

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        pass

    def load_model(self):
        pass
```

```python
class SiliconFlowChat(BaseModel):
    def __init__(self, path: str = '', model: str = "Qwen/Qwen3-8B") -> None:
        super().__init__(path)
        self.model = model

    def chat(self, prompt: str, history: List[dict], content: str) -> str:
        from openai import OpenAI
        API_KEY = os.getenv("SiliconFlow_API_KEY")
        BASE_URL = os.getenv("SiliconFlow_BASE_URL")
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL, max_retries=3)
        history.append({'role': 'user',
                        'content': PROMPT_TEMPLATE['RAG_PROMPT_TEMPLATE'].format(question=prompt, context=content)})
        response = client.chat.completions.create(
            model=self.model,
            messages=history,
            max_tokens=1024,
            temperature=0.1
        )
        return response.choices[0].message.content
```

7. 用一个字典来保存所有的 prompt，方便维护

```python
PROMPT_TEMPLATE = dict(
    RAG_PROMPT_TEMPLATE="""使用以上下文来回答用户的问题。如果你不知道答案，就说你不知道。总是使用中文回答。
        问题: {question}
        可参考的上下文：
        ···
        {context}
        ···
        如果给定的上下文无法让你做出回答，请回答数据库中没有这个内容，你不知道。
        有用的回答:"""
)
```

8. 开始基于知识库聊天了，我们上传了一个 Git 介绍的文档，然后可以针对这个文档来提问

```python
# 没有保存数据库
rf = ReadFiles('../data')
docs = rf.get_content(max_token_len=600, cover_content=150)  # 获取data目录下的所有文件内容并分割
vector = VectorStore(docs)
embedding = SiliconFlowEmbedding()  # 创建EmbeddingModel
vector.get_vector(EmbeddingModel=embedding)
# 将向量和文档内容保存到storage目录，下次再用可以直接加载本地数据库
vector.persist(path='../storage')

question = 'git的原理是什么？'

rag_content = vector.query(question, embedding, k=1)[0]
chat = SiliconFlowChat(model="deepseek-ai/DeepSeek-V3")
print(chat.chat(question, [], rag_content))
```

    Calculating embeddings: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 10.59it/s]
    

    Git的原理主要基于以下核心概念和机制：
    
    1. 三大区域结构：
    - 工作区：直接编辑文件的物理目录
    - 暂存区（Index）：临时存储变更的快照区
    - 版本库：永久存储项目历史的数据库
    
    2. 三大状态转换：
    - 已修改 → git add → 已暂存
    - 已暂存 → git commit → 已提交
    - 已提交 → git checkout → 工作区
    
    3. 分布式架构：
    每个开发者本地都有完整的版本库，包含全部历史记录，通过推送(push)/拉取(pull)实现协作。
    
    4. 数据存储原理：
    使用SHA-1哈希算法生成唯一对象ID，以键值对形式存储：
    - blob对象：存储文件内容
    - tree对象：记录目录结构
    - commit对象：包含提交元数据和指向tree的指针
    
    5. 版本控制机制：
    通过有向无环图(DAG)管理提交历史，分支只是指向特定提交的可变指针。
    
    这种设计使得Git具有强大的分支管理能力、高效的本地操作和完整的历史追溯功能。
    

## 3 Agent

### 3.1 LLM Agent

- LLM Agent 简介：大模型 Agent 是一个以 LLM 为核心“大脑”，并赋予其自主规划、记忆和使用工具能力的系统。
- LLM Agent 的类型：
    1. 任务导向型 Agent：专注于完成特定领域的、定义明确的任务，使用预设的流程和可调用的特定工具集。
    2. 规划与推理型 Agent：强调自主分解复杂任务、制定多步计划，采用特定的思维框架。
    3. 多 Agent 系统：由多个具有不同角色或能力的 Agent 协同工作，共同完成一个更宏大的目标。
    4. 探索与学习型 Agent： 不仅执行任务，还能在与环境的交互中主动学习新知识、新技能或优化自身策略，可能包含更复杂的记忆和反思机制。

### 3.2 搭建一个 Agent

1. 加载 python 依赖库

```python
import inspect
import os
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
```

2. 初始化客户端和模型

```python
loaded = load_dotenv(find_dotenv(), override=True)

API_KEY = os.getenv("SiliconFlow_API_KEY")
BASE_URL = os.getenv("SiliconFlow_BASE_URL")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
```

3. 定义工具函数

```python
def get_current_datetime() -> str:
    """
    获取当前日期和时间。
    :return: 当前日期和时间的字符串表示。
    """
    current_datetime = datetime.now()
    formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    return formatted_datetime


def add(a: float, b: float):
    """
    计算两个浮点数的和。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的和。
    """
    return str(a + b)


def mul(a: float, b: float):
    """
    计算两个浮点数的积。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 两个浮点数的积。
    """
    return str(a * b)


def compare(a: float, b: float):
    """
    比较两个浮点数的大小。
    :param a: 第一个浮点数。
    :param b: 第二个浮点数。
    :return: 比较结果的字符串表示。
    """
    if a > b:
        return f'{a} is greater than {b}'
    elif a < b:
        return f'{b} is greater than {a}'
    else:
        return f'{a} is equal to {b}'


def count_letter_in_string(a: str, b: str):
    """
    统计字符串中某个字母的出现次数。
    :param a: 要搜索的字符串。
    :param b: 要统计的字母。
    :return: 字母在字符串中出现的次数。
    """
    string = a.lower()
    letter = b.lower()

    count = string.count(letter)
    return f"The letter '{letter}' appears {count} times in the string."
```

4. 将工具类转换成特定的 JSON Schema 格式

```python
def function_to_json(func) -> dict:
    # 定义 Python 类型到 JSON 数据类型的映射
    type_map = {
        str: "string",  # 字符串类型映射为 JSON 的 "string"
        int: "integer",  # 整型类型映射为 JSON 的 "integer"
        float: "number",  # 浮点型映射为 JSON 的 "number"
        bool: "boolean",  # 布尔型映射为 JSON 的 "boolean"
        list: "array",  # 列表类型映射为 JSON 的 "array"
        dict: "object",  # 字典类型映射为 JSON 的 "object"
        type(None): "null",  # None 类型映射为 JSON 的 "null"
    }

    # 获取函数的签名信息
    try:
        signature = inspect.signature(func)
    except ValueError as e:
        # 如果获取签名失败，则抛出异常并显示具体的错误信息
        raise ValueError(
            f"无法获取函数 {func.__name__} 的签名: {str(e)}"
        )

    # 用于存储参数信息的字典
    parameters = {}
    for param in signature.parameters.values():
        # 尝试获取参数的类型，如果无法找到对应的类型则默认设置为 "string"
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            # 如果参数类型不在 type_map 中，抛出异常并显示具体错误信息
            raise KeyError(
                f"未知的类型注解 {param.annotation}，参数名为 {param.name}: {str(e)}"
            )
        # 将参数名及其类型信息添加到参数字典中
        parameters[param.name] = {"type": param_type}

    # 获取函数中所有必需的参数（即没有默认值的参数）
    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    # 返回包含函数描述信息的字典
    return {
        "type": "function",
        "function": {
            "name": func.__name__,  # 函数的名称
            "description": func.__doc__ or "",  # 函数的文档字符串（如果不存在则为空字符串）
            "parameters": {
                "type": "object",
                "properties": parameters,  # 函数参数的类型描述
                "required": required,  # 必须参数的列表
            },
        },
    }
```

5. 构造 Agent 类

```python
class Agent:
    def __init__(self, client: OpenAI, model: str = "Qwen/Qwen2.5-32B-Instruct", tools=None,
                 verbose: bool = True):
        if tools is None:
            tools = []
        self.client = client
        self.tools = tools
        self.model = model
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        self.verbose = verbose

    def get_tool_schema(self) -> List[Dict[str, Any]]:
        # 获取所有工具的 JSON 模式
        return [function_to_json(tool) for tool in self.tools]

    def handle_tool_call(self, tool_call):
        # 处理工具调用
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments
        function_id = tool_call.id

        function_call_content = eval(f"{function_name}(**{function_args})")

        return {
            "role": "tool",
            "content": function_call_content,
            "tool_call_id": function_id,
        }

    def get_completion(self, prompt) -> str:

        self.messages.append({"role": "user", "content": prompt})

        # 获取模型的完成响应
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=self.get_tool_schema(),
            stream=False,
        )
        if response.choices[0].message.tool_calls:
            self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
            # 处理工具调用
            tool_list = []
            for tool_call in response.choices[0].message.tool_calls:
                # 处理工具调用并将结果添加到消息列表中
                self.messages.append(self.handle_tool_call(tool_call))
                tool_list.append([tool_call.function.name, tool_call.function.arguments])
            if self.verbose:
                print("调用工具：", response.choices[0].message.content, tool_list)
            # 再次获取模型的完成响应，这次包含工具调用的结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.get_tool_schema(),
                stream=False,
            )

        # 将模型的完成响应添加到消息列表中
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        return response.choices[0].message.content
```

6. 启动 Agent，可以开心聊天了！

```python
SYSTEM_PROMPT = """
你是一个叫不要葱姜蒜的人工智能助手。你的输出应该与用户的语言保持一致。
当用户的问题需要调用工具时，你可以从提供的工具列表中调用适当的工具函数。
"""
```

```python
agent = Agent(
    client=client,
    model="Qwen/Qwen2.5-32B-Instruct",
    tools=[get_current_datetime, add, compare, count_letter_in_string],
)

while True:
    # 使用彩色输出区分用户输入和AI回答
    prompt = input("\033[94mUser: \033[0m")  # 蓝色显示用户输入提示
    if prompt == "exit":
        break
    response = agent.get_completion(prompt)
    print("\033[92mAssistant: \033[0m", response)  # 绿色显示AI助手回答
```

    User: 你好
    [92mAssistant: [0m 你好！有什么可以帮助你的吗？
    User: 9.12和9 .2哪个更大？
    调用工具：  [['compare', '{"a": 9.12, "b": 9.2}']]
    [92mAssistant: [0m 9.2 比 9.12 更大。
    User: 为什么？
    [92mAssistant: [0m 当我们比较两个数字的时候，我们会从左到右比较每一位的大小。对于 9.12 和 9.2，首先比较整数部分，它们都是 9，所以相等。然后比较小数部分，9.12 的小数部分是 12（可以认为是 1 和 2），而 9.2 的小数部分是 20（写成两位数时是 2 和 0）。因为 20 大于 12，所以 9.2 大于 9.12。
    
    实际上，9.2 可以写成 9.20，这更直观地显示出它的大小。因此，9.2 比 9.12 更大。
    User: strawberry中有几个r？
    调用工具：  [['count_letter_in_string', '{"a": "strawberry", "b": "r"}']]
    [92mAssistant: [0m 在单词 "strawberry" 中，字母 'r' 出现了 3 次。
    User: 现在是什么时候？
    调用工具：  [['get_current_datetime', '{}']]
    [92mAssistant: [0m 当前的时间是 2025 年 6 月 19 日 21 点 56 分 19 秒。请注意，这个时间是我的系统时间，可能与你的所在地时间有所不同。
    User: exit
    
