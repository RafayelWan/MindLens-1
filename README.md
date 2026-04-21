# Mind Lens

反讨好的 AI 思维分析工具——帮你理清纠结，而不是顺着你说。

通过 Prompt 工程 + CoT 思维链 + RAG 知识库三层约束机制，基于阿德勒心理学框架，为用户提供客观、结构化、有深度的问题分析。

## 项目结构

```
├── app/                     # 后端核心模块
│   ├── __init__.py
│   ├── config.py            # 配置加载（.env + prompt 文件）
│   ├── llm.py               # LLM 调用层（构建消息、API 调用、记忆管理）
│   ├── analysis.py          # 多轮分析业务逻辑（追问/轮次控制/JSON解析）
│   ├── memory.py            # 对话记忆管理（JSON 读写、自动裁剪）
│   └── server.py            # FastAPI 路由层（HTTP API + 静态文件托管）
├── prompts/                 # System Prompt
│   └── mind_lens.txt        # Mind Lens 分析 prompt（含心理学框架+追问逻辑）
├── rag/                     # RAG 知识库素材
│   ├── framework.md         # 阿德勒心理学分析框架
│   ├── analysis-template.md # 六步分析模板
│   └── case-examples.md     # 典型分析案例
├── ui/                      # 前端页面
│   ├── index.html           # 输入页
│   ├── result.html          # 分析结果页
│   └── styles.css           # 样式
├── chat.py                  # 终端聊天入口
├── run_server.py            # 网页服务启动入口
├── test_analysis.py         # 多轮分析终端测试脚本
├── requirements.txt         # Python 依赖
└── .env.example             # 环境变量模板
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env` 文件，填入你的 API Key：

```
LLM_API_KEY=你的API Key
LLM_BASE_URL=你的API地址
LLM_MODEL=你的模型名称
```

### 3. 使用方式

#### 网页版（Mind Lens）

```bash
python run_server.py
```

浏览器访问 `http://localhost:8000`，输入困扰你的问题，AI 会通过多轮追问深入了解后给出五维分析。

API 文档：`http://localhost:8000/docs`

#### 终端测试多轮分析

```bash
python test_analysis.py
```

无需启动服务器，直接在终端体验多轮追问 + 结构化分析的完整流程。

#### 终端聊天

```bash
python chat.py
```

命令行对话模式，支持流式输出和角色切换。

## 架构说明

```
app/analysis.py（业务逻辑层）
  ├── app/server.py    调用 → FastAPI HTTP 路由
  └── test_analysis.py 调用 → 终端测试
  
app/llm.py（LLM 调用层）
  ├── app/analysis.py  调用 → 多轮分析
  └── chat.py          调用 → 终端聊天
```

核心业务逻辑集中在 `app/analysis.py`，修改分析流程只需改这一个文件。

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/analysis/start` | POST | 开始新分析（清空记忆 + 首轮分析） |
| `/api/analysis/reply` | POST | 继续多轮对话（追问回复） |
| `/api/chat` | POST | 通用聊天（支持流式/非流式） |
| `/api/clear` | POST | 清空对话记忆 |
| `/api/memory` | GET | 查看当前对话记忆 |

分析接口返回结构化 JSON：

```json
{
  "cards": {
    "surface": "表层问题分析",
    "assumption": "隐含假设",
    "info_gap": "信息缺失",
    "blind_spot": "认知盲点",
    "suggestion": "建议（条件满足时才有值）"
  },
  "follow_up": "追问问题（或null）",
  "ready_for_suggestion": false,
  "round": 1
}
```

## 注意事项

- `.env` 文件包含 API Key，**请勿提交到 Git**（已在 .gitignore 中排除）
- `memory.json` 是运行时生成的对话记录文件
- 每个人需要使用自己的 API Key
