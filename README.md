# MindLens

LLM 聊天工具 + Mind Lens 问题分析网页，支持终端和网页两种交互方式。

## 项目结构

```
├── llm.py               # LLM 核心层（构建消息、调用 API、管理记忆）
├── chat.py              # 终端聊天入口（流式输出、角色切换）
├── server.py            # 网页服务入口（FastAPI，托管前端 + 提供 API）
├── config.py            # 配置加载（读取 .env + prompt 文件）
├── memory.py            # 对话记忆管理（JSON 读写、自动裁剪）
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板（复制后填入你的 Key）
├── prompts/             # System Prompt 角色文件
│   ├── default.txt      # 默认：友好中文助手
│   ├── coder.txt        # 编程助手
│   ├── translator.txt   # 翻译助手
│   └── mind_lens.txt    # Mind Lens 问题分析
└── ui/                  # Mind Lens 网页前端
    ├── index.html       # 输入页
    ├── result.html      # AI 分析结果页
    └── styles.css       # 样式
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制模板文件并填入你的信息：

```bash
cp .env.example .env
```

用文本编辑器打开 `.env`，填入你的 API Key：

```
LLM_API_KEY=你的API Key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4.7
```

也可以换别家的模型，URL 也需要相应更换。

> 智谱 API Key 获取地址：https://open.bigmodel.cn/usercenter/apikeys

### 3. 使用方式

#### 终端聊天

```bash
python chat.py
```

在命令行中直接与 LLM 对话，支持流式输出和角色切换。

#### 网页版（Mind Lens）

```bash
python server.py
```

启动后浏览器访问 `http://localhost:8000`，使用 Mind Lens 问题分析功能。

访问 `http://localhost:8000/docs` 可查看自动生成的 API 文档。

### 终端聊天命令

| 命令 | 功能 |
|------|------|
| `help` | 查看所有命令 |
| `prompt` | 查看当前角色和可用列表 |
| `prompt coder` | 切换到编程助手角色 |
| `prompt translator` | 切换到翻译助手角色 |
| `clear` | 清空对话记忆 |
| `quit` / `exit` | 退出 |

### 自定义角色

在 `prompts/` 文件夹中创建 `.txt` 文件即可添加新角色，文件名就是角色名。

## 架构说明

```
llm.py（核心层）
  ├── chat.py   调用 → 终端交互
  └── server.py 调用 → HTTP 路由 + 网页托管
```

`llm.py` 是唯一的业务逻辑层，`chat.py` 和 `server.py` 只负责各自的交互方式。修改 LLM 调用逻辑只需改 `llm.py`，两个入口自动生效。

## 注意事项

- `.env` 文件包含 API Key，**请勿提交到 Git**（已在 .gitignore 中排除）
- `memory.json` 是运行时自动生成的对话记录文件
- 每个人需要使用自己的 API Key
