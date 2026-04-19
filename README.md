# DAWN Demo

一个 LLM 聊天工具 + Mind Lens 问题分析网页。

## 项目结构

```
├── chat.py              # 终端聊天脚本（支持流式输出、记忆、角色切换）
├── config.py            # 配置加载模块
├── memory.py            # 对话记忆管理（JSON 读写）
├── requirements.txt     # Python 依赖
├── .env.example         # 环境变量模板（复制后填入你的 Key）
├── prompts/             # System Prompt 角色文件
│   ├── default.txt      # 默认：友好中文助手
│   ├── coder.txt        # 编程助手
│   └── translator.txt   # 翻译助手
└── ui/                  # Mind Lens 网页前端
    ├── index.html
    ├── result.html
    └── styles.css
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

然后用文本编辑器打开 `.env`，填入你的 API Key：

```
LLM_API_KEY=你的智谱API Key
LLM_BASE_URL=https://open.bigmodel.cn/api/paas/v4
LLM_MODEL=glm-4.7
```

> API Key 获取地址：https://open.bigmodel.cn/usercenter/apikeys

### 3. 运行聊天

```bash
python chat.py
```

### 聊天命令

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

## 注意事项

- `.env` 文件包含 API Key，**请勿提交到 Git**（已在 .gitignore 中排除）
- `memory.json` 是运行时自动生成的对话记录文件
- 每个人需要使用自己的 API Key
