# GPRF Query Expansion 智能检索查询扩展

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

基于生成式模型和伪相关反馈的查询扩展方法复现。

## ✨ 项目特色

- 🧠 **双路扩展机制**：结合BART生成模型和RM3伪相关反馈
- 🔍 **多检索器融合**：DPR密集检索 + RM3传统检索
- 📊 **完整评估**：Top-k准确率和EM准确率评估
- 🛠️ **模块化设计**：易于扩展和维护

## 🚀 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA (可选，用于GPU加速)

### 安装步骤

1. **克隆项目**

   ```bash
   git clone https://github.com/1226248523/GPRF-Query-Expansion-Project.git
   cd GPRF-Query-Expansion-Project
   ```
2. **创建虚拟环境**

   ```bash
   python -m venv venv

   # Windows:
   venv\Scripts\activate

   # Linux/Mac:
   source venv/bin/activate
   ```
3. **安装依赖**

   ```bash
   pip install -r requirements.txt
   pip install -e .  # 安装项目为可编辑包
   ```

### 基本使用

```python
from gprf import BartQueryGenerator, DPRRetriever, PRFExpander

# 1. 加载配置
from gprf.utils.config import load_config
config = load_config("configs/default.yaml")

# 2. 初始化组件
generator = BartQueryGenerator(config)
retriever = DPRRetriever(config)
expander = PRFExpander(config["paths"]["index_path"])

# 3. 处理查询
example = {
    "Question": "What is artificial intelligence?",
    "Answer": "AI is technology that mimics human intelligence",
    "Title": "AI Overview",
    "Sentence": "Artificial intelligence refers to computer systems..."
}

# 4. 生成扩展词
expansions = generator.generate_expansion_batch([example])
prf_terms = expander.get_prf_terms(example["Question"])

print("生成扩展词:", expansions[0])
print("PRF扩展词:", prf_terms)
```

## 📁 项目结构

```
gprf-query-expansion/
├── src/gprf/                 # 核心库
│   ├── __init__.py
│   ├── core/                # 生成器/检索器/扩展器
│   │   ├── __init__.py
│   │   ├── generators.py
│   │   ├── retrievers.py
│   │   └── expanders.py
│   └── utils/               # 配置与评估工具
│       ├── config.py
│       └── evaluation.py
├── tests/                   # 单元与集成测试
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   └── test_generators.py
│   ├── integration/
│   └── fixtures/
├── scripts/                 # 命令行脚本
│   └── run_evaluation.py
├── examples/                # 使用示例
│   └── basic_usage.py
├── configs/                 # YAML 配置
│   └── default.yaml
├── docs/                    # 文档（含原论文 PDF）
│   └── 智能检索中基于生成式模型和伪相关反馈的查询扩展方法_秦春秀.pdf
├── main.py                  # 旧版入口（保留参考）
├── requirements*.txt        # 依赖列表
├── pyproject.toml / setup.py
└── README.md 等项目元数据
```

## 🧪 运行测试

1. **安装开发依赖**
   ```bash
   pip install -r requirements-dev.txt
   ```
2. **运行全部测试**
   ```bash
   pytest
   ```
3. **按模块运行**
   ```bash
   pytest tests/unit/test_generators.py      # 仅单元测试
   pytest -k "generators"                    # 关键字过滤
   ```
4. **生成覆盖率报告**
   ```bash
   pytest --cov=gprf --cov-report=term-missing --cov-report=html
   ```

   结果将输出到 `htmlcov/` 目录，可在浏览器中查看。

## 📖 文档

- `docs/智能检索中基于生成式模型和伪相关反馈的查询扩展方法_秦春秀.pdf`：项目参考论文，描述了GPRF的方法背景与细节
- `README.md`：快速开始与项目说明
- `examples/basic_usage.py`：API 使用示例
- `CONTRIBUTING.md`：贡献流程与开发规范
- （计划中）`docs/api.md`、`docs/architecture.md` 等可扩展文档

## 🤝 贡献指南

欢迎贡献代码！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详情。

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- 基于论文《智能检索中基于生成式模型和伪相关反馈的查询扩展方法》思想设计核心代码
- 基于 Cursor 完成代码开发
- 使用了 Facebook DPR、BART 等 Huggingface 社区的优秀开源项目
