# Contributing

感谢参与 Storyboard Flow。提交改动前，请先阅读 [DEVELOPMENT.md](DEVELOPMENT.md) 中的数据迁移、Provider 和测试约束。

## 开发流程

1. 从最新主分支创建主题分支。
2. 保持改动聚焦，不混入无关格式化或重构。
3. 为行为变化添加或更新测试。
4. 运行本地验证：

```bash
.venv/bin/ruff check app streamlit_app.py tests
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q app streamlit_app.py tests
.venv/bin/python -m pip check
```

5. 提交 Pull Request，说明问题、实现方式、验证结果和兼容性影响。

## 代码要求

- 支持 Python 3.12。
- 沿用现有类型标注和模块边界。
- SQLite 迁移必须向后兼容且可重复执行。
- 测试数据必须写入临时目录，不能修改仓库中的 `data/`。
- 新 Provider 必须实现 `is_available()`，并覆盖不可用和错误路径。
- UI 变更需要用 Mock Provider 完成相关交互验证。
- 不提交模型权重、生成资产、数据库、`.env`、密钥或用户数据。

## Issue 与 Pull Request

Bug 报告请提供：

- 操作系统、Python 和 Streamlit 版本
- 使用的 Provider 与相关非敏感配置
- 最小复现步骤
- 期望行为和实际行为
- 已脱敏的错误日志

功能请求请说明使用场景、预期行为和可能影响的数据或 Provider 契约。

Pull Request 应保持可审查。涉及数据库、Rubric、导出格式或 Provider 接口的改动，需要明确说明向后兼容策略。

## 安全问题

不要在公开 Issue 中报告未修复的漏洞或附带真实密钥、私有素材和数据库。请遵循 [SECURITY.md](SECURITY.md)。

## License

提交贡献即表示你有权提供该内容，并同意其按本仓库的 [MIT License](LICENSE) 分发。
