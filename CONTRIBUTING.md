# Deep-AI-Notes 贡献指南

感谢你对 Deep-AI-Notes 项目的关注！我们欢迎所有形式的贡献。

## 🤝 如何贡献

### 报告 Bug

如果你发现了 Bug，请：

1. 检查 [Issues](https://github.com/FuTseYi/Deep-AI-Notes/issues) 中是否已有相关报告
2. 如果没有，创建新的 Issue，并提供：
   - Bug 的详细描述
   - 复现步骤
   - 预期行为和实际行为
   - 截图（如果适用）
   - 环境信息（操作系统、Node.js 版本等）

### 提出新功能

如果你有新功能建议：

1. 先在 [Issues](https://github.com/FuTseYi/Deep-AI-Notes/issues) 中讨论
2. 说明功能的用途和价值
3. 如果可能，提供实现思路

### 提交代码

1. **Fork 本仓库**

2. **克隆到本地**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Deep-AI-Notes.git
   cd Deep-AI-Notes
   ```

3. **创建特性分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **安装依赖**
   ```bash
   pnpm install
   ```

5. **进行修改并测试**
   ```bash
   pnpm dev  # 启动开发服务器
   pnpm build  # 测试构建
   ```

6. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

7. **推送到 GitHub**
   ```bash
   git push origin feature/your-feature-name
   ```

8. **创建 Pull Request**
   - 在 GitHub 上创建 PR
   - 填写 PR 模板
   - 等待审核

## 📝 代码规范

### Commit 信息规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `style:` 代码格式调整（不影响功能）
- `refactor:` 代码重构
- `perf:` 性能优化
- `test:` 测试相关
- `chore:` 构建/工具链相关

示例：
```
feat: add reading progress component
fix: resolve mermaid rendering issue in dark mode
docs: update deployment guide
```

### 代码风格

- **TypeScript**：用于配置文件
- **Vue 3**：使用 Composition API
- **CSS**：使用 BEM 命名规范
- **缩进**：2 空格
- **引号**：单引号

### 文件命名

- 组件：PascalCase（如 `ReadingProgress.vue`）
- 配置文件：kebab-case（如 `config.ts`）
- 样式文件：kebab-case（如 `custom.css`）

## 🧪 测试

在提交 PR 前，请确保：

- [ ] 本地开发服务器正常运行
- [ ] 生产构建成功
- [ ] 所有页面正常显示
- [ ] 响应式布局正常
- [ ] 深色模式正常
- [ ] 没有控制台错误

## 📖 文档

如果你的更改涉及用户可见的功能：

- 更新 README.md
- 添加使用示例
- 更新相关配置说明

## 💬 交流

- **Issues**：[GitHub Issues](https://github.com/FuTseYi/Deep-AI-Notes/issues)
- **Email**：tseyi.wk@icloud.com

## 📄 许可证

提交代码即表示你同意将代码以 MIT 许可证开源。

---

再次感谢你的贡献！🎉

