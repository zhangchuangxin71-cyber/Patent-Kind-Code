# GitHub 自动同步

本项目已将远程仓库配置为：`https://github.com/zhangchuangxin71-cyber/Patent-Kind-Code`。

每次完成一项较大的修改后，执行：

```bash
git add <修改的文件>
git commit -m "简要说明本次修改"
```

提交完成后，`.githooks/post-commit` 会自动推送当前分支到 GitHub。若网络或 GitHub 登录暂时不可用，提交仍会保留在本地；恢复后执行 `git push` 即可补传。若某次需要只提交到本地，可在命令前加 `SKIP_GITHUB_AUTO_PUSH=1`。

原始数据、缓存、模型权重、密钥和独立的 `ClipTalk` 仓库均已排除，以避免泄露凭据及超过 GitHub 文件大小限制。实验的汇总结果、代码、文档和图表会同步。
