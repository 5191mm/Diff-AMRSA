# 使用 GitHub Desktop 更新 Diff-AMRSA 仓库

本压缩包是适合提交到 GitHub 的项目源码版本 不包含 `.git` 目录 也不包含历史 `outputs/` 大型实验结果

1. 在 GitHub Desktop 中登录对 `5191mm/Diff-AMRSA` 有写入权限的 GitHub 账号
2. 克隆 `https://github.com/5191mm/Diff-AMRSA` 到本地新文件夹
3. 先从 `main` 新建并发布 `backup-before-release` 备份分支 再切回 `main`
4. **如果 GitHub Desktop 显示存在尚未完成的 Merge 或冲突 必须先解决或中止合并** 不要直接提交此压缩包 原上传文件的 `.git` 处于未完成的合并状态 且有冲突文件
5. 在克隆出的文件夹中删除旧项目文件但**保留 `.git` 目录**
6. 解压本压缩包 将里面 `Diff-AMRSA/` 目录的所有内容直接复制到克隆仓库根目录 不要多放一层文件夹
7. 在 Desktop 的 Changes 中核对新增 修改 删除列表 尤其是 `src/diff_amrsa/` 和 `scripts/diff_amrsa/` 是否存在
8. 提交 `Prepare Diff-AMRSA research code release` 然后点击 `Push origin`
9. 刷新仓库主页 检查 `main` 的 README 和 `docs/REPOSITORY_AUDIT.md`

不能通过覆盖文件消除本地 `.git` 合并冲突 最稳妥的办法是重新克隆一个干净仓库 并在新克隆副本上覆盖代码

旧版实验结果仍保留在原始上传的 ZIP 中 本发布包未携带这些大型输出
