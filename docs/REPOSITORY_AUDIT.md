# Diff-AMRSA 仓库审查与正式版重构记录

本报告只针对用户上传的 `Diff-AMRSA.zip` 检查和修改 不代表完成论文全流程的 GPU 实验复现

## 一、已确认的问题与处理

| 级别 | 原仓库问题 | 本次处理 |
|---|---|---|
| 严重 | ZIP 内同时包含根目录 `.git/` 和误嵌套的 `Diff-AMRSA/.git/` | 正式发布 ZIP 不包含任何 `.git/` 必须复制到干净克隆仓库 |
| 严重 | 根目录 Git 存在 `MERGE_HEAD` 且 `LICENSE~...` 是未解决冲突索引项 | 不传播合并状态 删除残留冲突文件 说明不能通过覆盖代码修复原 `.git` |
| 严重 | `data_processing/build_conformers_core.py` 依赖缺失的 `utils_data.py` | 新增 `data_processing/utils_data.py` 提供 PyG 转换 样本归档 训练先验和参考几何统计 |
| 高 | `README.md` 与 `MODEL_CARD.md` 仍然把仓库描述成未改动的 NVIDIA Megalodon | 新 README 介绍 Diff-AMRSA 原模型卡保留为 `MODEL_CARD_UPSTREAM.md` 新建研究扩展模型卡 |
| 高 | 旧版 `scripts/sample_diff_amrsa.py` 只进行领域适配采样及排序 与论文完整检索+流匹配方法不等价 | 删除该近重复入口 完整生成统一使用 `scripts/diff_amrsa/sample_pipeline.py` |
| 高 | `scripts/finetune.py` 和其他权重微调脚本几乎完全重复 | 保留 `scripts/train_domain_adaptation.py` 作为微调基线入口 删除多余副本 |
| 中 | 多组脚本名称包含 `process_mrsa_geom...` 和临时版本名 | 保留对应正式功能入口 清理重复文件 更新其余引用 |
| 中 | `train_activity_proxy.py` 与 `screen_and_rank.py` 通过脚本间临时导入共享特征提取 | 将特征提取统一移入 `src/diff_amrsa/activity_features.py` 两个阶段共享一致实现 |
| 中 | 多个 Bash 脚本及配置含作者旧机器绝对路径 | 主要入口改为项目相对路径 环境变量或明确占位符 |
| 中 | 依赖项未明确列出 scikit-learn joblib tqdm 等 | 更新 pyproject 与 requirements 添加所需依赖 |
| 中 | 旧仓库包含 `draw-old/` 旧 ZIP 文件脚本 `.bak` 文件以及大量历史运行结果 | GitHub 发布包不包含冗余归档和 `outputs/` 历史结果 原用户 ZIP 仍可保留作本地备份 |
| 中 | `pyproject.toml` 项目名称依旧是 `megalodon` `VERSION` 为 0.0.1 且 README 是上游项目 | 发布包的安装项目名改为 `diff-amrsa` 版本调整为 0.1.0 仍保留 `megalodon` Python 包来兼容预训练网络 |
| 中 | 生成计数将禁用流匹配的输出误计入 `refine_success` | 细分 `valid_initial` `attempted_refine` `refine_success` `written_candidates` |
| 中 | 训练/验证/测试的活性代理仅检查部分划分重叠且空数据会抛出不直观异常 | 增加三组分子身份重叠和空数据检查 缺失 MIC 训练目标时报清晰错误 |

## 二、主要改名清单

| 旧位置 | 正式版位置 / 处理 | 原因 |
|---|---|---|
| `process_mrsa_geom_scheme_a_fixedset_timeout_resume_mp.py` | 使用 `data_processing/build_conformers.py` | 该脚本与现有入口几乎完全重复 已清理旧副本 |
| `process_mrsa_geom_scheme_a_fixedset_mp.py` | 使用 `data_processing/build_conformers_core.py` | 旧脚本与现有核心几乎完全重复 |
| `process_mrsa_geom_scheme_a_adaptive_weight_processed.py` | `data_processing/assign_sample_weights.py` | 独立功能 采用职责命名 |
| `process_drug_concentration_tables.py` | `data_processing/merge_activity_tables.py` | 以数据处理任务命名 |
| `data_processing/prepare_mrsa_scheme_a_table.py` | `data_processing/prepare_dataset.py` | 完全重复 删除旧副本 |
| `scripts/train_mrsa_activity_predictor_scheme_a_v2.py` | `scripts/train_activity_predictor.py` | 完全重复 删除版本后缀副本 |
| `scripts/learn_mrsa_bin_weights_from_generated.py` | `scripts/learn_bin_weights.py` | 完全重复 删除旧副本 |
| `scripts/train_mrsa_finetune_weights_only_loss_log.py` | `scripts/train_domain_adaptation.py` | 明确微调基线的职责 |
| `scripts/train_mrsa_finetune_no_optimizer_resume.py` | `scripts/train_domain_adaptation_fresh.py` | 保留不恢复优化器状态的特殊入口 |
| `scripts/sample_mrsa_guided_scheme_a.py` | `scripts/sample_domain_adapted.py` | 明确并非完整 Diff-AMRSA 生成器 |
| `scripts/sample_mrsa_guided_scheme_a_robust.py` | `scripts/sample_domain_adapted_robust.py` | 保留更稳健的历史基线采样 |
| `scripts/sample_diff_amrsa.py` | 删除近重复副本 | 与上方基线采样几乎相同 容易造成误解 |
| `scripts/finetune.py` | 删除近重复副本 | 与领域微调脚本几乎相同 |
| `scripts/run_diff_amrsa_pipeline.sh` | `scripts/run_legacy_pipeline.sh` | 旧流程并未训练图对比 检索条件与流精修 |
| *新增* | `scripts/run_pipeline.sh` | 面向完整三模块的统一分阶段入口 |
| `draw/` | `evaluation/figures/` | 图表评估目录正式化 |
| `table/` | `evaluation/tables/` | 表格评估目录正式化 |
| `DIFF_AMRSA_QUICKSTART.md` | `docs/EXPERIMENT_GUIDE.md` | 统一文档位置 |
| `DIFF_AMRSA_IMPLEMENTATION_AUDIT.md` | `docs/METHOD_IMPLEMENTATION_AUDIT.md` | 旧审核移入历史文档 |

> `GEOM-Drugs` 是上游三维数据集的**正式名称** `geometry.py` 是几何度量实现 故这些专业术语保留 只删除临时与歧义的脚本命名 旧输出分组 `original_geom` 统一改称 `pretrained_baseline` 并需要在重新运行实验时使用新的结果目录

## 三、测试范围与结果

- `PYTHONPATH=src python -m pytest -q tests/diff_amrsa`：**12 passed** (PyTorch CPU + RDKit 环境)
- `python -m compileall -q src scripts data_processing evaluation tests`：通过
- `bash -n scripts/run_pipeline.sh scripts/run_legacy_pipeline.sh ...`：通过
- `bash scripts/run_pipeline.sh help`：通过
- `python data_processing/build_conformers.py --help`：通过
- 未能在当前环境做完整 PyG 处理 检查点加载 GPU 条件扩散训练或 Flow 模块大规模生成 因缺少 `torch_geometric` 完整训练数据与可靠 checkpoint

### 新增统计工具的口径

`data_processing/utils_data.py` 针对原本缺失的辅助文件提供可检查的实现 其 PyG 图字段与 Megalodon 源代码使用的输入字段一致 类别频率在全输入构象上计算 几何距离与角度参考统计为确定性抽取最多 5000 个构象的工程默认设置 **该统计不等同于原论文或上游仓库完整数据集提供的所有参考分布** 对照论文结果之前需要在目标环境执行并检查

## 四、尚需在服务器验证

1. 运行完整的 PyG `train_h.pt / val_h.pt / test_h.pt` 保存与再次加载测试 并检查类别统计先验与原 checkpoint 一致
2. 使用匹配版本的 PyTorch CUDA PyG 执行图编码器 条件扩散 微调 流匹配训练和完整的 SDF 推理
3. 单独运行原论文所用官方 PoseBusters 完整筛选和评估 本仓库 `posebusters_inspired` 不能替代 PB-valid
4. 重新计算实验指标 旧 `outputs/` 文件来自历史脚本 不可自动当作新模块训练结果
5. 检查候选分子 3D 构象和活性标签的科学合理性以及原始数据发布授权

## 五、GitHub 发布边界

源码发布包为了 GitHub 清爽维护 排除 `.git/` 目录 历史 `outputs/` 大型结果 `.bak` 文件 `draw-old/` 和嵌套源码归档 这些输出可留在本地用户原 ZIP 中 因为已存在历史数据的删除**并非数据遗失**。将发布包解压后内部 `Diff-AMRSA/` 所有文件复制到新克隆仓库根目录 上传后检查 `main` 分支。

**重要** 当前原上传本地 `.git` 处于未结束的 Git 合并状态 本发布包不修改该隐藏目录 不能通过复制代码解决它 建议用 GitHub Desktop 重新克隆干净副本后再覆盖源码。
