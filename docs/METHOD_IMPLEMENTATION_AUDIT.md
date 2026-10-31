> Historical implementation assessment. This document describes the initial feature-addition phase, not the current release-level rename and repair work. For the current repository see `REPOSITORY_AUDIT.md`.

# Diff-AMRSA 论文方法与代码对应审查

基准论文: `mrsa1031.zip/main.tex`，重点为 Materials and methods 第 2 节，以及 Fig. model。
基准仓库: `mega67(3).zip/mega67`。

## 结论

仓库原有实现包括 Megalodon 预训练三维扩散主干、MRSA 微调、采样、原始活性预测、理化性质分析和部分图表；但不能由这些脚本推断论文中的图对比学习、检索条件进入去噪器以及潜空间后验流匹配已存在。此次补充了独立代码并提供真实训练和推理入口。**没有修改任何原有 `.py`、`.yaml`、`.tex` 文件。**

| 论文位置 | 要求 | 对应新增实现 | 核查结论 |
|---|---|---|---|
| §2.1 | 按原始分组的 MRSA 数据划分、训练库隔离 | `dataset.py`, `export_splits.py` | 可读取原有 `train_h.pt/val_h.pt/test_h.pt`; 会拒绝泄漏。需在实际数据上运行 |
| §2.2 / Eq. graph representation | 分子图编码成归一化表示 `u` | `graph_encoder.py:GraphEncoder` | 已实现，128 维默认 |
| §2.2 / Eq. contrastive | 同非空骨架且同活性为正，异活性为负 | `graph_encoder.py:contrastive_loss`, `contrastive_batch_indices` | 已实现，可反传；包括同骨架硬负样本优先 |
| §2.3 / Eq. retrieval score | 4 项特征评分 `0.40/0.20/0.20/0.20` | `chemistry.py`, `retrieval.py` | 已实现，指纹/scaffold/SMARTS片段/二元活性；片段字典为显式工程选择 |
| §2.3 / Eq. retrieval context | Top-5, softmax `τ_r=0.20` | `retrieval.py:RetrievalLibrary` | 已实现；检索时排除同一标准化分子 |
| §2.3 / Eq. reverse diffusion | `p_θ(M_{s-1}|M_s,c_ret)` | `conditioning.py`, `conditional_model.py` | 实际注入每一步去噪的原子嵌入；不只是生成后重排。注入采用外部 hook，不改变原骨干源码 |
| §2.3 / Eq. diffusion output | 独立生成 `X,H,E,C`，查询不等于生成图 | 原有 `Graph3DInterpolantModel.sample` + 新 `sample_pipeline.py` | 调用骨干原始随机初始化和反向采样，随后构建生成图 |
| §2.4 / Eq. latent encoding | 图条件原子级坐标自编码器 + proper Kabsch | `flow.py:CoordinateAutoencoder` | 已实现两阶段训练和不允许镜像的 RMSD |
| §2.4 / Eq. flow matching | source perturb, linear latent path, conditional velocity MSE | `flow.py:LatentConformerRefiner`, `train_flow.py` | 已实现；`sigma_x=0.15` 为需调参的实现选择，论文没有给数值 |
| §2.4 / Eq. ODE | `N_f=5` 积分、只修改坐标 | `flow.py:refine`, `refine_rdkit_mol` | 已实现 5 步 midpoint ODE，输入生成图+生成构象；禁止改变原子/键 |
| §3.1 | AdamW 5e-5 / batch 16 / epochs 30 / diffusion 250 / raw candidates 10000 | 3 个训练脚本 + `sample_pipeline.py` | 默认参数对齐；能否达到结果有赖完整训练 |
| §3.1 | RF 分类器 + MIC 回归，仅 eligible exact active 参与 MIC | `train_activity_proxy.py` | 已实现训练/验证/测试集隔离；未复制原脚本中未经独立核查的数据划分 |
| §3.1 | chemical / geometry screening + Pareto top-100 | `screening.py`, `screen_and_rank.py` | 已实现基础筛选和三目标 Pareto；置信度门限为工程默认值，不是论文给定数值 |
| §3.3 | PoseBusters 全套 PB-valid 几何检查 | 新六项透明启发式检查 + UFF/finite checks | **已提供六项非等价启发式检查，但尚未实现与论文 PoseBusters 配置完全一致的度量**，不能据此声称复现 0.748 |
| §3.5 | 四级消融：baseline / retrieval / contrastive / flow | `build_library.py` (非对比随机对照) + 新条件训练/采样的 flag | 提供所需实验入口，但必须分别重新训练并运行，不能从论文表直接写入实验结果 |

## 必须如实说明的界限

1. 论文的 47,353 行初始清洗表已在仓库，但它不是 46,969 个三维划分记录，也不是 37,585 个训练样本。**不能将全表直接用于检索/活性预测训练**。
2. 上传的 zip 不包括论文所需的远程 `processed/{train,val,test}_h.pt`、GEOM 预训练权重、训练完成的 Diff-AMRSA 条件扩散、对比编码器和流匹配权重。不能在本地据此得到论文中 0.9060、0.748 等结果。
3. 源码的独立模块/小样本测试通过，不等同于 CUDA 训练或 PoseBusters 的全流程复现。模型结构/条件注入的具体层位置和 flow 潜变量宽度并未被论文唯一规定；此处实现为一种具体、可训练的匹配方案。
4. 对图的化学键身份只做只读保留，坐标端会检查有限性等基本有效性。三维构象变化可能影响立体化学解释，正式结果仍需严格 3D/立体化学复核。
5. 原有旧版检查脚本和绘图脚本保持原状，它们的历史输出不是新模块运行的凭据。若需要在论文里声称新模块训练/测试完成，应提供新脚本的 checkpoint、生成 SDF、manifest 与指标记录。

## 原文件改动记录

**旧文件改动数：0。** 全部变更均为新增文件：`src/diff_amrsa/`、`scripts/diff_amrsa/`、`tests/diff_amrsa/`、本报告和 `docs/EXPERIMENT_GUIDE.md`。原有 Megalodon checkpoint 的权重 key 和主干模型源代码不变；新增的 `ConditionedMRSA` 子类通过 forward hooks 条件注入并独立存储 adapter 参数。
