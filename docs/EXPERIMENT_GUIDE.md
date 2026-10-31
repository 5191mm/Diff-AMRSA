> Commands below document the full research workflow. They require a prepared PyG split and matching checkpoints. For a current publication overview and setup see the repository `README.md`.

# Diff-AMRSA 可运行流程（新增模块）

此流程基于 `mrsa1031.zip/main.tex` 和 `figure/model.png`。执行以下命令前须完成仓库依赖安装，并准备可信的预训练权重和原始 `processed` 数据。所有命令从仓库根目录执行。

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python -m pip install -e .
# Install CUDA-compatible torch and PyG wheels using the setup guidance in README.md
```

以下占位符由实际实验路径替换。`$DATA` 目录至少要包含 `processed/train_h.pt`, `val_h.pt`, `test_h.pt` 及 Megalodon 所需的 train prior/statistics 文件。`$BASE_CKPT` 应为本地可信的 MRSA 适配 Megalodon checkpoint。`$CONFIG` 应与该 checkpoint 完全匹配。

```bash
export DATA=/path/to/custom_mrsa_schemeA_fixed_30_10
export CONFIG=scripts/conf/drugs/mrsa_diffusion_finetune_scheme_a.yaml
export BASE_CKPT=/path/to/trusted_mrsa_finetuned.ckpt
export ACTIVITY_TABLE=mrsa_scheme_a/processed_drug_mic_dataset_schemeA.csv
mkdir -p run_diff_amrsa
```

**注意**：当前配置里的 `data.dataset_root` 写的是旧机器绝对路径。请复制一份新配置（例如 `run_diff_amrsa/local.yaml`）并只修改其中 `data.dataset_root: /实际数据目录`，再将 `CONFIG` 指向新配置。原始文件不用动。

## 1. 提取准确的三组分子身份（必做）

```bash
python scripts/diff_amrsa/export_splits.py \
  --dataset-root "$DATA" \
  --activity-table "$ACTIVITY_TABLE" \
  --output-dir run_diff_amrsa/splits
```

不重新随机分组；直接读取现有 PyG split。表中与 SMILES 身份不一致/冲突的记录会被跳过；相同标准化身份若横跨 train/val/test 则明确报错。这样避免引入测试数据泄漏。

## 2. 先训练图编码器，再构建检索库

```bash
python scripts/diff_amrsa/train_graph_encoder.py \
  --split-dir run_diff_amrsa/splits \
  --output-dir run_diff_amrsa/graph \
  --epochs 30 --batch-size 32 --lr 5e-5 --tau-c 0.10
```

产物：`graph_encoder.pt`、`train_retrieval.npz`。检索为 Morgan Tanimoto + Murcko scaffold + SMARTS 药效团片段 + 标签，权重 0.40/0.20/0.20/0.20，K=5，`tau_r=0.20`。

## 3. 条件扩散微调（从原 MRSA checkpoint 加载主干）

```bash
python scripts/diff_amrsa/train_conditioned_diffusion.py \
  --config "$CONFIG" --backbone-ckpt "$BASE_CKPT" \
  --library run_diff_amrsa/graph/train_retrieval.npz \
  --split-dir run_diff_amrsa/splits \
  --outdir run_diff_amrsa/conditioned \
  --epochs 30 --batch-size 16 --lr 5e-5
```

训练器会验证检索库内容只来自 train。去噪条件投影通过 forward hooks 接入原子隐藏层；主干文件无改动。将最优 `conditioned-*.ckpt` 作为下游 `COND_CKPT`。需要仅训练新投影层时可加 `--freeze-backbone`，但这与论文中的联合适配不是同一个实验设置。

## 4. 从原 PyG 分割导出真实构象，再训练流匹配

```bash
python scripts/diff_amrsa/export_conformers.py \
  --dataset-root "$DATA" \
  --split-dir run_diff_amrsa/splits \
  --outdir run_diff_amrsa/conformers \
  --max-per-molecule 3

python scripts/diff_amrsa/train_flow.py \
  --train-sdf run_diff_amrsa/conformers/train.sdf \
  --val-sdf run_diff_amrsa/conformers/val.sdf \
  --split-dir run_diff_amrsa/splits \
  --library run_diff_amrsa/graph/train_retrieval.npz \
  --graph-checkpoint run_diff_amrsa/graph/graph_encoder.pt \
  --outdir run_diff_amrsa/flow \
  --autoencoder-epochs 20 --flow-epochs 30 --lr 5e-5
```

其中首先根据 validation 选择最佳坐标编码器/解码器并冻结，再训练速度场。训练样本是同一分子固定原子索引下的高斯扰动源坐标与目标坐标，使用 proper Kabsch（不允许镜像）。通过预测的图表无法重建真实 SDF。默认 `sigma_x=0.15 Å`、latent=32、autoencoder epochs=20 是**具体工程选择**，论文未明确这些值；可在方法一致性实验中调整。

## 5. 从真正的扩散采样得到独立候选，并实施坐标精修

```bash
export COND_CKPT=/path/to/best_conditioned.ckpt
export QUERY=$(python -c "import json; print(json.loads(open('run_diff_amrsa/splits/train.jsonl').readline())['smiles'])")

python scripts/diff_amrsa/sample_pipeline.py \
  --config "$CONFIG" \
  --conditioned-checkpoint "$COND_CKPT" \
  --library run_diff_amrsa/graph/train_retrieval.npz \
  --split-dir run_diff_amrsa/splits \
  --graph-checkpoint run_diff_amrsa/graph/graph_encoder.pt \
  --flow-checkpoint run_diff_amrsa/flow/best_flow.pt \
  --query-smiles "$QUERY" \
  --outdir run_diff_amrsa/generated \
  --n-graphs 10000 --batch-size 100 --timesteps 250
```

`QUERY` 只控制检索：**不等于扩散生成的分子图**。此命令输出 `initial.sdf`、`refined.sdf`、`manifest.json`。若去噪输出化学结构无效，文件会记录失败项，而不会虚报成功。需要关闭坐标精修（测试最后一步消融）可添加 `--disable-flow`，省略 `--flow-checkpoint`。

## 6. 训练 MRSA 代理模型，完成 Pareto 排序

```bash
python scripts/diff_amrsa/train_activity_proxy.py \
  --split-dir run_diff_amrsa/splits \
  --activity-table "$ACTIVITY_TABLE" \
  --outdir run_diff_amrsa/activity --n-estimators 200

python scripts/diff_amrsa/screen_and_rank.py \
  --sdf run_diff_amrsa/generated/refined.sdf \
  --library run_diff_amrsa/graph/train_retrieval.npz \
  --activity-proxy run_diff_amrsa/activity/activity_proxy.joblib \
  --outdir run_diff_amrsa/ranked --top-k 100
```

本实现训练分类器及仅 active、exact、<100 µg/mL 的 log10 MIC 回归器。候选综合使用 QED、预测活性概率、领域相关性进行 Pareto 排序。默认 QED ≥0.3、分类置信度 ≥0.1 等筛选阈值是**工程默认值**，可变更，不应直接当作论文给定参数。

## 7. 消融实验的可复现入口

- **MRSA-adapted diffusion baseline**：运行原 `scripts/sample.py`，使用相同原始 MRSA checkpoint 和 `--timesteps 250 --n_graphs 10000`。
- **+ Retrieval**：`build_library.py` 不带 `--graph-checkpoint` 得到固定随机编码器的控制条件库，用该库**另行训练**一份条件扩散模型；这只是可运行的弱对照（不等同于论文可能采用的无对比预训练编码器）。
- **+ Graph contrastive**：步骤 2 + 3，生成时加 `--disable-flow`。
- **+ Flow**：步骤 2～5 的完整方法。
- 消融需要一致的输入划分、生成样本数、评估指标和独立训练结果。`--zero-context` 仅用于测试条件是否起作用，**不是**已经重训的正式消融模型。

## 8. 本地自测与验收

```bash
PYTHONPATH=src pytest -q tests/diff_amrsa
python -m compileall -q src/diff_amrsa scripts/diff_amrsa
```

单元测试验证了检索数学计算、标签掩码、上下文注入、训练/测试集隔离、坐标对齐和图身份保持，但不能替代 GPU 训练或生成 10,000 个分子的实验。附带的旧版 `evaluation/figures/` 与 `evaluation/tables/` 脚本原样保留。新增的六项 `pb_inspired_*` 检查只是公开阈值的启发式版本，不等价于 PoseBusters。正式论文需要重新生成数据并通过统一的官方 PoseBusters / UFF / RMSD 测评流程验证图表数字，不能仅凭本补充代码声称已有论文结果被复现。

## 9. 环境边界

- 依赖仍以仓库原 `pyproject.toml/requirements.txt` 为准。新的训练和推理入口需要 PyTorch-Geometric、`torch_scatter` 等原仓库依赖，并需与 CUDA 匹配。
- 当前上传的 zip 未包含远程的数据 `processed/` 文件、预训练/微调 `.ckpt` 和本方法新阶段训练权重。
- 从不在不可信 checkpoint/pickle/joblib 文件上运行 `torch.load(weights_only=False)` / `joblib.load`。这些文件应来自用户本人训练与可信实验环境。
- 新增模型实际注入位置和片段 SMARTS 定义必须作为补充材料记录，不能说论文精确规定了这些未公开实现细节。
