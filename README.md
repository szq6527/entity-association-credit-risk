# 企业信用风险传导预测（工作文档）

## 数据介绍

### 1. 当前已落地数据（仓库内已有）

| 数据主题 | 文件路径 | 关键字段（示例） | 用途 |
|---|---|---|---|
| 上市公司基本情况 | `上市公司基本情况文件180625851(仅供北京大学使用)/HLD_Copro.xlsx` | `Stkcd`, `Reptdt`, `Nnindcd`, `Regcap` | 公司静态属性、行业信息、样本对齐主键 |
| 对外担保事件 | `上市公司对外担保情况表175710486(仅供北京大学使用)/STK_Guarantee_Main.xlsx` | `Symbol`, `Guarantee`, `StartDate`, `EndDate`, `ActualGuaranteeAmount`, `GuaranteeTypeID` | 构建担保关系边（核心关系） |
| 控制人信息 | `上市公司控制人文件200916280(仅供北京大学使用)/HLD_Contrshr.xlsx` | `Stkcd`, `Reptdt`, `S0701b`, `S0704b` | 控制权与治理相关特征 |
| 股权关系链 | `股权关系文件200853795(仅供北京大学使用)/HLD_Shrrelchain.xlsx` | `Stkcd`, `S0601a`, `S0603a`, `S0604a`, `S0606a` | 构建投资/关联关系边 |
| 股权变更事件 | `股权变更情况文件200923078(仅供北京大学使用)/HLD_Chgequity.xlsx` | `Stkcd`, `S0801a`, `S0802a`, `S0805a`, `S0808a` | 关系动态变化、事件特征 |
| 资产负债表 | `财务情况/FS_Combas.xlsx` | `Stkcd`, `Accper`, 各财务科目编码 | 偿债能力、资本结构特征 |
| 利润表 | `利润表201210555(仅供北京大学使用)/FS_Comins.xlsx` | `Stkcd`, `Accper`, 各损益科目编码 | 盈利能力、经营质量特征 |
| 现金流量表 | `现金流量表(直接法)201221487(仅供北京大学使用)/FS_Comscfd.xlsx` | `Stkcd`, `Accper`, 各现金流科目编码 | 现金流稳定性特征 |
| 股票公司主数据 | `股票市场/TRDNEW_Co.xlsx` | `Stkcd`, `Listdt`, `Markettype`, `Statco`, `Nnindcd` | 股票样本池筛选、交易数据映射 |
| 日个股回报率（批次1） | `日个股回报率文件021102515(仅供北京大学使用)/TRD_Dalyr*.xlsx` | `Stkcd`, `Trddt`, `Clsprc`, `Dretwd`, `Dretnd`, `Trdsta`, `Markettype` | 日频市场表现、波动与收益特征 |
| 日个股回报率（批次2） | `日个股回报率文件021539700(仅供北京大学使用)/TRD_Dalyr*.xlsx` | `Stkcd`, `Trddt`, `Clsprc`, `Dretwd`, `Dretnd`, `Trdsta`, `Markettype` | 补充分段交易期，增强时间连续性 |
| 日个股回报率（批次3） | `日个股回报率文件022416998(仅供北京大学使用)/TRD_Dalyr*.xlsx` | `Stkcd`, `Trddt`, `Clsprc`, `Dretwd`, `Dretnd`, `Trdsta`, `Markettype` | 扩展覆盖区间，支持滚动窗口建模 |
| 上市公司信用评级 | `上市公司信用评级情况表160746111(仅供北京大学使用)/DEBT_BOND_RATING.xlsx` | `Symbol`, `DeclareDate`, `RatingDate`, `LongTermRating`, `RatingProspect`, `RatingInstitution` | 违约风险先验、信用状态迁移特征 |

### 2. 新增数据补充说明（股市 + 信用评级）

- 日个股回报率数据已在仓库内落地为 3 个批次目录，字段结构一致，主表均为 `TRD_Dalyr*.xlsx`。
- 信用评级数据已落地为 `DEBT_BOND_RATING.xlsx`，可按公司-时间维度并入样本。
- 证券代码统一建议：优先将 `Symbol` 映射为 `Stkcd` 后再做跨表 join，避免主键不一致。
- 评级特征建议：`LongTermRating` 按等级有序编码，`RatingProspect` 做类别编码并保留变动事件。

### 3. 数据主键与时间对齐约定

- 公司主键：`Stkcd`（评级表中的 `Symbol` 需先映射）
- 时间主键：财务类按 `Accper/Reptdt`，事件类按公告日/发生日，交易类按 `Trddt`，评级按 `RatingDate/DeclareDate`
- 初始样本范围：A股上市公司（后续可按研究需要扩展）

## 预测标签

（待补充）

## 数据处理流程

### 1. 脚本入口

- 脚本：`scripts/prepare_data_stage1.py`
- 依赖：项目内 `third_party/python/openpyxl`（已安装）

### 2. 一键运行（推荐）

- `bash scripts/run_all.sh`
- 可选：若要同时构建日频市场大表，使用 `RUN_DAILY=1 bash scripts/run_all.sh`
- 可选：若要在流程末尾训练 DGL 版 HTGNN，使用 `RUN_DGL=1 bash scripts/run_all.sh`
- 可选：若要在流程末尾训练当前最优配置的 DGL 改进版，使用 `RUN_DGL_IMPROVED=1 bash scripts/run_all.sh`

### 3. 分步运行方式

- 快速版（节点 + 担保边 + 评级事件，不跑超大财务表）：
  - `PYTHONPATH=./third_party/python python3 scripts/prepare_data_stage1.py --skip-financial`
- 财务版（节点 + 财务特征）：
  - `PYTHONPATH=./third_party/python python3 scripts/prepare_data_stage1.py --skip-rating --skip-guarantee`
- 全量版（最慢，可额外带日频）：
  - `PYTHONPATH=./third_party/python python3 scripts/prepare_data_stage1.py --include-daily`

### 3. 处理规则（当前已实现）

- 统一跳过 CSMAR 文件中字段说明/单位两行（`skiprows=[1,2]`）。
- 证券代码标准化为 6 位字符串（`Stkcd/Symbol -> 000001` 格式）。
- 日期字段统一转为 `YYYY-MM-DD`。
- 数值字段统一做 `to_numeric`，非法值转空值。
- 担保边中被担保方名称做名称映射（上市公司全称/简称精确匹配）并输出匹配标记。

### 4. 当前输出（`processed/stage1/`）

- `nodes_company.csv`：3915 家 A 股样本公司。
- `features_financial.csv`：155771 条公司-报告期财务特征，覆盖 3915 家公司。
- `events_rating.csv`：8529 条信用评级事件，覆盖 936 家公司。
- `edges_guarantee.csv`：228948 条担保事件边，源公司覆盖 2423 家。
- `summary.json`：本次运行统计摘要。

### 5. 当前数据结论

- 财务特征覆盖率高（对样本公司覆盖率 100%）。
- 评级覆盖率约 23.9%，可作为增强信号但不能作为唯一标签来源。
- 担保目标映射到“上市公司节点”的比例很低（约 0.19%），说明多数被担保方是非上市主体。
- 因此建图建议：
  - 第一阶段先用“上市公司节点 + 担保出边统计特征”做静态多关系模型。
  - 第二阶段再扩展“非上市企业节点”或引入企业名称实体图，提升风险传导路径完整性。

### 6. 最终可用数据（上市公司口径）

| 文件 | 说明 | 关键字段 |
|---|---|---|
| `processed/stage1/nodes_company.csv` | 上市公司样本池（A 股） | `Stkcd`, `Listdt`, `Markettype`, `Nnindcd`, `Regcap` |
| `processed/stage1/features_financial.csv` | 财务特征（公司-报告期） | `Stkcd`, `Accper`, `A001101000`, `A001109000`, `A001111000`, `B001100000`, `C001001000` |
| `processed/stage1/features_guarantee_yearly.csv` | 担保暴露特征（公司-年份） | `src_stkcd`, `year`, `guar_event_cnt`, `guar_amt_sum`, `guar_listed_target_ratio` |
| `processed/stage1/events_rating.csv` | 信用评级事件（公司-事件时间） | `Stkcd`, `event_date`, `LongTermRating`, `RatingProspect`, `RatingInstitution` |
| `processed/stage1/edges_guarantee_listed_to_listed.csv` | 上市公司→上市公司担保边（较稀疏） | `src_stkcd`, `dst_stkcd`, `event_date`, `ActualGuaranteeAmount` |
| `processed/stage1/panel_company_year.csv` | 训练面板（公司-年份） | 见下方字段说明 |

### 7. 训练面板字段说明（`panel_company_year.csv`）

- 主键：`Stkcd`, `year`
- 公司静态字段：`Markettype`, `Nnindcd`, `Statco`, `Regcap`
- 财务字段：`total_assets`, `total_liabilities`, `total_equity`, `revenue_total`, `revenue_main`, `cashflow_operating`, `asset_liability_ratio`
- 担保字段：`guar_event_cnt`, `guar_amt_sum`, `guar_amt_mean`, `guar_listed_target_cnt`, `guar_nonlisted_target_cnt`, `guar_listed_target_ratio`
- 评级字段：`rating_event_cnt`, `rating_agency_nunique`, `rating_longterm_nonnull_cnt`, `rating_latest_longterm`, `rating_latest_prospect`
- 标签占位：`label_default_next_year`, `label_st_next_year`（当前为空，待标签工程）
- 覆盖标记：`has_financial`, `has_rating`

### 8. 训练面板生成命令

- `python3 scripts/build_training_panel.py`
- 输出：
  - `processed/stage1/panel_company_year.csv`
  - `processed/stage1/panel_company_year_summary.json`

### 9. 评级任务训练数据（已构造）

- 脚本：`python3 scripts/build_rating_task_dataset.py`
- 输出目录：`processed/stage1/rating_task/`
  - `rating_panel_labeled.csv`：评级任务总样本（公司-年份）
  - `train.csv`：训练集（2010-2018）
  - `val.csv`：验证集（2019-2021）
  - `test.csv`：测试集（2022-2024）
  - `summary.json`：样本统计

当前构造结果：
- 总样本：2987
- 覆盖公司：857
- 年份范围：2011-2023（因需同时存在当年评级与下一年评级）
- 降级标签占比（`label_downgrade_next_year`）：约 4.79%

关键字段：
- 当前评级：`rating_norm`, `rating_score`, `rating_latest_prospect`, `prospect_code`
- 下一年标签：`label_rating_next_year`, `label_rating_score_next_year`, `label_downgrade_next_year`
- 其余财务/担保/静态字段与 `panel_company_year.csv` 保持一致。

### 10. 最终版异构时序图（已构造）

- 脚本：`PYTHONPATH=./third_party/python python3 scripts/build_final_hetero_temporal_graph.py`
- 输出目录：`processed/final_hetero_temporal_graph/`
  - `node_mapping.csv`：统一节点索引（3915 家上市公司）
  - `feature_schema.json`：节点特征字段与关系类型定义
  - `metadata.json`：全局与逐年规模统计
  - `snapshots/{year}/node_features.csv`：年度节点特征快照
  - `snapshots/{year}/edges_guarantee.csv`：担保关系边（上市公司→上市公司）
  - `snapshots/{year}/edges_equity_assoc.csv`：股权关联边
  - `snapshots/{year}/edges_co_controller.csv`：同实控人关联边

当前构造统计（2010-2024）：
- 节点总数：3915
- 时间快照数：15
- 边总量：
  - `guarantee = 40`
  - `equity_assoc = 456`
  - `co_controller = 44216`

说明：
- 关系边已去除自环（`src != dst`）。
- 该图包为框架无关格式（CSV + JSON），可直接对接 DGL/PyG。

## 模型与实验

### 1. 评级下调二分类 baseline（已实现）

- 脚本：`python3 scripts/train_rating_downgrade_baseline.py`
- 输入：
  - `processed/stage1/rating_task/train.csv`
  - `processed/stage1/rating_task/val.csv`
  - `processed/stage1/rating_task/test.csv`
  - `processed/stage1/edges_guarantee_listed_to_listed.csv`
- 输出目录：`processed/stage1/rating_task/experiments/`
  - `baseline_metrics.json`
  - `val_predictions.csv`
  - `test_predictions.csv`

### 2. 特征与方法

- 主任务：`label_downgrade_next_year`（下一年是否评级下调）
- 模型：
  - `logreg_balanced`（带类别权重的逻辑回归）
  - `rf_balanced`（带类别权重的随机森林）
- 特征：
  - 训练面板原始字段（财务 + 担保暴露 + 当前评级 + 静态属性）
  - 新增关系特征：由 `edges_guarantee_listed_to_listed.csv` 生成的年度图特征（入/出事件数、邻居数、金额及其累计量）

### 3. 当前结果（best: `rf_balanced`）

- 验证集（2019-2021）：
  - `AUC = 0.8222`
  - `AP = 0.4023`
  - `Precision@5% = 0.3929`
  - `Recall@5% = 0.3548`
- 测试集（2022-2024）：
  - `AUC = 0.7104`
  - `AP = 0.2115`
  - `Precision@5% = 0.2609`
  - `Recall@5% = 0.2857`

> 该 baseline 用于后续 HTGNN/多关系图模型对照，建议统一沿用同一时间切分与指标集合。

### 4. DGL 版 HTGNN（已实现）

- 脚本：
  - `PYTHONPATH=./third_party/python DGLDEFAULTDIR=./.dgl DGLBACKEND=pytorch python3 scripts/train_htgnn_dgl.py --device cpu`
- 输出目录：`processed/final_hetero_temporal_graph/experiments_htgnn_dgl/`
  - `metrics.json`
  - `train_curve.csv`
  - `val_predictions.csv`
  - `test_predictions.csv`
  - `best_model.pt`

当前一轮训练结果（window=3）：
- 验证集：`AUC = 0.5611`, `AP = 0.0923`
- 测试集：`AUC = 0.5397`, `AP = 0.0489`

注：当前 DGL 版本已跑通流程，但指标低于随机森林 baseline，后续需要重点优化图结构与训练策略。

### 5. DGL 改进版（多维度改进 + 继续实验）

- 脚本：
- `PYTHONPATH=./third_party/python DGLDEFAULTDIR=./.dgl DGLBACKEND=pytorch python3 scripts/train_htgnn_dgl_improved.py --device cpu --exp-name v10_all_notab_bce_w6_h128 --relations guarantee,shared_nonlisted_guarantee,equity_assoc,equity_change,co_controller,market_corr --no-tabular-residual --window 6 --hidden-dim 128 --dropout 0.25 --lr 5e-4 --weight-decay 8e-4 --loss bce --epochs 220 --patience 45 --seed 42`
- 改进点（全过程）：
  - 数据/特征：加入每年关系图结构统计（各关系入/出度、权重和）、`log1p` 变换、`lag1` 与 `chg1` 时序特征。
  - 模型结构：关系注意力空间编码（relation attention）+ GRU + 时间注意力（temporal attention）。
  - 训练策略：`BCEWithLogitsLoss/Focal BCE`（含类别不平衡权重）+ AdamW + LR 调度 + 早停。
  - 评估策略：在验证集搜索最优阈值（by F1），同时报告 `@0.5` 与 `@tuned` 指标。
  - 继续优化：加入关系可配置、`BCE/Focal` 可切换、tabular 残差可切换，并做关系消融 + 多种子复现实验。
- 输出目录：
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/{exp_name}/`
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/best_current/`（当前最佳实验快照）
- 对比汇总：
  - `processed/final_hetero_temporal_graph/experiments_htgnn_dgl_improved/comparison_summary.json`

当前多组实验结果（测试集 AP）：
- 旧 DGL：`0.0489`
- 改进 `v1_w3_h96`：`0.1017`
- 改进 `v2_w5_h128`：`0.1098`
- 改进 `v3_w4_h128`：`0.1525`
- 改进 `v10_all_notab_bce_w6_h128`：`0.1673`（当前最佳，`AUC=0.7724`）

结论：
- 改进版相对旧 DGL 有明显提升（测试 AP 从 `0.0489` 提升到 `0.1673`，约 `3.42x`），但仍低于 RF baseline（`0.2115`），后续需继续优化关系构图与训练目标。

## 结果与分析

（待补充）

## 参考与附录

（待补充）
