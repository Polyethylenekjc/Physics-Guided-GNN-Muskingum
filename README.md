# Physics-Guided GNN Muskingum

基于物理约束的流域尺度多站点径流预测模型。每个站点采用 LSTM 编码历史序列，通过跨站点特征选择与图传播融合上游信息，并使用 Muskingum 汇流规律进行物理约束。

## 主要特性

- 多站点联合预测（一次训练输出所有站点）
- 变量级因果筛选（格兰杰因果 + 上游约束 + top-k 变量）
- Muskingum 物理汇流约束与物理损失
- 可配置训练参数与断点续训
- Rich 进度条与阶段日志

## 数据结构

```
Data/Processed/
  runoffs/<Station>.csv
  weather/<Station>.csv
```

`runoffs` 至少包含一列 `runoff`；`weather` 可包含 `runoff` 列（会自动剔除）。

## 快速开始

1) 安装依赖

```
pip install -r requirements.txt
```

2) 训练

```
PYTHONPATH=src python scripts/train_multistation.py --config config/pg_gnn.yaml
```

## 配置说明

配置文件：`config/pg_gnn.yaml`

- `history`: 历史时间窗口长度（默认 15 天）
- `horizon`: 预测步长
- `variable_top_k`: 变量级因果筛选的 top-k（默认 9）
- `station_top_k`: 跨站点注意力的 top-k
- `lambda_phy`: 物理损失权重
- `dt`: Muskingum 时间步长
- `resume`: 是否断点续训

## 输出文件

- `output/variable_causal_scores.csv`: 变量级因果分数缓存
- `output/variable_selection.csv`: 每个站点的最终变量选择
- `output/selection_weights_epoch_XXX.csv`: 注意力权重统计
- `output/checkpoint.pt`: 训练断点

## 代码结构

```
src/pg_gnn/
  data.py                 # 数据集与特征选择
  causal.py               # 格兰杰因果分析
  graph.py                # 拓扑与掩码
  model/
    node_encoder.py       # LSTM 编码器
    cross_station.py      # 跨站点选择
    graph_layer.py        # 图传播层
    routing.py            # Muskingum 系数
    loss.py               # 物理损失
    model.py              # 模型封装
scripts/
  train_multistation.py   # 训练脚本
```

## 备注

- 训练前请检查 `config/pg_gnn.yaml` 中的站点顺序是否与实际流域拓扑一致。
- 如果更新因果分析逻辑或变量集合，建议删除 `output/variable_causal_scores.csv` 以强制重算。
