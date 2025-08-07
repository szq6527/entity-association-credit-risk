# multirel_gat_project/data_loader.py

import torch
from torch_geometric.data import HeteroData
import numpy as np

def get_synthetic_data(num_nodes=200, num_features=32):
    """
    生成一个合成的、多关系、半监督、二分类图数据。

    - 节点类型: 1种 ('company')
    - 关系类型: 2种 ('invests', 'guarantees')
    - 任务: 节点二分类 ('risk' vs 'no_risk')
    """
    print("--- Generating Synthetic Multi-Relational Data ---")
    data = HeteroData()

    # 1. 创建同构节点：'company'
    data['company'].num_nodes = num_nodes
    data['company'].x = torch.randn(num_nodes, num_features)  # 随机生成节点特征

    # 2. 创建两种关系类型的边
    # 'company' --invests--> 'company'
    invests_edges = torch.randint(0, num_nodes, (2, 300), dtype=torch.long)
    data['company', 'invests', 'company'].edge_index = invests_edges

    # 'company' --guarantees--> 'company'
    guarantees_edges = torch.randint(0, num_nodes, (2, 500), dtype=torch.long)
    data['company', 'guarantees', 'company'].edge_index = guarantees_edges
    
    # 3. 创建二分类标签 (0 for 'no_risk', 1 for 'risk')
    # 假设有20%的节点是风险节点
    num_risk_nodes = int(num_nodes * 0.2)
    labels = torch.zeros(num_nodes, dtype=torch.long)
    risk_indices = np.random.choice(num_nodes, num_risk_nodes, replace=False)
    labels[risk_indices] = 1
    data['company'].y = labels

    # 4. 创建半监督学习的掩码 (masks)
    # 20% 训练, 30% 验证, 50% 测试
    indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * 0.2)
    val_size = int(num_nodes * 0.3)
    
    train_indices = torch.from_numpy(indices[:train_size])
    val_indices = torch.from_numpy(indices[train_size : train_size + val_size])
    test_indices = torch.from_numpy(indices[train_size + val_size:])

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data['company'].train_mask = train_mask
    data['company'].val_mask = val_mask
    data['company'].test_mask = test_mask

    print(f"Nodes: {num_nodes} companies")
    print(f"Node Features: {num_features} dims")
    print(f"Edge Types: {data.edge_types}")
    print(f"Training nodes: {train_mask.sum()}, Validation nodes: {val_mask.sum()}, Test nodes: {test_mask.sum()}")
    print("--------------------------------------------------")

    return data, num_features, 2 # 2 for binary classification

if __name__ == '__main__':
    get_synthetic_data()