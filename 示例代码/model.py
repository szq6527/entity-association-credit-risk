# multirel_gat_project/model.py

import torch
import torch.nn.functional as F
from torch.nn import ModuleDict, Linear
from torch_geometric.nn import GATConv

class MultiRelationalGAT(torch.nn.Module):
    """
    一个处理同构节点、多重关系的HAN-like模型。
    实现了两层注意力机制：
    1. 关系内注意力 (Intra-relation attention)
    2. 关系间注意力 (Inter-relation attention)
    """
    def __init__(self, in_channels, hidden_channels, out_channels, relations, heads=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.relations = relations

        # --- 1. 关系内注意力层 (Intra-relation) ---
        # 为每一种关系创建一个独立的GAT卷积层
        self.gat_convs = ModuleDict({
            # 'relation_name': GATConv(...)
            # PyG的异构API会自动处理元组形式的relation_name
            '_'.join(rel): GATConv((-1, -1), hidden_channels, heads=heads)
            for rel in relations
        })

        # --- 2. 关系间注意力层 (Inter-relation) ---
        # 这个线性层+激活函数用于计算不同关系的重要性得分
        # 输入是每个节点聚合后的表示 (hidden_channels)
        self.semantic_attention = Linear(hidden_channels * heads, 1)
        
        # --- 3. 输出层 ---
        # 最终的分类器
        self.out_lin = Linear(hidden_channels * heads, out_channels)

    def forward(self, x_dict, edge_index_dict):
        # x_dict: {'company': [num_nodes, in_features]}
        # edge_index_dict: {('company', 'rel', 'company'): [2, num_edges]}
        
        # --- 关系内聚合 ---
        # 对每一种关系，使用其专属的GAT层进行信息聚合
        relation_embeddings = []
        for rel in self.relations:
            # 获取源节点和目标节点的类型，虽然这里都是'company'
            src, rel_type, dst = rel
            # 根据关系从字典中获取对应的边和GAT层
            edge_index = edge_index_dict[rel]
            gat_conv = self.gat_convs['_'.join(rel)]
            
            # GAT聚合，输入是源和目标节点的特征元组
            # 这里源和目标是同一种类型，所以都是x_dict[src]
            emb = gat_conv((x_dict[src], x_dict[dst]), edge_index)
            relation_embeddings.append(emb)
        
        # relation_embeddings 是一个列表，包含每个关系聚合后的节点表示
        # list of [num_nodes, hidden_channels * heads]

        # --- 关系间聚合 (Semantic Attention) ---
        # 将所有关系聚合的嵌入堆叠起来，准备计算注意力
        # shape: [num_nodes, num_relations, hidden_channels * heads]
        stacked_embeddings = torch.stack(relation_embeddings, dim=1)
        
        # 计算每种关系对每个节点的重要性得分 (alpha)
        # 1. 使用tanh激活函数增加非线性
        # 2. 通过线性层得到一个分数
        # 3. squeeze(-1)去掉最后一个维度
        # shape: [num_nodes, num_relations]
        attn_scores = self.semantic_attention(torch.tanh(stacked_embeddings)).squeeze(-1)
        
        # 使用softmax将分数转换为注意力权重 (beta)
        # shape: [num_nodes, num_relations]
        attn_weights = F.softmax(attn_scores, dim=1)

        # 加权求和，融合所有关系的信息
        # unsqueeze(-1)是为了让权重能和嵌入进行广播乘法
        # shape: [num_nodes, hidden_channels * heads]
        final_embedding = torch.sum(attn_weights.unsqueeze(-1) * stacked_embeddings, dim=1)

        # --- 输出 ---
        # 经过最终的线性层进行分类
        out = self.out_lin(final_embedding)
        
        # 返回原始logits，损失函数将处理softmax
        return {'company': out}