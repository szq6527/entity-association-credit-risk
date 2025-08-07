# multirel_gat_project/main.py

import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data_loader import get_synthetic_data
from model import MultiRelationalGAT

def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    # 模型的输入是字典
    out = model(data.x_dict, data.edge_index_dict)
    
    # 获取 'company' 节点的输出和标签
    out = out['company']
    y = data['company'].y
    
    # 使用训练掩码计算损失
    mask = data['company'].train_mask
    loss = criterion(out[mask], y[mask])
    
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    
    out = model(data.x_dict, data.edge_index_dict)
    out = out['company']
    y = data['company'].y
    
    # 获取概率，用于计算AUC
    prob = F.softmax(out, dim=1)[:, 1] # 只取类别1的概率
    
    # 分别在验证集和测试集上计算指标
    val_mask = data['company'].val_mask
    test_mask = data['company'].test_mask
    
    # 验证集
    val_loss = F.cross_entropy(out[val_mask], y[val_mask])
    val_pred = out[val_mask].argmax(dim=1)
    val_acc = (val_pred == y[val_mask]).sum() / val_mask.sum()
    val_auc = roc_auc_score(y[val_mask].cpu(), prob[val_mask].cpu())

    # 测试集
    test_loss = F.cross_entropy(out[test_mask], y[test_mask])
    test_pred = out[test_mask].argmax(dim=1)
    test_acc = (test_pred == y[test_mask]).sum() / test_mask.sum()
    test_auc = roc_auc_score(y[test_mask].cpu(), prob[test_mask].cpu())

    return (val_loss, val_acc, val_auc), (test_loss, test_acc, test_auc)

def main():
    # 1. 加载数据
    data, num_features, num_classes = get_synthetic_data()

    # 2. 设置设备
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    data = data.to(device)

    # 3. 初始化模型、优化器和损失函数
    model = MultiRelationalGAT(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=num_classes,
        relations=data.edge_types,
        heads=4
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    # CrossEntropyLoss 内部包含了 LogSoftmax 和 NLLLoss
    criterion = torch.nn.CrossEntropyLoss()

    print("\n--- Starting Training ---")
    
    # 4. 训练循环
    for epoch in range(1, 51): # 增加epoch数量以便观察收敛
        train_loss = train(model, data, optimizer, criterion)
        (val_loss, val_acc, val_auc), (test_loss, test_acc, test_auc) = test(model, data)
        
        if epoch % 10 == 0:
            print(f"Epoch: {epoch:03d}, Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f} | "
                  f"Test Acc: {test_acc:.4f}, Test AUC: {test_auc:.4f}")

    print("--- Training Finished ---")

if __name__ == "__main__":
    main()