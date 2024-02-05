import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv


def get_base_model(name):
    # 这段代码是实现基础图神经网络模型的一段示例代码。
    # 它可以帮助开发者更方便地使用PyTorch中的图神经网络功能，并且提供了几个常用的基础模型实现，例如GAT、GCN、GraphSAGE、SG、或GIN等。
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name):
    # 这段代码提供了一个简单的函数，用于获取指定名称的激活函数对象。
    # 通过传入不同的名称，可以获取常用的激活函数，例如ReLU、Hardtanh、ELU、LeakyReLU等。这个函数可以帮助开发者更方便地设置和使用不同的激活函数。
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def generate_split(num_samples, train_ratio, val_ratio, generator=None):
    # 定义一个函数，用于生成数据集划分的掩码张量
    # 参数num_samples表示样本数量
    # 参数train_ratio表示训练集的比例
    # 参数val_ratio表示验证集的比例
    # 参数generator表示随机数生成器，默认为None

    # 计算训练集、验证集和测试集的长度
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    # 创建一个包含从0到num_samples的张量作为数据
    data = torch.arange(0, num_samples)
    dataset = TensorDataset(data)

    # 使用random_split函数将数据集划分为训练集、测试集和验证集
    train_set, test_set, val_set = random_split(dataset, (train_len, test_len, val_len), generator=generator)

    # 获取训练集、测试集和验证集的索引
    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices

    # 创建三个掩码张量，用于选择相应的样本
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    # 将相应的索引位置设置为True，表示对应的样本被选中
    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    # 返回训练集、测试集和验证集的掩码张量
    return train_mask, test_mask, val_mask
