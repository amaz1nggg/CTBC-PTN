import numpy as np
import networkx as nx
import torch
from typing import Sequence
from cdlib import algorithms
from cdlib.utils import convert_graph_formats


def community_detection(name):
#这段代码定义了一个名为community_detection()的函数，用于根据输入的算法名称返回相应的社区检测算法。
#这段代码的作用是根据给定的算法名称，返回相应的社区检测算法函数。
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]

def ced(edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        p: float,
        threshold: float = 1.) -> torch.Tensor:
# 这段代码定义了一个名为ced()的函数，该函数实现了一种基于边权重和阈值的条件边采样算法。
# 换句话说，ced()函数根据边权重和阈值对边进行采样，并返回被选中的边的边索引。
    edge_weight = edge_weight / edge_weight.mean() * (1. - p)
    edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
    edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
    sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
    return edge_index[:, sel_mask]

def cav_dense(feature: torch.Tensor,
              node_cs: np.ndarray,
              p: float,
              max_threshold: float = 0.7) -> torch.Tensor:
#这段代码定义了一个名为cav_dense()的函数，该函数实现了一种基于节点聚类系数和阈值的条件稠密化算法。
#换句话说，cav_dense()函数根据节点聚类系数和阈值对节点特征进行稠密化，即根据权重和阈值将一部分特征置为0。
    x = feature.abs()
    w = x.t() @ torch.tensor(node_cs).to(feature.device)
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, torch.ones_like(w) * max_threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def cav(feature: torch.Tensor,
        node_cs: np.ndarray,
        p: float,
        max_threshold: float = 0.7) -> torch.Tensor:
#这段代码定义了一个名为cav()的函数，它实现了一种基于节点聚类系数和阈值的条件自适应稀疏化算法。
#换句话说，cav()函数根据节点聚类系数和阈值对节点特征进行自适应稀疏化，即根据权重和阈值将一部分特征置为0，并处理了数据集中冗余属性的情况。
    x = feature.abs()
    device = feature.device
    w = x.t() @ torch.tensor(node_cs).to(device)
    w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
    w = w.log()
    w = (w.max() - w) / (w.max() - w.min())
    w = w / w.mean() * p
    w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
    w = w.where(w > 0, torch.zeros(1).to(device))
    drop_mask = torch.bernoulli(w).to(torch.bool)
    feature = feature.clone()
    feature[:, drop_mask] = 0.
    return feature


def transition(communities: Sequence[Sequence[int]],
               num_nodes: int) -> np.ndarray:
#创建一个长度为num_nodes的数组classes，并用-1进行填充。
#遍历communities中的每个社区，将社区中的节点索引映射到classes数组中对应的位置上，并赋值为该社区的索引值。
#返回classes数组，其中每个元素表示对应节点所属的社区或类别。
    classes = np.full(num_nodes, -1)
    for i, node_list in enumerate(communities):
        classes[np.asarray(node_list)] = i
    return classes


def get_edge_weight(edge_index: torch.Tensor,
                    com: np.ndarray,
                    com_cs: np.ndarray) -> torch.Tensor:
#定义了一个匿名函数edge_mod，接受一个包含两个节点索引的列表x，根据节点所属社区的聚类系数计算边的权重。
#如果两个节点属于同一个社区，则权重为该社区的聚类系数；否则，权重为两个社区聚类系数之和的负数。
#定义了一个匿名函数normalize，接受一个数组x，将数组进行归一化处理，将值映射到0到1的范围内。
#根据输入的边索引edge_index和节点社区信息com，使用edge_mod函数计算每条边的权重，形成一个NumPy数组edge_weight。
#将edge_weight进行归一化处理，得到的结果作为一个Tensor并返回。
    edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
    normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
    edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
    edge_weight = normalize(edge_weight)
    return torch.from_numpy(edge_weight).to(edge_index.device)


def community_strength(graph: nx.Graph,
                       communities: Sequence[Sequence[int]]) -> (np.ndarray, np.ndarray):
#总结起来，这段代码的作用是根据输入的图对象和节点所属的社区信息计算社区强度，并将每个节点的社区强度进行赋值和返回。
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs
