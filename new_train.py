import argparse
import os.path as osp
import random
from typing import Dict

from torch_geometric.utils import to_networkx
import sys
sys.path.append('/project/src')
from src import *
import numpy as np
def train(epoch: int) -> int:
    # 这段代码实现了一个训练函数，其中对数据进行预处理，并根据模型计算得到的结果计算损失函数，并进行梯度更新。
    model.train()  # 将模型设置为训练模式
    optimizer.zero_grad()  # 清空梯度

    # 使用CED方法对边进行预处理
    edge_index_1 = ced(data.edge_index, edge_weight, p=param['ced_drop_rate_1'], threshold=args.ced_thr)
    edge_index_2 = ced(data.edge_index, edge_weight, p=param['ced_drop_rate_2'], threshold=args.ced_thr)
    edge_index_3 = data.edge_index

    if args.dataset == 'WikiCS':
        # 对节点特征进行预处理，使用CAV方法
        x1 = cav_dense(data.x, node_cs, param["cav_drop_rate_1"], max_threshold=args.cav_thr)
        x2 = cav_dense(data.x, node_cs, param["cav_drop_rate_2"], max_threshold=args.cav_thr)
        x3 = data.x.clone()
    else:
        x1 = cav(data.x, node_cs, param["cav_drop_rate_1"], max_threshold=args.cav_thr)
        x2 = cav(data.x, node_cs, param['cav_drop_rate_2'], max_threshold=args.cav_thr)
        x3 = data.x

    # 使用模型计算节点表示
    z1 = model(x1, edge_index_1)
    z2 = model(x2, edge_index_2)
    z3 = model(x3, edge_index_3)

    # 计算损失函数
    loss = model.team_up_loss(z1, z2, z3,
                              cs=node_cs,
                              current_ep=epoch,
                              t0=param['t0'],
                              gamma_max=param['gamma'],
                              batch_size=args.batch_size if args.dataset in ['Coauthor-CS'] else None)

    # 反向传播和参数优化
    loss.backward()
    optimizer.step()

    return loss.item()  # 返回损失函数的值

def test() -> Dict:
    # 这段代码实现了一个测试函数，用于在给定的测试数据上评估模型的性能。根据数据集类型的不同，使用不同的方式计算分类准确率，并返回一个包含准确率结果的字典。
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        z = model(data.x, data.edge_index)  # 使用模型计算节点表示

    res = {}  # 创建一个空字典用于存储结果
    seed = np.random.randint(0, 32767)  # 生成一个随机种子
    split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1,
                           generator=torch.Generator().manual_seed(seed))  # 生成数据集划分

    evaluator = MulticlassEvaluator()  # 创建一个多类别评估器

    if args.dataset == 'WikiCS':
        accs = []
        for i in range(20):
            cls_acc = log_regression(z, dataset, evaluator, split=f'wikics:{i}',
                                     num_epochs=800)  # 使用逻辑回归模型计算分类准确率
            accs.append(cls_acc['acc'])  # 将准确率添加到列表中
        acc = sum(accs) / len(accs)  # 求准确率的平均值
    else:
        cls_acc = log_regression(z, dataset, evaluator, split='rand:0.1',
                                 num_epochs=3000, preload_split=split)  # 使用逻辑回归模型计算分类准确率
        acc = cls_acc['acc']  # 获取准确率

    res["acc"] = acc  # 将准确率存储在结果字典中
    return res  # 返回结果字典


if __name__ == '__main__':

#解释命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--dataset_path', type=str, default="./datasets")
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=39788)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--verbose', type=str, default='train,eval')
    parser.add_argument('--cls_seed', type=int, default=12345)
    parser.add_argument('--val_interval', type=int, default=100)
    parser.add_argument('--cd', type=str, default='leiden')
    parser.add_argument('--ced_thr', type=float, default=1.)
    parser.add_argument('--cav_thr', type=float, default=1.)

#设置默认参数
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'ced_drop_rate_1': 0.3,
        'ced_drop_rate_2': 0.4,
        'cav_drop_rate_1': 0.1,
        'cav_drop_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1000,
        'weight_decay': 1e-5,
        't0': 500,
        'gamma': 1.,
        'projection_hidden_size':512,
        'projection_size':256,
        'moving_average_decay':0.5
    }

    param_keys = default_param.keys()

    # 为每个参数键添加命令行参数
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')

# 解析命令行参数
    args = parser.parse_args()

# 创建参数对象并应用命令行参数
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

# 将命令行参数的值赋给参数字典中对应的键
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)

# 构造注释字符串
    comment = f'{args.dataset}_node_{param["cav_drop_rate_1"]}_{param["cav_drop_rate_2"]}'\
            f'_edge_{param["ced_drop_rate_1"]}_{param["ced_drop_rate_2"]}'\
            f'_t0_{param["t0"]}_gamma_{param["gamma"]}'

    print(f"training settings: \n"
          f"data: {args.dataset}\n"
          f"community detection method: {args.cd}\n"
          f"device: {args.device}\n"
          f"batch size if used: {args.batch_size}\n"
          f"communal edge dropping (ced) rate: {param['ced_drop_rate_1']}/{param['ced_drop_rate_2']}\n"
          f"communal attr voting (cav) rate: {param['cav_drop_rate_1']}/{param['cav_drop_rate_2']}\n"
          f"gamma: {param['gamma']}\n"
          f"t0: {param['t0']}\n"
          f"epochs: {param['num_epochs']}\n"
          )

    random.seed(12345)
    torch.manual_seed(args.seed)
    # 设置随机种子，用于生成随机数
    device = torch.device(args.device)
    # 设置设备，将args.device赋值给device

    path = osp.join(args.dataset_path, args.dataset)
    # 构建数据集路径，使用args.dataset_path和args.dataset拼接而成

    # 加载 data_undirected.pt 文件
    dataset = torch.load('dataset/WikiCS/processed/data_undirected.pt')

    data = dataset[0]
    # 获取数据集中的第一个数据

    print('Detecting communities...')
    # 打印信息，表示正在检测社区

    g = to_networkx(data, to_undirected=True)
    # 将data转换为networkx图的格式，设置为无向图

    communities = community_detection(args.cd)(g).communities
    # 使用args.cd指定的社区检测算法，在图g上进行社区检测，并获取检测到的社区列表

    com = transition(communities, g.number_of_nodes())
    # 对检测到的社区进行转换，转换为节点编号的形式

    com_cs, node_cs = community_strength(g, communities)
    # 计算社区的强度，得到社区强度列表和节点强度列表

    edge_weight = get_edge_weight(data.edge_index, com, com_cs)
    # 获取边的权重，传入边索引、转换后的社区和社区强度作为参数

    com_size = [len(c) for c in communities]
    # 计算每个社区的大小，将结果存储在com_size列表中

    print(f'Done! {len(com_size)} communities detected. \n')
    # 打印信息，表示社区检测完成，输出检测到的社区数量

    gat_model = gat(out_channels=param['num_hidden'])
    encoder = Encoder(
        gat_model,
        param['projection_hidden_size'],
        param['projection_size']
    ).to(device)

    model = CSGCL(encoder,
                  gat_model,
                  param['num_hidden'],
                  param['num_proj_hidden'],
                  param['projection_hidden_size'],
                  param['projection_size'],
                  param['moving_average_decay'],
                  param['tau']
                  ).to(device)
    # 实例化一个CSGCL对象，传入encoder对象、隐藏层的数量、投影层的隐藏层数量、tau值、投影层的隐藏层大小、投影层的大小、移动平均衰减值作为参数，并将其转移到指定设备上

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    # 使用Adam优化器，传入model的参数、学习率和权重衰减作为参数

    last_epoch = 0
    # 初始化最后一个epoch为0

    log = args.verbose.split(',')
    # 将args.verbose按逗号分隔，并将结果存储在log列表中

    for epoch in range(1 + last_epoch, param['num_epochs'] + 1):
        loss = train(epoch)
        # 调用train函数进行训练，并将返回的损失值赋值给loss变量

        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
            # 如果'train'在log列表中，则打印训练过程中的epoch和损失值

        if epoch % args.val_interval == 0:
            res = test()
            # 调用test函数进行测试，并将返回的结果赋值给res变量

            if 'eval' in log:
                print(f'(E) | Epoch={epoch:04d}, avg_acc = {res["acc"]}')
                # 如果'eval'在log列表中，则打印测试过程中的epoch和平均准确率
