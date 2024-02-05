from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv


class MLP(nn.Module):
    '''对应论文中的p和predictor'''

    def __init__(self, inp_size, outp_size, hidden_size):
        '''
        inp_size: 输入的维度
        outp_size: 输出的维度
        hidden_size: 隐藏层的维度
        '''
        super().__init__()
        # MLP的架构是一个输入层，接一个隐藏层，再接PReLU进行非线性化处理，最后
        # 全连接层进行输出
        self.net = nn.Sequential(
            nn.Linear(inp_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, outp_size)
        )

    def forward(self, x):
        '''数据通过MLP时没有进行其他的处理，因此仅仅通过net()'''
        return self.net(x)

class gat(nn.Module):
    def __init__(self,
                 in_channels: int = 300,
                 out_channels: int = 200,
                 activation: nn.Module = nn.ReLU(),
                 k: int = 2,
                 skip: bool = False):
        super(gat, self).__init__()
        self.k = k
        self.skip = skip

        if not self.skip:
            # 如果不使用跳跃连接
            self.conv = nn.ModuleList([GATConv(in_channels, out_channels, heads=1)])

            # 堆叠k-1个GATConv层
            for _ in range(1, k - 1):
                self.conv.append(GATConv(out_channels, out_channels, heads=1))

            # 添加最后一个GATConv层
            self.conv.append(GATConv(out_channels, out_channels, heads=1))
        else:
            # 如果使用跳跃连接
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = nn.ModuleList([GATConv(in_channels, out_channels, heads=1)])

            # 堆叠k个GATConv层
            for _ in range(1, k):
                self.conv.append(GATConv(out_channels, out_channels, heads=1))

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if not self.skip:
            # 如果不使用跳跃连接
            for i in range(self.k):
                # 应用激活函数和GATConv层
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            # 如果使用跳跃连接
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]

            for i in range(1, self.k):
                # 对前面的输出进行累加，并应用激活函数和GATConv层
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))

            return hs[-1]

class Encoder(nn.Module):

    def __init__(self,
                 gat,
                 projection_hidden_size,
                 projection_size):

        super().__init__()

        self.gat = gat
        ## projector对应论文中的p，输入为512维，因为本文定义的嵌入表示的维度为512

    def forward(self, x, edge_index):
        representations = self.gat(x, edge_index)  # 初始的嵌入表示
        representations = representations.view(-1, representations.size(-1))
        return representations

class EMA():

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(ema_updater, ma_model, current_model):
    '''参数更新方式，MOCO-like'''
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class LogReg(nn.Module):
    # 这段代码定义了一个名为LogReg的神经网络模型，用于进行逻辑回归任务。
    # LogReg模型是一个简单的逻辑回归模型，它使用单个线性层对输入特征进行映射，并且通过weights_init()函数来初始化模型中的权重。
    # 在前向传播函数中，输入特征经过线性层的计算，得到输出结果。
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

class CSGCL(torch.nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 gat,
                 num_hidden: int,
                 num_proj_hidden: int,
                 projection_hidden_size,
                 projection_size,
                 moving_average_decay,
                 tau: float = 0.5):
        super(CSGCL, self).__init__()
        self.encoder = encoder
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_proj_hidden)
        self.tau = tau

        ## 三个分支的网络encoder初始化
        self.online_encoder = Encoder(
            gat, projection_hidden_size, projection_size)
        self.target_encoder1 = Encoder(
            gat, projection_hidden_size, projection_size)
        self.target_encoder1.load_state_dict(self.online_encoder.state_dict())

        self.target_encoder2 = Encoder(
            gat, projection_hidden_size, projection_size)
        self.target_encoder2.load_state_dict(self.online_encoder.state_dict())

        ## 目标网络不需要直接进行权重更新，接受online_encoder的权重进行MOCO-like更新
        set_requires_grad(self.target_encoder1, False)
        set_requires_grad(self.target_encoder2, False)

        self.target_ema_updater = EMA(moving_average_decay)

    def update_ma(self):
        assert self.target_encoder1 or self.target_encoder2 is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder1, self.online_encoder)
        update_moving_average(self.target_ema_updater,
                              self.target_encoder2, self.online_encoder)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tensor:
        z = self.encoder(x, edge_index)
        z1 = self.projection(z)
        return z1

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.fc1(z))
        z = torch.relu(self.fc2(z))
        return z

    def _sim(self,
             z1: torch.Tensor,
             z2: torch.Tensor,) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _infonce(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor) -> torch.Tensor:

        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1))
        between_sim = temp(self._sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_infonce(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self._sim(z1[mask], z1))
            between_sim = f(self._sim(z1[mask], z2))
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)

    def _team_up(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor,
                 cs: torch.Tensor,
                 current_ep: int,
                 t0: int,
                 gamma_max: int) -> torch.Tensor:
        # 计算gamma值，确保其在范围内
        gamma = min(max(0, int((current_ep - t0) / 100)), gamma_max)

        # 定义辅助函数
        temp = lambda x: torch.exp(x / self.tau)

        # 计算反射相似度
        refl_sim = temp(self._sim(z1, z1) + gamma * cs + gamma * cs.unsqueeze(dim=1))

        # 计算两个输入之间的相似度
        between_sim = temp(self._sim(z1, z2) + gamma * cs + gamma * cs.unsqueeze(dim=1))

        # 计算最终结果并返回
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_team_up(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         cs: torch.Tensor,
                         current_ep: int,
                         t0: int,
                         gamma_max: int,
                         batch_size: int) -> torch.Tensor:
        gamma = min(max(0, int((current_ep - t0) / 100)), gamma_max)
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        temp = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = temp(self._sim(z1[mask], z1) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])
            between_sim = temp(self._sim(z1[mask], z2) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def infonce(self,
                z1: torch.Tensor,
                z2: torch.Tensor,
                z3: torch.Tensor,
                mean: bool = True,
                batch_size: Optional[int] = None) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h3 = self.projection(z3)

        if batch_size is None:
            l1 = self._infonce(h1, h2)
            l2 = self._infonce(h2, h1)
            l3 = self._infonce(h1, h3)
            l4 = self._infonce(h3, h1)
            l5 = self._infonce(h2, h3)
            l6 = self._infonce(h3, h2)
        else:
            l1 = self._batched_infonce(h1, h2, batch_size)
            l2 = self._batched_infonce(h2, h1, batch_size)
            l3 = self._batched_infonce(h1, h3, batch_size)
            l4 = self._batched_infonce(h3, h1, batch_size)
            l5 = self._batched_infonce(h2, h3, batch_size)
            l6 = self._batched_infonce(h3, h2, batch_size)

        ret = (l1 + l2 + l3 + l4 + l5 + l6) / 6.0  # Calculate the average of all losses
        ret = ret.mean() if mean else ret.sum()

        return ret

    def team_up_loss(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     z3: torch.Tensor,
                     cs: np.ndarray,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     batch_size: Optional[int] = None) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        h3 = self.projection(z3)
        cs = torch.from_numpy(cs).to(h1.device)

        if batch_size is None:
            l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
            l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
            l3 = self._team_up(h1, h3, cs, current_ep, t0, gamma_max)
            l4 = self._team_up(h3, h1, cs, current_ep, t0, gamma_max)
            l5 = self._team_up(h2, h3, cs, current_ep, t0, gamma_max)
            l6 = self._team_up(h3, h2, cs, current_ep, t0, gamma_max)
        else:
            l1 = self._batched_team_up(h1, h2, cs, current_ep, t0, gamma_max, batch_size)
            l2 = self._batched_team_up(h2, h1, cs, current_ep, t0, gamma_max, batch_size)
            l3 = self._batched_team_up(h1, h3, cs, current_ep, t0, gamma_max, batch_size)
            l4 = self._batched_team_up(h3, h1, cs, current_ep, t0, gamma_max, batch_size)
            l5 = self._batched_team_up(h2, h3, cs, current_ep, t0, gamma_max, batch_size)
            l6 = self._batched_team_up(h3, h2, cs, current_ep, t0, gamma_max, batch_size)

        ret = (l1 + l2 + l3 + l4 + l5 + l6) / 6.0  # Calculate the average of all losses
        ret = ret.mean() if mean else ret.sum()

        return ret