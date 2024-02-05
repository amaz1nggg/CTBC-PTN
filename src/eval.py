from typing import Optional
import torch
from torch.optim import Adam
import torch.nn as nn
from new_model import LogReg

def get_idx_split(dataset, split, preload_split):
    # 定义一个函数，用于获取索引划分的结果
    # 参数dataset表示数据集
    # 参数split表示划分方法
    # 参数preload_split表示预加载的划分

    if split[:4] == 'rand':
        # 如果划分方法以'rand'开头，则采用随机划分
        train_ratio = float(split.split(':')[1])
        num_nodes = dataset[0].x.size(0)
        train_size = int(num_nodes * train_ratio)
        indices = torch.randperm(num_nodes)
        # 返回划分结果的字典，包含训练集、验证集和测试集的索引
        return {
            'train': indices[:train_size],
            'val': indices[train_size:2 * train_size],
            'test': indices[2 * train_size:]
        }
    elif split == 'ogb':
        # 如果划分方法为'ogb'，则调用数据集的get_idx_split方法获取划分结果
        return dataset.get_idx_split()
    elif split.startswith('wikics'):
        # 如果划分方法以'wikics'开头，则采用wikics划分
        split_idx = int(split.split(':')[1])
        # 返回划分结果的字典，包含训练集、验证集和测试集的索引
        return {
            'train': dataset[0].train_mask[:, split_idx],
            'test': dataset[0].test_mask,
            'val': dataset[0].val_mask[:, split_idx]
        }
    elif split == 'preloaded':
        # 如果划分方法为'preloaded'，则使用预加载的划分
        assert preload_split is not None, 'use preloaded split, but preloaded_split is None'
        train_mask, test_mask, val_mask = preload_split
        # 返回划分结果的字典，包含训练集、验证集和测试集的索引
        return {
            'train': train_mask,
            'test': test_mask,
            'val': val_mask
        }
    else:
        # 如果划分方法不匹配上述情况，则抛出异常
        raise RuntimeError(f'Unknown split type {split}')


def log_regression(z,
                   dataset,
                   evaluator,
                   num_epochs: int = 5000,
                   test_device: Optional[str] = None,
                   split: str = 'rand:0.1',
                   verbose: bool = False,
                   preload_split=None):
#这段代码定义了一个逻辑回归（logistic regression）的训练函数log_regression()，用于在给定的数据集上进行模型训练和评估。
#最后，返回一个包含最佳测试集准确率的字典。
    test_device = z.device if test_device is None else test_device
    z = z.detach().to(test_device)
    num_hidden = z.size(1)
    y = dataset[0].y.view(-1).to(test_device)
    num_classes = dataset[0].y.max().item() + 1
    classifier = LogReg(num_hidden, num_classes).to(test_device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    split = get_idx_split(dataset, split, preload_split)
    split = {k: v.to(test_device) for k, v in split.items()}
    f = nn.LogSoftmax(dim=-1)
    nll_loss = nn.NLLLoss()
    best_test_acc = 0
    best_val_acc = 0

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(z[split['train']])
        loss = nll_loss(f(output), y[split['train']])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            if 'val' in split:
                test_acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                val_acc = evaluator.eval({
                    'y_true': y[split['val']].view(-1, 1),
                    'y_pred': classifier(z[split['val']]).argmax(-1).view(-1, 1)
                })['acc']
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_test_acc = test_acc
                
            else:
                acc = evaluator.eval({
                    'y_true': y[split['test']].view(-1, 1),
                    'y_pred': classifier(z[split['test']]).argmax(-1).view(-1, 1)
                })['acc']
                
                if best_test_acc < acc:
                    best_test_acc = acc
               
            if verbose:
                print(f'logreg epoch {epoch}: best test acc {best_test_acc}, '
                     )

    return {'acc': best_test_acc}


class MulticlassEvaluator:
#这段代码定义了一个多类别评估器（MulticlassEvaluator）类，用于计算分类器在多类别任务上的性能指标。
#这个多类别评估器类主要用于评估分类器在多类别任务上的性能，通过计算准确率来衡量模型的分类精度。
    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _eval(y_true, y_pred):
        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)
        total = y_true.size(0)
        correct = (y_true == y_pred).to(torch.float32).sum()
        return (correct / total).item()
 
    def eval(self, res):
        return {'acc': self._eval(**res)}
