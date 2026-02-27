import numpy as np
import random

class AllOnAgent:
    """
    全开模式策略：无论状态如何，所有基站始终保持开启 (Action = 1)。
    用于评估系统的性能上限 (QoS 最好) 和能耗下限 (最费电)。
    """
    def __init__(self, num_actions=2):
        self.num_actions = num_actions

    def select_actions(self, node_features, adj):
        """
        node_features: [N, F]
        adj: [N, N]
        返回: [N] 大小的全 1 数组
        """
        num_nodes = node_features.shape[0]
        # 全 1 代表所有基站 Active
        return np.ones(num_nodes, dtype=int)

class RandomAgent:
    """
    随机策略：每个基站以 50% 概率随机开启或休眠。
    用于评估算法的下限，证明 DRL 确实学到了东西 (如果 DRL 比随机还差，那就是失败)。
    """
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def select_actions(self, node_features, adj):
        """
        返回: [N] 大小的随机 0/1 数组
        """
        num_nodes = node_features.shape[0]
        # 生成 0 或 1 的随机整数
        return np.random.randint(0, 2, size=num_nodes)

class FullSleepAgent:
    """
    (可选) 全关模式：用于测试极端情况，理论上 QoS 会全崩。
    """
    def select_actions(self, node_features, adj):
        num_nodes = node_features.shape[0]
        return np.zeros(num_nodes, dtype=int)