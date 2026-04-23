"""
结构处理：支持 Mock 模式（无 PDB 也能跑）
"""

import numpy as np
import torch

T7_LENGTH = 883  # T7 RNAP 长度


class MockStructureProcessor:
    """
    零依赖的结构处理器。
    真实场景替换为 PDBParser 解析 1CEZ.pdb。
    """

    def __init__(self, length=T7_LENGTH):
        self.num_residues = length
        self.ca_coords = np.random.randn(length, 3) * 10  # 假坐标

    def build_graph(self, k=8):
        # 返回随机边（真实场景用 KNN/半径图）
        N = self.num_residues
        edge_index = torch.randint(0, N, (2, N * k))
        edge_attr = torch.randn(N * k, 3)
        return edge_index, edge_attr

    def extract_node_features(self):
        # 返回 [N, 20] 假结构特征
        return torch.randn(self.num_residues, 20)

    def get_mutation_context(self, site, radius=10.0):
        mask = torch.zeros(self.num_residues, dtype=torch.bool)
        # 模拟邻居：前后5个残基
        start = max(0, site - 5)
        end = min(self.num_residues, site + 6)
        mask[start:end] = True
        return mask
