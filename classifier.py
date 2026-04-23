import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import StructuralFeatureEncoder, SimpleGNN, MutationDeltaEncoder
from mechanism_heads import MechanismDisentanglementModule
from mechanism_ontology import ALL_MECHANISMS


class T7RNAPMechClassifier(nn.Module):
    def __init__(self, esm_dim=1280, struct_dim=20, hidden_dim=256,
                 num_gnn_layers=2, mechanism_list=None, dropout=0.2,
                 use_orthogonality=True):
        super().__init__()
        self.use_orthogonality = use_orthogonality

        self.struct_encoder = StructuralFeatureEncoder(struct_dim, hidden_dim//2, hidden_dim//2)
        self.gnn = SimpleGNN(esm_dim + hidden_dim//2, hidden_dim, num_gnn_layers, dropout)
        self.delta_encoder = MutationDeltaEncoder(hidden_dim)

        if mechanism_list is None:
            mechanism_list = ALL_MECHANISMS
        self.mech_module = MechanismDisentanglementModule(mechanism_list, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def encode_single(self, esm_emb, struct_feat, edge_index):
        """处理单个样本: esm_emb [N,D], struct_feat [N,D_s], edge_index [2,E]"""
        s = self.struct_encoder(struct_feat)
        x = torch.cat([esm_emb, s], dim=-1)
        h = self.gnn(x, edge_index)
        return h

    def forward(self, esm_wt, esm_mut, struct_feat, edge_index, mutation_mask):
        """
        支持 batch:
          esm_wt/mut: [B,N,D] or [N,D]
          struct_feat: [N,D_s] (共享)
          edge_index: [2,E] (共享)
          mutation_mask: [B,N] or [N]
        """
        if esm_wt.dim() == 3:
            # Batch 模式: 逐个样本编码
            B = esm_wt.size(0)
            z_list = []
            for i in range(B):
                h_wt = self.encode_single(esm_wt[i], struct_feat, edge_index)
                h_mut = self.encode_single(esm_mut[i], struct_feat, edge_index)
                z = self.delta_encoder(h_wt, h_mut, mutation_mask[i])
                z_list.append(z)
            z = torch.stack(z_list, dim=0)  # [B, hidden_dim]
        else:
            # 单样本模式
            h_wt = self.encode_single(esm_wt, struct_feat, edge_index)
            h_mut = self.encode_single(esm_mut, struct_feat, edge_index)
            z = self.delta_encoder(h_wt, h_mut, mutation_mask)

        mech_out = self.mech_module(z)

        return {
            'mutation_repr': z,
            'mechanism_outputs': mech_out['mechanism_outputs'],
            'mechanism_probs': mech_out['mechanism_probs'],
            'category_logits': mech_out['category_logits'],
            'dominant_category': mech_out['dominant_category'],
        }

    def compute_loss(self, preds, labels, weights):
        total = 0.0
        losses = {}

        probs = preds['mechanism_probs']
        targets = labels['multilabel']

        bce = F.binary_cross_entropy(probs, targets, reduction='mean')
        total += bce * weights.get('multilabel', 1.0)
        losses['multilabel'] = bce

        if 'dominant_category' in labels:
            ce = F.cross_entropy(preds['category_logits'], labels['dominant_category'])
            total += ce * weights.get('dominant', 0.5)
            losses['dominant'] = ce

        dir_loss = 0
        mag_loss = 0
        count = 0
        for i, m in enumerate(self.mech_module.mechanism_list):
            mask = targets[:, i] > 0.5
            if mask.any():
                out = preds['mechanism_outputs'][m]
                d = torch.tanh(out['effect_direction'])
                dir_loss += F.mse_loss(d[mask], labels['effect_direction'][:, i][mask].float())
                mag_loss += F.mse_loss(out['effect_magnitude'][mask], labels['effect_magnitude'][:, i][mask])
                count += 1

        if count > 0:
            total += (dir_loss / count) * weights.get('direction', 0.3)
            total += (mag_loss / count) * weights.get('magnitude', 0.2)
            losses['direction'] = dir_loss / count
            losses['magnitude'] = mag_loss / count

        if self.use_orthogonality:
            ortho = self.mech_module.compute_orthogonality_loss(probs)
            total += ortho * weights.get('ortho', 0.1)
            losses['ortho'] = ortho

        losses['total'] = total
        return losses
