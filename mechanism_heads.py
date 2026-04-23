import torch
import torch.nn as nn
import torch.nn.functional as F

from mechanism_ontology import ALL_MECHANISMS, CATEGORY_MAP, CATEGORY_NAMES


class MechanismExpert(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64):   # 128->64
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.3),    # 新增dropout
            nn.Linear(hidden_dim, 3)
        )

    def forward(self, z):
        out = self.net(z)
        return {
            'presence_logit': out[:, 0],
            'effect_magnitude': out[:, 1],
            'effect_direction': out[:, 2]
        }


class MechanismDisentanglementModule(nn.Module):
    def __init__(self, mechanism_list, input_dim=256):
        super().__init__()
        self.mechanism_list = mechanism_list

        # 关键修复：把 "1.1.1" -> "1_1_1" 作为 ModuleDict key
        self._key_map = {m: m.replace('.', '_') for m in mechanism_list}
        self._reverse_map = {v: k for k, v in self._key_map.items()}

        self.experts = nn.ModuleDict({
            self._key_map[m]: MechanismExpert(input_dim) for m in mechanism_list
        })
        self.category_classifier = nn.Linear(input_dim, 5)

    def forward(self, z):
        outputs = {}
        probs = []
        for m in self.mechanism_list:
            key = self._key_map[m]
            out = self.experts[key](z)
            outputs[m] = out  # 对外仍用原始 ID "1.1.1"
            probs.append(torch.sigmoid(out['presence_logit']))

        probs = torch.stack(probs, dim=1)

        cat_logits = []
        for cat_name, mechs in CATEGORY_MAP.items():
            idx = [self.mechanism_list.index(m) for m in mechs if m in self.mechanism_list]
            if idx:
                cat_logits.append(probs[:, idx].sum(dim=1, keepdim=True))
        cat_logits = torch.cat(cat_logits, dim=1)

        dominant = torch.argmax(cat_logits, dim=1)

        return {
            'mechanism_outputs': outputs,
            'mechanism_probs': probs,
            'category_logits': cat_logits,
            'dominant_category': dominant,
        }

    def compute_orthogonality_loss(self, mechanism_probs):
        mp = mechanism_probs
        mp_centered = mp - mp.mean(dim=0, keepdim=True)
        cov = torch.mm(mp_centered.t(), mp_centered) / mp.size(0)
        mask = ~torch.eye(cov.size(0), dtype=torch.bool, device=cov.device)
        return cov[mask].pow(2).mean()
