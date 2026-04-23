#!/usr/bin/env python3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from classifier import T7RNAPMechClassifier
from mechanism_ontology import (MECHANISM_ONTOLOGY, ALL_MECHANISMS,
                                CATEGORY_MAP, CATEGORY_NAMES, detect_conflicts)
from structure_utils import MockStructureProcessor

# ================== 轻量 ESM 编码器（和训练完全一致）====================
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

PROJ_PATH = 'data/processed/esm_embeddings/projection_matrix.pt'
if os.path.exists(PROJ_PATH):
    PROJ_MATRIX = torch.load(PROJ_PATH)
else:
    torch.manual_seed(42)
    PROJ_MATRIX = torch.nn.init.orthogonal_(torch.randn(20, 1280))
    os.makedirs('data/processed/esm_embeddings', exist_ok=True)
    torch.save(PROJ_MATRIX, PROJ_PATH)


def encode_sequence(seq):
    indices = [AA_TO_IDX.get(aa, 0) for aa in seq]
    onehot = torch.zeros(len(seq), 20)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0
    return onehot @ PROJ_MATRIX


T7_WT_SEQ = (
    "MNTINIAKNDFSDIELAAIPFNTLADHYGERLAREQLALEHESYEMGEARFRKMFERQLK"
    "AGEVADNAAAKPLITTLLPKMIARINDWFEEVKAKRGKRPTAFQFLQEIKPEAVAYITIK"
    "TTLACLTSADNTTVQAVASAIGRAIEDEARFGRIRDLEAKHFKKNVEEQLNKRVGHVYKK"
    "AFMQVVEADMLSKGLLGGEAWSSWHKEDSIHVGVRCIEMLIESTGMVSLHRQNAGVVGQD"
    "SETIELAPEYAEAIATRAGALAGISPMFQPCVVPPKPWTGITGGGYWANGRRPLALVRTH"
    "SKKALMRYEDVYMPEVYKAINIAQNTAWKINKKVLAVANVITKWKHCPVEDIPAIEREEL"
    "PMKPEDIDMNPEALTAWKRAAAAVYRKDKARKSRRISLEFMLEQANKFANHKAIWFPYNM"
    "DWRGRVYAVSMFNPQGNDMTKGLLTLAKGKPIGKEGYYWLKIHGANCAGVDKVPFPERIK"
    "FIEENHENIMACAKSPLENTWWAEQDSPFCFLAFCFEYAGVQHHGLSYNCSLPLAFDGSC"
    "SGIQHFSAMLRDEVGGRAVNLLPSETVQDIYGIVAKKVNEILQADAINGTDNEVVTVTDE"
    "NTGEISEKVKLGTKALAGQWLAYGVTRSVTKRSVMTLAYGSKEFGFRQQVLEDTIQPAID"
    "SGKGLMFTQPNQAAGYMAKLIWESVSVTVVAAVEAMNWLKSAAKLLAAEVKDKKTGEILR"
    "KRCAVHWVTPDGFPVWQEYKKPIQTRLNLMFLGQFRLQPTINTNKDSEIDAHKQESGIAP"
    "NFVHSQDGSHLRKTVVWAHEKYGIESFALIHDSFGTIPADAANLFKAVRETMVDTYESCD"
    "VLADFYDQFADQLHESQLDKMPALPAKGNLNLRDILESDFAFA"
)


class T7MechPredictor:
    def __init__(self, checkpoint_path='outputs/exp_001/best_model.pt'):
        self.device = torch.device('cpu')
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = ckpt['config']

        self.model = T7RNAPMechClassifier(
            esm_dim=self.cfg['model']['esm_dim'],
            struct_dim=self.cfg['model']['struct_dim'],
            hidden_dim=self.cfg['model']['hidden_dim'],
            num_gnn_layers=self.cfg['model']['num_gnn_layers'],
            dropout=0,
            use_orthogonality=False
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()

        self.processor = MockStructureProcessor(length=883)
        self.edge_index, _ = self.processor.build_graph()
        self.struct_feat = self.processor.extract_node_features()

    def predict(self, site, wt_aa, mut_aa, prob_threshold=0.3):
        idx = site - 1

        actual_wt = T7_WT_SEQ[idx]
        if actual_wt != wt_aa:
            print(f"WARNING: WT mismatch at {site}: expected {actual_wt}, got {wt_aa}")
            return None

        esm_wt = encode_sequence(T7_WT_SEQ)
        mut_seq = T7_WT_SEQ[:idx] + mut_aa + T7_WT_SEQ[idx + 1:]
        esm_mut = encode_sequence(mut_seq)
        mask = self.processor.get_mutation_context(idx)

        with torch.no_grad():
            preds = self.model(
                esm_wt.unsqueeze(0),
                esm_mut.unsqueeze(0),
                self.struct_feat,
                self.edge_index,
                mask.unsqueeze(0)
            )

        probs = preds['mechanism_probs'].squeeze().numpy()
        dominant_cat = CATEGORY_NAMES[preds['dominant_category'].item()]

        # 收集激活机制（threshold 默认 0.3，更宽松）
        activated = []
        for i, mech in enumerate(ALL_MECHANISMS):
            if probs[i] > prob_threshold:
                out = preds['mechanism_outputs'][mech]
                activated.append({
                    'id': mech,
                    'name': MECHANISM_ONTOLOGY[mech][0],
                    'category': MECHANISM_ONTOLOGY[mech][1],
                    'prob': float(probs[i]),
                    'direction': 'positive' if torch.tanh(out['effect_direction']).item() > 0.3 else 'negative',
                    'magnitude': float(out['effect_magnitude'].item())
                })

        # 如果没有超过 threshold 的，取 Top-3
        if not activated:
            top_indices = np.argsort(probs)[-3:][::-1]
            for i in top_indices:
                if probs[i] > 0.05:  # 至少 >0.05
                    mech = ALL_MECHANISMS[i]
                    out = preds['mechanism_outputs'][mech]
                    activated.append({
                        'id': mech,
                        'name': MECHANISM_ONTOLOGY[mech][0],
                        'category': MECHANISM_ONTOLOGY[mech][1],
                        'prob': float(probs[i]),
                        'direction': 'positive' if torch.tanh(out['effect_direction']).item() > 0.3 else 'negative',
                        'magnitude': float(out['effect_magnitude'].item())
                    })

        dom_mech = max(activated, key=lambda x: x['prob'])['id'] if activated else 'unknown'

        cat_scores = {cat: 0.0 for cat in ['stability', 'activity', 'promoter', 'quality', 'allostery']}
        for a in activated:
            cat_scores[a['category']] += a['prob']

        return {
            'mutation': f"{wt_aa}{site}{mut_aa}",
            'dominant_category': dominant_cat,
            'dominant_mechanism': dom_mech,
            'activated_mechanisms': activated,
            'category_scores': cat_scores,
            'representation': preds['mutation_repr'].squeeze().numpy(),
            'conflicts': detect_conflicts([a['id'] for a in activated])
        }

    def print_report(self, report):
        if report is None:
            print("Invalid mutation")
            return

        print(f"\n{'=' * 60}")
        print(f"MECHANISM REPORT: {report['mutation']}")
        print(f"{'=' * 60}")
        print(f"Dominant: {report['dominant_mechanism']} ({report['dominant_category']})")

        print(f"\nActivated Mechanisms (Top):")
        for m in sorted(report['activated_mechanisms'], key=lambda x: -x['prob']):
            print(f"  • {m['id']} {m['name']:<22} | {m['category']:<10} | "
                  f"P={m['prob']:.2f} | {m['direction']:<8} | Mag={m['magnitude']:.2f}")

        print(f"\nCategory Scores:")
        for cat, score in sorted(report['category_scores'].items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"  {cat:<12} {bar} {score:.2f}")

        if report['conflicts']:
            print(f"\n[!] CONFLICTS:")
            for c in report['conflicts']:
                print(f"    ⚠ {c[2]}")
        else:
            print(f"\n[✓] No conflicts")
        print(f"{'=' * 60}")


def demo():
    predictor = T7MechPredictor()

    test_mutations = [
        (43, 'S', 'E'),  # 应激活 1.1.1
        (633, 'S', 'P'),  # 应激活 1.1.1
        (786, 'Q', 'M'),  # 应激活 1.1.1
        (639, 'Y', 'F'),  # 应激活 1.4.3
        (631, 'K', 'A'),  # 应激活 1.4.3
    ]

    reports = []
    for site, wt, mut in test_mutations:
        r = predictor.predict(site, wt, mut, prob_threshold=0.3)
        if r:
            predictor.print_report(r)
            reports.append(r)

    return reports


if __name__ == '__main__':
    demo()


