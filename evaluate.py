#!/usr/bin/env python3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from classifier import T7RNAPMechClassifier
from dataset import T7MutationDataset, collate_fn
from mechanism_ontology import ALL_MECHANISMS, CATEGORY_NAMES


def evaluate(model, loader, device):
    model.eval()
    all_probs = []
    all_targets = []
    all_cat_preds = []
    all_cat_targets = []

    for batch in loader:
        esm_wt = batch['esm_wt'].to(device)
        esm_mut = batch['esm_mut'].to(device)
        struct = batch['struct_feat'].to(device)
        edge_index = batch['edge_index'].to(device)
        mask = batch['mutation_mask'].to(device)
        labels = batch['labels']

        with torch.no_grad():
            preds = model(esm_wt, esm_mut, struct, edge_index, mask)

        all_probs.append(preds['mechanism_probs'].cpu().numpy())
        all_targets.append(labels['multilabel'].cpu().numpy())
        all_cat_preds.append(preds['category_logits'].cpu().numpy())
        all_cat_targets.append(labels['dominant_category'].cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)  # [N, 15]
    targets = np.concatenate(all_targets, axis=0)  # [N, 15]
    cat_preds = np.concatenate(all_cat_preds, axis=0)
    cat_targets = np.concatenate(all_cat_targets, axis=0)

    # 每个机制的 Average Precision
    print("\n" + "=" * 60)
    print("Per-Mechanism Performance (Average Precision)")
    print("=" * 60)
    aps = []
    for i, mech in enumerate(ALL_MECHANISMS):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], probs[:, i])
            aps.append(ap)
            print(f"  {mech:<8} AP={ap:.3f}")

    print(f"\nMean Average Precision (mAP): {np.mean(aps):.4f}")

    # 主导大类准确率
    cat_acc = (cat_preds.argmax(axis=1) == cat_targets).mean()
    print(f"Dominant Category Accuracy: {cat_acc:.4f}")

    # 正交性检查
    corr = np.corrcoef(probs.T)
    mask = ~np.eye(len(corr), dtype=bool)
    mean_corr = np.abs(corr[mask]).mean()
    print(f"Mean Mechanism Correlation: {mean_corr:.4f} (lower=more disentangled)")
    print("=" * 60)


def main():
    checkpoint = torch.load('outputs/exp_001/best_model.pt', map_location='cpu')
    cfg = checkpoint['config']
    device = torch.device('cpu')

    test_ds = T7MutationDataset(
        cfg['data']['annotation_file'],
        cfg['data']['esm_dir'],
        use_mock=cfg['data'].get('use_mock_data', False),
        split='test'
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_fn)

    model = T7RNAPMechClassifier(
        esm_dim=cfg['model']['esm_dim'],
        struct_dim=cfg['model']['struct_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_gnn_layers=cfg['model']['num_gnn_layers'],
        dropout=0,
        use_orthogonality=False
    )
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    evaluate(model, test_loader, device)


if __name__ == '__main__':
    main()
