#!/usr/bin/env python3
import os

# Fix 1: Force working directory to script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# Fix 2: Flat imports (all files are in the same folder)
from classifier import T7RNAPMechClassifier
from dataset import T7MutationDataset, collate_fn


def load_config(path):
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    # Fix 3: Explicit UTF-8 encoding
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, device, weights):
    model.train()
    total_losses = {}

    for batch in tqdm(loader, desc="Train"):
        optimizer.zero_grad()

        esm_wt = batch['esm_wt'].to(device)
        esm_mut = batch['esm_mut'].to(device)
        struct = batch['struct_feat'].to(device)
        edge_index = batch['edge_index'].to(device)
        mask = batch['mutation_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        preds = model(esm_wt, esm_mut, struct, edge_index, mask)
        loss_dict = model.compute_loss(preds, labels, weights)

        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

    return {k: v / len(loader) for k, v in total_losses.items()}


@torch.no_grad()
def validate(model, loader, device, weights):
    model.eval()
    total_losses = {}

    for batch in loader:
        esm_wt = batch['esm_wt'].to(device)
        esm_mut = batch['esm_mut'].to(device)
        struct = batch['struct_feat'].to(device)
        edge_index = batch['edge_index'].to(device)
        mask = batch['mutation_mask'].to(device)
        labels = {k: v.to(device) for k, v in batch['labels'].items()}

        preds = model(esm_wt, esm_mut, struct, edge_index, mask)
        loss_dict = model.compute_loss(preds, labels, weights)

        for k, v in loss_dict.items():
            total_losses[k] = total_losses.get(k, 0) + v.item()

    return {k: v / len(loader) for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--output_dir', default='outputs/exp_001')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    cfg = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_ds = T7MutationDataset(
        cfg['data']['annotation_file'],
        cfg['data']['esm_dir'],
        use_mock=cfg['data'].get('use_mock_data', False),
        split='train'
    )
    val_ds = T7MutationDataset(
        cfg['data']['annotation_file'],
        cfg['data']['esm_dir'],
        use_mock=cfg['data'].get('use_mock_data', False),
        split='val'
    )

    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg['training']['batch_size'],
                            shuffle=False, collate_fn=collate_fn)

    # Model
    model = T7RNAPMechClassifier(
        esm_dim=cfg['model']['esm_dim'],
        struct_dim=cfg['model']['struct_dim'],
        hidden_dim=cfg['model']['hidden_dim'],
        num_gnn_layers=cfg['model']['num_gnn_layers'],
        dropout=cfg['model']['dropout'],
        use_orthogonality=cfg['model']['use_orthogonality']
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg['training']['lr'],
                           weight_decay=cfg['training']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['training']['num_epochs'])

    weights = {
        'multilabel': cfg['training']['loss_multilabel_weight'],
        'dominant': cfg['training']['loss_dominant_weight'],
        'direction': cfg['training']['loss_direction_weight'],
        'magnitude': cfg['training']['loss_magnitude_weight'],
        'ortho': cfg['training']['loss_ortho_weight']
    }

    best = float('inf')
    patience = 10  # 新增：早停耐心值
    patience_counter = 0  # 新增

    for epoch in range(1, cfg['training']['num_epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device, weights)
        val_metrics = validate(model, val_loader, device, weights)

        print(f"[Epoch {epoch:02d}] Train: {train_metrics['total']:.4f} | "
              f"Val: {val_metrics['total']:.4f} | "
              f"(multilabel={val_metrics['multilabel']:.3f}, ortho={val_metrics.get('ortho', 0):.4f})")

        # 早停逻辑：只关注 multilabel loss
        current_val = val_metrics['multilabel']
        if current_val < best:
            best = current_val
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'config': cfg
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  -> Saved best (val_multilabel={best:.4f})")
        else:
            patience_counter += 1
            print(f"  -> No improve ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        scheduler.step()

    print(f"Done. Best model saved to {args.output_dir}/best_model.pt")


if __name__ == '__main__':
    main()

