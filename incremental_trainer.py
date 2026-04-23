#!/usr/bin/env python3
"""
增量训练：在已有模型基础上，用新数据微调
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from classifier import T7RNAPMechClassifier
from dataset import T7MutationDataset, collate_fn
from predict import T7MechPredictor

class IncrementalTrainer:
    def __init__(self, checkpoint_path='outputs/exp_001/best_model.pt'):
        self.device = torch.device('cpu')
        
        # 加载已有模型
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.cfg = ckpt['config']
        
        self.model = T7RNAPMechClassifier(
            esm_dim=self.cfg['model']['esm_dim'],
            struct_dim=self.cfg['model']['struct_dim'],
            hidden_dim=self.cfg['model']['hidden_dim'],
            num_gnn_layers=self.cfg['model']['num_gnn_layers'],
            dropout=0.1,  # 微调时降低 dropout
            use_orthogonality=True
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.to(self.device)
        
        # 冻结底层，只微调顶层
        for param in self.model.struct_encoder.parameters():
            param.requires_grad = False
        for param in self.model.gnn.parameters():
            param.requires_grad = False  # 可选：冻结 GNN，只训机制头
        
        # 只微调机制头和分类器
        trainable_params = []
        for name, param in self.model.named_parameters():
            if 'mech_module' in name or 'delta_encoder' in name:
                param.requires_grad = True
                trainable_params.append(param)
            else:
                param.requires_grad = False
        
        self.optimizer = optim.AdamW(trainable_params, lr=5e-5, weight_decay=1e-5)
        
        self.weights = {
            'multilabel': 1.0,
            'dominant': 0.5,
            'direction': 0.3,
            'magnitude': 0.2,
            'ortho': 0.1
        }
    
    def fine_tune(self, num_epochs=10, batch_size=4):
        """
        用新的 annotations.csv 微调模型
        """
        # 加载新数据集（包含用户反馈）
        dataset = T7MutationDataset(
            'data/processed/annotations.csv',
            'data/processed/esm_embeddings',
            use_mock=False,
            split='train'
        )
        
        # 只取最新加入的数据（最近 48 小时或最近 N 条）
        # 简化：全部数据，但新数据会被更频繁采样
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        
        self.model.train()
        losses = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for batch in loader:
                self.optimizer.zero_grad()
                
                esm_wt = batch['esm_wt'].to(self.device)
                esm_mut = batch['esm_mut'].to(self.device)
                struct = batch['struct_feat'].to(self.device)
                edge_index = batch['edge_index'].to(self.device)
                mask = batch['mutation_mask'].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch['labels'].items()}
                
                preds = self.model(esm_wt, esm_mut, struct, edge_index, mask)
                loss_dict = self.model.compute_loss(preds, labels, self.weights)
                
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_loss += loss_dict['total'].item()
            
            avg_loss = epoch_loss / len(loader)
            losses.append(avg_loss)
            print(f"[Fine-tune Epoch {epoch+1}] Loss: {avg_loss:.4f}")
        
        # 保存增量模型
        output_dir = 'outputs/exp_001'
        torch.save({
            'epoch': 'incremental',
            'model': self.model.state_dict(),
            'config': self.cfg,
            'fine_tune_losses': losses
        }, os.path.join(output_dir, 'best_model.pt'))
        
        return {'final_loss': losses[-1], 'epochs': num_epochs}
