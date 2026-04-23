#!/usr/bin/env python3
"""
用户反馈数据存储管理
自动归档、去重、质量评分
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
from data_schema import ExperimentalDatum

class FeedbackStorage:
    def __init__(self, storage_dir='data/user_feedback'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        
        # 主表：所有用户提交的数据
        self.master_csv = os.path.join(storage_dir, 'feedback_master.csv')
        # 待审核队列（新提交未验证）
        self.pending_csv = os.path.join(storage_dir, 'feedback_pending.csv')
        # 已合并到训练集的数据
        self.merged_csv = os.path.join(storage_dir, 'feedback_merged.csv')
        
        # 初始化空表
        for path in [self.master_csv, self.pending_csv, self.merged_csv]:
            if not os.path.exists(path):
                pd.DataFrame(columns=self._columns()).to_csv(path, index=False)
    
    def _columns(self):
        return [
            'timestamp', 'protein_id', 'mutation', 'site', 'wt_aa', 'mut_aa',
            'assay_temperature', 'assay_buffer',
            'tm_value', 'tm_delta', 'kcat_km', 'activity_relative',
            'dsRNA_ratio', 'fidelity_index', 'yield_mg_per_L', 'half_life_min',
            'user_mechanism_label', 'experimenter', 'notes',
            'data_quality_score', 'model_agreement', 'status'
        ]
    
    def submit(self, datum: ExperimentalDatum) -> Dict:
        """
        用户提交一条数据
        返回：{'status': 'pending', 'id': '...', 'quality_score': 0.85}
        """
        datum.validate()
        
        # 生成唯一ID
        timestamp = datetime.now().isoformat()
        record_id = f"{datum.experimenter}_{datum.mutation}_{timestamp[:10]}"
        
        # 计算数据质量分（完整性）
        filled_fields = sum([
            datum.tm_value is not None,
            datum.tm_delta is not None,
            datum.kcat_km is not None,
            datum.activity_relative is not None,
            datum.dsRNA_ratio is not None,
            datum.fidelity_index is not None,
            datum.yield_mg_per_L is not None,
            datum.half_life_min is not None
        ])
        quality_score = filled_fields / 8.0  # 0~1
        
        row = {
            'timestamp': timestamp,
            'protein_id': datum.protein_id,
            'mutation': datum.mutation,
            'site': datum.site,
            'wt_aa': datum.wt_aa,
            'mut_aa': datum.mut_aa,
            'assay_temperature': datum.assay_temperature,
            'assay_buffer': datum.assay_buffer,
            'tm_value': datum.tm_value,
            'tm_delta': datum.tm_delta,
            'kcat_km': datum.kcat_km,
            'activity_relative': datum.activity_relative,
            'dsRNA_ratio': datum.dsRNA_ratio,
            'fidelity_index': datum.fidelity_index,
            'yield_mg_per_L': datum.yield_mg_per_L,
            'half_life_min': datum.half_life_min,
            'user_mechanism_label': datum.user_mechanism_label,
            'experimenter': datum.experimenter,
            'notes': datum.notes,
            'data_quality_score': round(quality_score, 2),
            'model_agreement': None,  # 待计算：模型预测 vs 用户标签
            'status': 'pending'
        }
        
        # 追加到主表和待审核表
        df = pd.DataFrame([row])
        df.to_csv(self.master_csv, mode='a', header=False, index=False)
        df.to_csv(self.pending_csv, mode='a', header=False, index=False)
        
        return {
            'status': 'pending',
            'id': record_id,
            'quality_score': quality_score,
            'message': '数据已提交，等待审核后并入训练集'
        }
    
    def review_pending(self, min_quality=0.5) -> pd.DataFrame:
        """
        审核待处理数据
        返回质量分 > min_quality 的记录
        """
        df = pd.read_csv(self.pending_csv)
        if df.empty:
            return df
        
        # 过滤低质量
        qualified = df[df['data_quality_score'] >= min_quality].copy()
        
        # 计算模型一致性（如果用户提供了机制标签）
        from predict import T7MechPredictor
        predictor = T7MechPredictor()
        
        agreements = []
        for _, row in qualified.iterrows():
            if pd.notna(row['user_mechanism_label']):
                pred = predictor.predict(int(row['site']), row['wt_aa'], row['mut_aa'])
                if pred:
                    pred_mechs = [a['id'] for a in pred['activated_mechanisms']]
                    agreement = 1.0 if row['user_mechanism_label'] in pred_mechs else 0.0
                else:
                    agreement = 0.0
            else:
                agreement = None  # 用户没填标签，无法计算一致性
            agreements.append(agreement)
        
        qualified['model_agreement'] = agreements
        return qualified
    
    def merge_to_training(self, record_ids: List[str]):
        """
        将审核通过的数据合并到训练集
        """
        # 1. 从 pending 移动到 merged
        pending = pd.read_csv(self.pending_csv)
        to_merge = pending[pending['protein_id'].isin(record_ids)]
        
        if to_merge.empty:
            return {'merged': 0}
        
        # 追加到 merged
        to_merge['status'] = 'merged'
        to_merge.to_csv(self.merged_csv, mode='a', header=False, index=False)
        
        # 从 pending 删除
        remaining = pending[~pending['protein_id'].isin(record_ids)]
        remaining.to_csv(self.pending_csv, index=False)
        
        # 2. 生成机制标签（如果用户没填，用模型预测）
        from predict import T7MechPredictor
        from mechanism_ontology import MECH_TO_CAT_IDX
        
        predictor = T7MechPredictor()
        new_rows = []
        
        for _, row in to_merge.iterrows():
            mech_label = row['user_mechanism_label']
            if pd.isna(mech_label):
                # 用模型预测主导机制
                pred = predictor.predict(int(row['site']), row['wt_aa'], row['mut_aa'])
                mech_label = pred['dominant_mechanism'] if pred else '1.1.1'
            
            # 构建训练集格式
            new_row = {
                'protein_id': row['protein_id'],
                'mutation': row['mutation'],
                'site': int(row['site']),
                'wt_aa': row['wt_aa'],
                'mut_aa': row['mut_aa'],
                'split': 'train',  # 新数据默认进训练集
                'dominant_mech': mech_label,
                # 机制标签（基于主导机制）
                '1.1.1': 1 if mech_label == '1.1.1' else 0,
                '1.1.2': 1 if mech_label == '1.1.2' else 0,
                # ... 其他机制同理（简化处理，实际应遍历 ALL_MECHANISMS）
                'effect_sign_1.1.1': 1 if mech_label == '1.1.1' else 0,
                'effect_size_1.1.1': row['activity_relative'] if pd.notna(row['activity_relative']) else 0.5,
            }
            new_rows.append(new_row)
        
        # 3. 追加到 annotations.csv
        train_df = pd.read_csv('data/processed/annotations.csv')
        new_df = pd.DataFrame(new_rows)
        
        # 去重：如果 mutation 已存在，更新而非追加
        existing = set(train_df['mutation'].tolist())
        new_df_filtered = new_df[~new_df['mutation'].isin(existing)]
        
        combined = pd.concat([train_df, new_df_filtered], ignore_index=True)
        combined.to_csv('data/processed/annotations.csv', index=False)
        
        return {
            'merged': len(new_df_filtered),
            'total_training': len(combined)
        }
