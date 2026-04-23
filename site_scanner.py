#!/usr/bin/env python3
"""
位点扫描器：输入一个位点，预测最优突变及机制
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from predict import T7MechPredictor, T7_WT_SEQ

# 20 种氨基酸
AA_LIST = list('ACDEFGHIKLMNPQRSTVWY')


class SiteScanner:
    def __init__(self, predictor=None):
        self.predictor = predictor or T7MechPredictor()

    def scan_site(self, site):
        """
        扫描单个位点的所有可能突变
        返回: 按机制激活概率排序的突变建议列表
        """
        idx = site - 1
        wt_aa = T7_WT_SEQ[idx]

        print(f"\n{'=' * 60}")
        print(f"SITE SCANNER: Position {site} (WT={wt_aa})")
        print(f"{'=' * 60}")

        results = []

        for mut_aa in AA_LIST:
            if mut_aa == wt_aa:
                continue  # 跳过同义突变

            report = self.predictor.predict(site, wt_aa, mut_aa, prob_threshold=0.2)
            if report is None:
                continue

            # 计算综合得分
            top_mech = report['activated_mechanisms'][0] if report['activated_mechanisms'] else None

            if top_mech:
                results.append({
                    'mutation': report['mutation'],
                    'mechanism': top_mech['id'],
                    'mechanism_name': top_mech['name'],
                    'category': top_mech['category'],
                    'probability': top_mech['prob'],
                    'direction': top_mech['direction'],
                    'magnitude': top_mech['magnitude'],
                    'num_activated': len(report['activated_mechanisms']),
                    'category_scores': report['category_scores'],
                    'representation': report['representation']
                })

        # 按机制概率排序
        results.sort(key=lambda x: x['probability'], reverse=True)

        return {
            'site': site,
            'wt_aa': wt_aa,
            'candidates': results
        }

    def recommend_mutations(self, site, target_category=None, top_k=5):
        """
        推荐突变
        target_category: 'stability'/'activity'/'quality'/'allostery'/'promoter'
        """
        scan = self.scan_site(site)
        candidates = scan['candidates']

        if target_category:
            # 过滤目标大类
            filtered = [c for c in candidates if c['category'] == target_category]
            if not filtered:
                print(f"WARNING: No {target_category} mechanism found at site {site}")
                filtered = candidates
            candidates = filtered

        return {
            'site': site,
            'wt_aa': scan['wt_aa'],
            'recommendations': candidates[:top_k]
        }

    def print_scan_report(self, site, target_category=None):
        """打印人类可读的扫描报告"""
        rec = self.recommend_mutations(site, target_category)

        print(f"\nTop Recommendations for Site {rec['site']} (WT={rec['wt_aa']})")
        if target_category:
            print(f"Target: {target_category}")
        print("-" * 60)

        for i, cand in enumerate(rec['recommendations'], 1):
            print(f"{i}. {cand['mutation']:<8} → {cand['mechanism_name']:<22} "
                  f"({cand['category']}) | P={cand['probability']:.2f} | "
                  f"Effect: {cand['direction']:<8} | Mag={cand['magnitude']:.2f}")

        return rec


if __name__ == '__main__':
    scanner = SiteScanner()

    # 示例：扫描位点 43（已知 S43E 是 cavity filling）
    scanner.print_scan_report(43, target_category='stability')

    # 示例：扫描位点 639（已知 Y639F 是 fidelity）
    scanner.print_scan_report(639, target_category='quality')
