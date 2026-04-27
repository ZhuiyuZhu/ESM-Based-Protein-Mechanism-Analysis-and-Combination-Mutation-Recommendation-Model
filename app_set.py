#!/usr/bin/env python3
"""
T7 RNAP 机制分类器 - Streamlit V1.0
核心升级:
1. 模型: 轻量ESM -> ESM2-650M (1280维)
2. 数据: 73条 -> 238条文献验证突变
3. 单突变解析: 机制协同度评分 + V1.0加权适应度
4. 组合设计: 基于1MSW真实结构的4象限上位效应评估
5. 实验反馈: 突出dsRNA/quality数据收集
6. 权重: 热稳35% + dsRNA35% + 产率20% + 其他10%
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import torch

# ================== 路径处理 ==================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)

# ================== 模型路径探测 ==================
def find_model_path():
    candidates = [
        'outputs/exp_001/best_model.pt',
        'outputs/exp_002/best_model.pt',
        'best_model.pt',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

MODEL_PATH = find_model_path()

# ================== 导入项目模块 ==================
try:
    from predict import T7MechPredictor, T7_WT_SEQ
    from mechanism_ontology import MECHANISM_ONTOLOGY, ALL_MECHANISMS, CATEGORY_NAMES
    from combinatorial import CombinatorialOptimizer, MutantReport
    import streamlit as st
except ImportError as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# ================== 1MSW 结构坐标加载 ==================
def load_1msw_coords(cif_path='data/structures/1MSW.cif'):
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure('T7', cif_path)
        best_chain = None
        max_ca = 0
        for model in structure:
            for chain in model:
                ca_count = sum(1 for r in chain if 'CA' in r)
                if ca_count > max_ca and ca_count > 500:
                    max_ca = ca_count
                    best_chain = chain
        if best_chain is None:
            return None
        coords = np.zeros((883, 3), dtype=np.float32)
        for residue in best_chain:
            res_id = residue.get_id()[1]
            if res_id and 1 <= res_id <= 883 and 'CA' in residue:
                coords[res_id - 1] = residue['CA'].get_coord()
        for i in range(883):
            if np.all(coords[i] == 0):
                valid = np.where(np.any(coords != 0, axis=1))[0]
                if len(valid) > 0:
                    nearest = valid[np.argmin(np.abs(valid - i))]
                    coords[i] = coords[nearest]
        return coords
    except Exception as e:
        print(f"[1MSW加载失败] {e}")
        return None

MSW_COORDS = load_1msw_coords()

# ================== V1.0 权重体系 ==================
MECH_WEIGHTS = {
    '1.1.1': 0.12, '1.1.2': 0.08, '1.1.3': 0.10, '1.1.4': 0.03, '1.1.5': 0.02,
    '1.2.1': 0.10, '1.2.2': 0.04, '1.2.3': 0.04, '1.2.4': 0.01, '1.2.5': 0.01,
    '1.3.1': 0.02, '1.3.2': 0.01, '1.3.3': 0.02,
    '1.4.1': 0.15, '1.4.2': 0.05, '1.4.3': 0.12, '1.4.4': 0.03,
    '1.5.1': 0.03, '1.5.2': 0.02, '1.5.3': 0.00,
}

CATEGORY_WEIGHTS = {
    'stability': 0.35, 'activity': 0.20, 'promoter': 0.05,
    'quality': 0.35, 'allostery': 0.05,
}

# ================== 缓存加载模型 ==================
@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        raise FileNotFoundError("Model not found.")
    import predict
    original_init = predict.T7MechPredictor.__init__
    def patched_init(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = MODEL_PATH
        original_init(self, checkpoint_path)
    predict.T7MechPredictor.__init__ = patched_init
    return T7MechPredictor()

try:
    predictor = load_model()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.stop()

# ================== 核心辅助函数 ==================
def compute_structure_distance(site1: int, site2: int) -> float:
    if MSW_COORDS is not None:
        return float(np.linalg.norm(MSW_COORDS[site1-1] - MSW_COORDS[site2-1]))
    return abs(site1 - site2) * 3.5

def classify_epistasis(site1: int, site2: int, mech1: str, mech2: str):
    dist = compute_structure_distance(site1, site2)
    same_mech = (mech1 == mech2)
    is_close = dist < 20.0
    if is_close and same_mech:
        return "上位效应风险", "⚠️ 近距离+同机制: 强烈上位效应，不建议叠加", 0.9
    elif not is_close and same_mech:
        return "叠加候选", "✅ 远距离+同机制: 可能独立贡献，可尝试叠加", 0.3
    elif is_close and not same_mech:
        return "谨慎突变", "⚠️ 近距离+不同机制: 结构干扰风险，建议先做双突变验证", 0.7
    else:
        return "多功能推荐", "🌟 远距离+不同机制: 理想多功能酶，强烈推荐", 0.1

def compute_weighted_fitness(report: Dict) -> float:
    score = 0.0
    for m in report.get('activated_mechanisms', []):
        w = MECH_WEIGHTS.get(m['id'], 0.01)
        mag = m['magnitude']
        prob = m['prob']
        if m['id'] in ['1.4.1', '1.4.3', '1.4.4', '1.4.2']:
            score += w * prob * mag
        elif m['direction'] in ['positive', 'enhance', 'increase']:
            score += w * prob * mag
        else:
            score -= w * prob * mag * 0.5
    cat_scores = report.get('category_scores', {})
    for cat, w in CATEGORY_WEIGHTS.items():
        score += cat_scores.get(cat, 0) * w * 0.5
    return round(score, 3)

def scan_site(site: int, target_category=None, top_k: int = 10):
    idx = site - 1
    wt_aa = T7_WT_SEQ[idx]
    results = []
    for mut_aa in 'ACDEFGHIKLMNPQRSTVWY':
        if mut_aa == wt_aa:
            continue
        report = predictor.predict(site, wt_aa, mut_aa, prob_threshold=0.15)
        if report is None or not report.get('activated_mechanisms'):
            continue
        fitness = compute_weighted_fitness(report)
        pos_mechs = [m for m in report['activated_mechanisms']
                     if m['direction'] in ['positive', 'enhance', 'increase'] or m['id'] in ['1.4.1', '1.4.3']]
        has_stab = any(m['id'].startswith('1.1') for m in pos_mechs)
        has_dsRNA = any(m['id'] in ['1.4.1', '1.4.3', '1.5.2'] for m in pos_mechs)
        top = max(report['activated_mechanisms'], key=lambda x: x['prob'])
        results.append({
            'mutation': report['mutation'], 'mechanism': top['id'],
            'mechanism_name': top['name'], 'category': top['category'],
            'probability': float(top['prob']), 'direction': top['direction'],
            'magnitude': float(top['magnitude']), 'fitness': fitness,
            'synergy_count': len(pos_mechs), 'is_multifunction': has_stab and has_dsRNA,
            'has_stability': has_stab, 'has_dsRNA': has_dsRNA,
            'category_scores': report['category_scores'],
        })
    results.sort(key=lambda x: (x['is_multifunction'], x['fitness'], x['synergy_count']), reverse=True)
    if target_category:
        filtered = [r for r in results if r['category'] == target_category]
        results = filtered if filtered else results
    return {'site': site, 'wt_aa': wt_aa, 'candidates': results[:top_k]}

def evaluate_combination_v1(mutation_list: List[str]):
    reports = []
    valid_mutations = []
    for m in mutation_list:
        m = m.strip()
        if len(m) < 3:
            continue
        wt, site_str, mut_aa = m[0], m[1:-1], m[-1]
        try:
            site = int(site_str)
        except:
            continue
        r = predictor.predict(site, wt, mut_aa, prob_threshold=0.15)
        if r is None:
            continue
        r['fitness'] = compute_weighted_fitness(r)
        r['site'] = site
        reports.append(r)
        valid_mutations.append(m)
    if len(reports) < 2:
        return None
    
    # 协同度
    sites = [r['site'] for r in reports]
    mechs = [r['dominant_mechanism'] for r in reports]
    dists = [compute_structure_distance(sites[i], sites[j]) for i in range(len(sites)) for j in range(i+1, len(sites))]
    avg_dist = np.mean(dists) if dists else 0
    unique_mechs = len(set(mechs))
    diversity_score = unique_mechs / len(mechs) if mechs else 0
    conflict_penalty = sum(classify_epistasis(sites[i], sites[j], mechs[i], mechs[j])[2] 
                          for i in range(len(sites)) for j in range(i+1, len(sites)))
    conflict_penalty /= (len(reports) * (len(reports)-1) / 2 + 1e-6)
    synergy = (min(1.0, avg_dist/50.0) * 0.3 + diversity_score * 0.4 + (1-conflict_penalty) * 0.3)
    
    # 两两分析
    pair_analysis = []
    for i in range(len(reports)):
        for j in range(i+1, len(reports)):
            label, advice, risk = classify_epistasis(sites[i], sites[j], mechs[i], mechs[j])
            pair_analysis.append({
                'pair': f"{valid_mutations[i]} + {valid_mutations[j]}",
                'distance': round(compute_structure_distance(sites[i], sites[j]), 1),
                'mech_i': mechs[i], 'mech_j': mechs[j],
                'label': label, 'advice': advice, 'risk': risk
            })
    
    cat_scores = {'stability': 0, 'activity': 0, 'promoter': 0, 'quality': 0, 'allostery': 0}
    activated_mechs = set()
    for r in reports:
        for m in r.get('activated_mechanisms', []):
            activated_mechs.add(m['id'])
        for cat, score in r.get('category_scores', {}).items():
            cat_scores[cat] = max(cat_scores[cat], score)
    
    high_risk = [p for p in pair_analysis if p['risk'] > 0.6]
    return {
        'mutations': valid_mutations, 'sites': sites,
        'fitness': round(sum(r.get('fitness', 0) for r in reports), 2),
        'synergy': {'score': round(synergy, 3), 'label': "🌟极佳" if synergy > 0.75 else "✅良好" if synergy > 0.5 else "⚠️一般" if synergy > 0.3 else "❌差",
                   'avg_distance': round(avg_dist, 1), 'diversity_score': round(diversity_score, 3), 'conflict_penalty': round(conflict_penalty, 3)},
        'pair_analysis': pair_analysis, 'category_scores': cat_scores,
        'activated': list(activated_mechs), 'conflicts': len(high_risk), 'high_risk_pairs': high_risk
    }

# ================== 主动学习 (V1.0增强) ==================
@dataclass
class ExperimentalDatum:
    protein_id: str; mutation: str; site: int; wt_aa: str; mut_aa: str
    assay_temperature: float = 37.0; assay_buffer: str = "Tris-HCl pH 7.9"
    tm_value: Optional[float] = None; tm_delta: Optional[float] = None
    dsRNA_ratio: Optional[float] = None; yield_mg_per_L: Optional[float] = None
    activity_relative: Optional[float] = None; kcat_km: Optional[float] = None
    fidelity_index: Optional[float] = None; half_life_min: Optional[float] = None
    cap_efficiency: Optional[float] = None; full_length_ratio: Optional[float] = None
    user_mechanism_label: Optional[str] = None; experimenter: str = "anonymous"; notes: str = ""
    
    def validate(self):
        assert 1 <= self.site <= 883
        assert len(self.wt_aa) == 1 and len(self.mut_aa) == 1
        assert self.wt_aa != self.mut_aa
        has_core = any([self.tm_value, self.tm_delta, self.dsRNA_ratio, self.yield_mg_per_L])
        has_sec = any([self.activity_relative, self.kcat_km, self.fidelity_index, self.half_life_min, self.cap_efficiency, self.full_length_ratio])
        assert has_core or has_sec, "至少填写一项（建议优先dsRNA、Tm、产量）"
        return True
    
    def to_dict(self):
        return {
            'timestamp': datetime.now().isoformat(), 'protein_id': self.protein_id,
            'mutation': self.mutation, 'site': self.site, 'wt_aa': self.wt_aa, 'mut_aa': self.mut_aa,
            'assay_temperature': self.assay_temperature, 'assay_buffer': self.assay_buffer,
            'tm_value': self.tm_value, 'tm_delta': self.tm_delta,
            'dsRNA_ratio': self.dsRNA_ratio, 'yield_mg_per_L': self.yield_mg_per_L,
            'activity_relative': self.activity_relative, 'kcat_km': self.kcat_km,
            'fidelity_index': self.fidelity_index, 'half_life_min': self.half_life_min,
            'cap_efficiency': self.cap_efficiency, 'full_length_ratio': self.full_length_ratio,
            'user_mechanism_label': self.user_mechanism_label, 'experimenter': self.experimenter,
            'notes': self.notes, 'data_quality_score': 0.0, 'model_agreement': None, 'status': 'pending'
        }

class FeedbackStorage:
    def __init__(self, storage_dir='data/user_feedback'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.pending_csv = os.path.join(storage_dir, 'feedback_pending.csv')
        if not os.path.exists(self.pending_csv):
            cols = ['timestamp', 'protein_id', 'mutation', 'site', 'wt_aa', 'mut_aa',
                   'assay_temperature', 'assay_buffer', 'tm_value', 'tm_delta',
                   'dsRNA_ratio', 'yield_mg_per_L', 'activity_relative', 'kcat_km',
                   'fidelity_index', 'half_life_min', 'cap_efficiency', 'full_length_ratio',
                   'user_mechanism_label', 'experimenter', 'notes',
                   'data_quality_score', 'model_agreement', 'status']
            pd.DataFrame(columns=cols).to_csv(self.pending_csv, index=False)
    
    def submit(self, datum: ExperimentalDatum):
        datum.validate()
        row = datum.to_dict()
        core_filled = sum([row['tm_value'] is not None, row['tm_delta'] is not None, row['dsRNA_ratio'] is not None, row['yield_mg_per_L'] is not None])
        sec_filled = sum([row['activity_relative'] is not None, row['kcat_km'] is not None, row['fidelity_index'] is not None, row['half_life_min'] is not None, row['cap_efficiency'] is not None, row['full_length_ratio'] is not None])
        row['data_quality_score'] = round((core_filled / 4.0) * 0.6 + (sec_filled / 6.0) * 0.4, 2)
        pd.DataFrame([row]).to_csv(self.pending_csv, mode='a', header=False, index=False)
        return {'status': 'pending', 'quality_score': row['data_quality_score'], 'message': '数据已提交'}
    
    def get_pending(self):
        return pd.read_csv(self.pending_csv) if os.path.exists(self.pending_csv) else pd.DataFrame()
    
    def merge_selected(self, protein_ids: List[str]):
        df = self.get_pending()
        to_merge = df[df['protein_id'].isin(protein_ids)]
        if to_merge.empty: return 0
        train_path = 'data/processed/annotations.csv'
        if not os.path.exists(train_path): return 0
        train_df = pd.read_csv(train_path)
        new_rows = []
        for _, row in to_merge.iterrows():
            mech = row['user_mechanism_label'] if pd.notna(row['user_mechanism_label']) else '1.1.1'
            new_row = {
                'protein_id': row['protein_id'], 'mutation': row['mutation'],
                'site': int(row['site']), 'wt_aa': row['wt_aa'], 'mut_aa': row['mut_aa'],
                'split': 'train', 'dominant_mech': mech,
            }
            for m in ALL_MECHANISMS:
                new_row[m] = 1 if m == mech else 0
                new_row[f'effect_sign_{m}'] = 1 if m == mech else 0
                new_row[f'effect_size_{m}'] = 0.5 if m == mech else 0.0
            new_rows.append(new_row)
        new_df = pd.DataFrame(new_rows)
        existing = set(train_df['mutation'].tolist())
        new_df = new_df[~new_df['mutation'].isin(existing)]
        pd.concat([train_df, new_df], ignore_index=True).to_csv(train_path, index=False)
        df[~df['protein_id'].isin(protein_ids)].to_csv(self.pending_csv, index=False)
        return len(new_df)

# ================== Streamlit 页面 ==================
st.set_page_config(page_title="T7 RNAP V1.0", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.sidebar.title("🔬 T7 RNAP Mech-Evo V1.0")
st.sidebar.markdown("机制感知定向进化平台  \nESM2-650M | 238条 | 1MSW结构")
page = st.sidebar.radio("导航", ["🏠 首页", "🔍 位点扫描", "🧬 突变解析", "🔗 组合设计", "📊 实验反馈"])

if page == "🏠 首页":
    st.title("T7 RNA 聚合酶机制分类器 V1.0")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### V1.0 核心升级
        | 升级项 | V0.1 | **V1.0** |
        |--------|------|----------|
        | ESM模型 | 轻量投影 | **ESM2-650M (1280d)** |
        | 训练数据 | ~73条 | **238条文献突变** |
        | 结构基础 | Mock随机图 | **1MSW真实结构** |
        | 单突变评估 | 单一概率 | **多机制协同度+加权适应度** |
        | 组合设计 | 简单冲突 | **4象限上位效应(距离+机制)** |
        
        ### 权重体系
        - **热稳定性 (1.1.x)**: 35%
        - **dsRNA减少 (1.4.x/1.5.2)**: 35% ⭐
        - **总产率 (1.2.x)**: 20%
        - **其他**: 10%
        
        ### 4象限上位效应
        ```
                    相同机制      不同机制
        近距离   ⚠️ 上位风险   ⚠️ 谨慎突变
        远距离   ✅ 叠加候选   🌟 多功能推荐
        ```
        """)
    with col2:
        st.info(f"**版本**: v1.0  \n**模型**: ✅已加载  \n**ESM**: 650M (1280d)  \n**数据**: 238条  \n**结构**: 1MSW (2.1Å)")
        if MSW_COORDS is not None:
            st.success("✅ 1MSW坐标已加载")
        else:
            st.warning("⚠️ 1MSW未加载，使用序列距离近似")

elif page == "🔍 位点扫描":
    st.title("🔍 位点扫描器 V1.0")
    st.markdown("按**多机制协同度**和**加权适应度**排序。优先推荐同时提升热稳+降低dsRNA的突变。")
    col_input, col_result = st.columns([1, 3])
    with col_input:
        scan_site_num = st.number_input("位点", 1, 883, 43, 1)
        scan_target = st.selectbox("目标大类", ["All", "stability", "activity", "quality", "allostery", "promoter"])
        scan_topk = st.slider("Top-K", 3, 19, 8)
        scan_btn = st.button("🚀 扫描", type="primary")
    with col_result:
        if scan_btn:
            with st.spinner("扫描中..."):
                result = scan_site(int(scan_site_num), scan_target if scan_target != "All" else None, scan_topk)
            st.subheader(f"Site {result['site']} (WT={result['wt_aa']})")
            if not result['candidates']:
                st.warning("无显著机制激活")
            else:
                df = pd.DataFrame([{
                    "突变": c['mutation'], "主导机制": c['mechanism_name'], "大类": c['category'],
                    "适应度": f"{c['fitness']:.2f}", "协同度": c['synergy_count'],
                    "🌟多功能": "✅" if c['is_multifunction'] else "",
                    "热稳": "✅" if c['has_stability'] else "", "dsRNA↓": "✅" if c['has_dsRNA'] else "",
                } for c in result['candidates']])
                st.dataframe(df, use_container_width=True, hide_index=True)
                best = result['candidates'][0]
                st.markdown(f"**🏆 最佳: `{best['mutation']}`** 适应度:{best['fitness']:.2f} 协同:{best['synergy_count']}")
                if best['is_multifunction']:
                    st.balloons()
                    st.success("🌟 黄金组合: 同时提升热稳定性+降低dsRNA!")

elif page == "🧬 突变解析":
    st.title("🧬 突变机制解析 V1.0")
    col_input, col_result = st.columns([1, 3])
    with col_input:
        pred_site = st.number_input("位点", 1, 883, 639, 1)
        pred_wt = st.text_input("WT", "Y", max_chars=1).upper()
        pred_mut = st.text_input("Mut", "F", max_chars=1).upper()
        pred_btn = st.button("🔬 解析", type="primary")
    with col_result:
        if pred_btn:
            with st.spinner("解析中..."):
                report = predictor.predict(int(pred_site), pred_wt, pred_mut, prob_threshold=0.15)
            if report is None:
                st.error(f"WT不匹配! 位点{pred_site}实际WT是{T7_WT_SEQ[int(pred_site)-1]}")
            else:
                fitness = compute_weighted_fitness(report)
                col_m = st.columns(4)
                col_m[0].metric("V1.0适应度", f"{fitness:.2f}")
                pos_count = len([m for m in report['activated_mechanisms'] if m['direction'] in ['positive', 'enhance', 'increase'] or m['id'] in ['1.4.1', '1.4.3']])
                col_m[1].metric("正向机制", pos_count)
                col_m[2].metric("主导大类", report['dominant_category'])
                col_m[3].metric("主导机制", report['dominant_mechanism'])
                
                col_info, col_viz = st.columns([1, 1])
                with col_info:
                    st.markdown("**激活机制 (按权重排序):**")
                    for m in sorted(report['activated_mechanisms'], key=lambda x: MECH_WEIGHTS.get(x['id'],0)*x['prob']*x['magnitude'], reverse=True):
                        badge = "🔥" if m['id'] in ['1.4.1', '1.4.3', '1.1.1', '1.2.1'] else ""
                        st.markdown(f"- `{m['id']}` **{m['name']}** {badge}  P={m['prob']:.2f} | W={MECH_WEIGHTS.get(m['id'],0):.2f}")
                    has_stab = any(m['id'].startswith('1.1') for m in report['activated_mechanisms'] if m['direction'] in ['positive', 'enhance'])
                    has_dsRNA = any(m['id'] in ['1.4.1', '1.4.3'] for m in report['activated_mechanisms'])
                    if has_stab and has_dsRNA:
                        st.balloons(); st.success("🌟 黄金组合: 热稳+dsRNA↓!")
                with col_viz:
                    cats = ['stability', 'activity', 'promoter', 'quality', 'allostery']
                    scores = [report['category_scores'][c] for c in cats]
                    angles = np.linspace(0, 2*np.pi, len(cats), endpoint=False).tolist()
                    scores_plot = scores + scores[:1]; angles += angles[:1]
                    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
                    ax.fill(angles, scores_plot, color='#2ecc71', alpha=0.3)
                    ax.plot(angles, scores_plot, color='#2ecc71', linewidth=2, marker='o')
                    ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats, fontsize=10)
                    ax.set_ylim(0, max(scores)*1.3 if max(scores)>0 else 1)
                    ax.set_title(f"{report['mutation']}", pad=20, fontsize=11)
                    plt.tight_layout(); st.pyplot(fig)

elif page == "🔗 组合设计":
    st.title("🔗 组合突变设计 V1.0")
    st.markdown("""
    **4象限上位效应评估** (基于1MSW真实结构距离)
    - 🌟 远距离+不同机制: 理想多功能酶
    - ✅ 远距离+同机制: 可尝试叠加
    - ⚠️ 近距离+不同机制: 谨慎突变
    - ❌ 近距离+同机制: 上位效应风险
    """)
    combo_input = st.text_area("突变列表（逗号分隔）", value="S43E, Y639F, V725A")
    combo_btn = st.button("🔗 评估", type="primary")
    if combo_btn:
        muts = [m.strip() for m in combo_input.split(",") if m.strip()]
        with st.spinner("基于1MSW结构评估..."):
            combo = evaluate_combination_v1(muts)
        if combo is None:
            st.error("需要至少2个有效突变")
        else:
            st.subheader("综合评估")
            c = st.columns(4)
            c[0].metric("V1.0适应度", f"{combo['fitness']:.2f}")
            c[1].metric("协同度", combo['synergy']['label'])
            c[2].metric("平均距离", f"{combo['synergy']['avg_distance']}Å")
            c[3].error(f"⚠️ 高风险: {combo['conflicts']}") if combo['conflicts'] > 0 else c[3].success("✅ 无高风险")
            
            st.subheader("4象限上位效应分析")
            for pair in combo['pair_analysis']:
                with st.container():
                    cp = st.columns([1, 2, 1])
                    if pair['label'] == "多功能推荐": cp[0].success(f"🌟 {pair['label']}")
                    elif pair['label'] == "叠加候选": cp[0].info(f"✅ {pair['label']}")
                    elif pair['label'] == "谨慎突变": cp[0].warning(f"⚠️ {pair['label']}")
                    else: cp[0].error(f"❌ {pair['label']}")
                    cp[1].markdown(f"**{pair['pair']}**  \n距离:`{pair['distance']}Å` | 机制:`{pair['mech_i']}` vs `{pair['mech_j']}`  \n{pair['advice']}")
                    cp[2].progress(1.0-pair['risk'], text=f"安全度:{int((1-pair['risk'])*100)}%")
            
            st.subheader("机制大类覆盖")
            fig, ax = plt.subplots(figsize=(8,4))
            cats_sorted = list(combo['category_scores'].keys())
            scores_sorted = list(combo['category_scores'].values())
            colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
            ax.barh(cats_sorted, scores_sorted, color=colors[:len(cats_sorted)])
            ax.set_xlabel("Score"); ax.set_title(f"{' + '.join(combo['mutations'])}")
            plt.tight_layout(); st.pyplot(fig)

elif page == "📊 实验反馈":
    st.title("📊 主动学习 V1.0")
    st.markdown("**优先收集dsRNA、热稳定性和产率数据**")
    storage = FeedbackStorage()
    with st.form("feedback_form"):
        st.subheader("基础信息")
        c1, c2, c3 = st.columns(3)
        with c1: fb_id = st.text_input("实验ID", "Lab_001"); fb_exp = st.text_input("实验者", "anonymous")
        with c2: fb_site = st.number_input("位点", 1, 883, 43); fb_wt = st.text_input("WT", "S", max_chars=1).upper()
        with c3: fb_mut = st.text_input("Mut", "E", max_chars=1).upper(); fb_buf = st.text_input("缓冲液", "Tris-HCl pH 7.9")
        
        st.subheader("🔥 核心指标 (强烈建议)")
        c1, c2, c3, c4 = st.columns(4)
        with c1: fb_tm = st.number_input("Tm (°C)", None, step=0.1, format="%.1f"); fb_dtm = st.number_input("ΔTm", None, step=0.1, format="%.1f")
        with c2: fb_dsRNA = st.number_input("dsRNA (%) ⭐", None, step=0.1, format="%.2f"); fb_fid = st.number_input("保真度", None, step=0.01, format="%.3f")
        with c3: fb_yield = st.number_input("产量(mg/L) ⭐", None, step=1.0, format="%.1f"); fb_act = st.number_input("相对活性", None, step=0.1, format="%.2f")
        with c4: fb_hl = st.number_input("半衰期(min)", None, step=1.0, format="%.1f"); fb_kcat = st.number_input("kcat/Km", None, step=1000.0, format="%.1f")
        
        st.subheader("🧬 Quality指标 (可选)")
        cq1, cq2 = st.columns(2)
        with cq1: fb_cap = st.number_input("加帽效率(%)", None, step=1.0, format="%.1f")
        with cq2: fb_full = st.number_input("全长比例(%)", None, step=1.0, format="%.1f")
        
        fb_mech = st.selectbox("机制标签", ["自动推断", "1.1.1 cavity_filling", "1.1.2 salt_bridge", "1.1.3 helix_rigidification",
             "1.2.1 dna_binding_enhance", "1.4.1 rna_release", "1.4.3 fidelity_enhance", "1.5.2 foot_regulation"])
        fb_notes = st.text_area("备注", placeholder="例如: 42°C下dsRNA显著降低...")
        submitted = st.form_submit_button("📤 提交", type="primary")
    
    if submitted:
        try:
            datum = ExperimentalDatum(
                protein_id=fb_id, mutation=f"{fb_wt}{int(fb_site)}{fb_mut}", site=int(fb_site),
                wt_aa=fb_wt, mut_aa=fb_mut, assay_temperature=37.0, assay_buffer=fb_buf,
                tm_value=fb_tm, tm_delta=fb_dtm, dsRNA_ratio=fb_dsRNA, yield_mg_per_L=fb_yield,
                activity_relative=fb_act, kcat_km=fb_kcat, fidelity_index=fb_fid, half_life_min=fb_hl,
                cap_efficiency=fb_cap, full_length_ratio=fb_full,
                user_mechanism_label=fb_mech if fb_mech != "自动推断" else None,
                experimenter=fb_exp, notes=fb_notes
            )
            result = storage.submit(datum)
            st.success(f"✅ 质量分: {result['quality_score']:.2f}")
            if result['quality_score'] >= 0.6: st.balloons(); st.success("高质量数据!")
            elif result['quality_score'] >= 0.3: st.info("中等质量，建议补充核心指标")
            else: st.warning("质量较低，建议填写dsRNA/Tm/产量")
        except Exception as e:
            st.error(f"❌ 失败: {e}")
    
    st.markdown("---"); st.subheader("🔒 管理员")
    admin_pwd = st.text_input("密码", type="password")
    if admin_pwd == "t7admin":
        pending = storage.get_pending()
        if pending.empty: st.info("无待审核")
        else:
            st.write(f"待审核: {len(pending)}")
            st.dataframe(pending[['timestamp', 'protein_id', 'mutation', 'experimenter', 'data_quality_score', 'dsRNA_ratio', 'tm_value', 'yield_mg_per_L']], use_container_width=True)
            to_merge = st.multiselect("选择合并", options=pending['protein_id'].tolist(),
                default=pending[pending['data_quality_score']>=0.6]['protein_id'].tolist() if 'data_quality_score' in pending.columns else [])
            if st.button("🔄 合并到训练集", type="primary"):
                with st.spinner("合并中..."): n = storage.merge_selected(to_merge)
                if n > 0: st.success(f"✅ 合并{n}条! 请重新运行generate_data.py+train.py"); st.info("下载更新后的annotations.csv并重新训练")
                else: st.warning("无新数据合并")

st.sidebar.markdown("---")
st.sidebar.markdown("**v1.0 Production**  \n650M | 238条 | 1MSW结构")
