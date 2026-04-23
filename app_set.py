#!/usr/bin/env python3
"""
T7 RNAP 机制分类器 - Streamlit 完整版
含：位点扫描、突变解析、组合设计、实验反馈（主动学习）
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, List

# ================== 路径处理：确保在项目根目录 ==================
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ================== 轻量 ESM 编码器（和 predict.py 一致）====================
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

PROJ_PATH = 'data/processed/esm_embeddings/projection_matrix.pt'
if os.path.exists(PROJ_PATH):
    import torch
    PROJ_MATRIX = torch.load(PROJ_PATH)
else:
    import torch
    torch.manual_seed(42)
    PROJ_MATRIX = torch.nn.init.orthogonal_(torch.randn(20, 1280))
    os.makedirs('data/processed/esm_embeddings', exist_ok=True)
    torch.save(PROJ_MATRIX, PROJ_PATH)

# ================== 模型路径自动探测（兼容根目录和子目录）====================
def find_model_path():
    candidates = [
        'outputs/exp_001/best_model.pt',
        'best_model.pt',
        'outputs/exp_001/best_model.pth'
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
except ImportError as e:
    st.error(f"导入项目模块失败: {e}")
    st.stop()

# ================== 缓存加载模型 ==================
@st.cache_resource
def load_model():
    if MODEL_PATH is None:
        raise FileNotFoundError(
            "Model not found. Expected 'outputs/exp_001/best_model.pt' or 'best_model.pt' in repo root."
        )
    # 临时 patch predict.py 的默认路径
    import predict
    original_init = predict.T7MechPredictor.__init__
    def patched_init(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = MODEL_PATH
        original_init(self, checkpoint_path)
    predict.T7MechPredictor.__init__ = patched_init
    
    predictor = T7MechPredictor()
    return predictor

try:
    predictor = load_model()
except Exception as e:
    st.error(f"模型加载失败: {e}")
    st.info("请确认 best_model.pt 已上传至仓库根目录或 outputs/exp_001/")
    st.stop()

# ================== 辅助函数 ==================
def encode_sequence(seq):
    """One-hot -> 正交投影"""
    import torch
    indices = [AA_TO_IDX.get(aa, 0) for aa in seq]
    onehot = torch.zeros(len(seq), 20)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0
    return onehot @ PROJ_MATRIX

def scan_site(site, target_category=None, top_k=10):
    """扫描单个位点的所有可能突变"""
    idx = site - 1
    wt_aa = T7_WT_SEQ[idx]
    results = []
    
    for mut_aa in AA_LIST:
        if mut_aa == wt_aa:
            continue
        report = predictor.predict(site, wt_aa, mut_aa, prob_threshold=0.2)
        if report is None or not report['activated_mechanisms']:
            continue
        
        top = max(report['activated_mechanisms'], key=lambda x: x['prob'])
        results.append({
            'mutation': report['mutation'],
            'mechanism': top['id'],
            'mechanism_name': top['name'],
            'category': top['category'],
            'probability': float(top['prob']),
            'direction': top['direction'],
            'magnitude': float(top['magnitude']),
            'category_scores': report['category_scores'],
            'representation': report['representation']
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    if target_category:
        filtered = [r for r in results if r['category'] == target_category]
        results = filtered if filtered else results
    return {
        'site': site,
        'wt_aa': wt_aa,
        'candidates': results[:top_k]
    }

def evaluate_combination(mutation_list):
    """评估突变组合"""
    reports = []
    for m in mutation_list:
        if len(m) < 3:
            continue
        wt, site_str, mut_aa = m[0], m[1:-1], m[-1]
        site = int(site_str)
        r = predictor.predict(site, wt, mut_aa, prob_threshold=0.2)
        if r is None:
            continue
        reports.append(MutantReport(
            mutation=m,
            site=site,
            mechanism_probs={a['id']: a['prob'] for a in r['activated_mechanisms']},
            dominant_mech=r['dominant_mechanism'],
            effect_magnitude=sum(a['magnitude'] for a in r['activated_mechanisms']),
            representation=r['representation'],
            category_scores=r['category_scores']
        ))
    
    if len(reports) < 2:
        return None
    opt = CombinatorialOptimizer(reports, max_size=len(reports), min_site_dist=3)
    return opt._evaluate([r.mutation for r in reports])

# ================== 主动学习：数据类与存储（内联简化版）====================
@dataclass
class ExperimentalDatum:
    protein_id: str
    mutation: str
    site: int
    wt_aa: str
    mut_aa: str
    assay_temperature: float = 37.0
    assay_buffer: str = "Tris-HCl pH 7.9"
    tm_value: Optional[float] = None
    tm_delta: Optional[float] = None
    kcat_km: Optional[float] = None
    activity_relative: Optional[float] = None
    dsRNA_ratio: Optional[float] = None
    fidelity_index: Optional[float] = None
    yield_mg_per_L: Optional[float] = None
    half_life_min: Optional[float] = None
    user_mechanism_label: Optional[str] = None
    experimenter: str = "anonymous"
    notes: str = ""
    
    def validate(self):
        assert 1 <= self.site <= 883, "位点必须在 1-883 之间"
        assert len(self.wt_aa) == 1 and len(self.mut_aa) == 1, "氨基酸单字母"
        assert self.wt_aa != self.mut_aa, "WT 和 Mut 不能相同"
        has_metric = any([self.tm_value, self.tm_delta, self.kcat_km,
                         self.activity_relative, self.dsRNA_ratio,
                         self.fidelity_index, self.yield_mg_per_L, self.half_life_min])
        assert has_metric, "至少填写一项定量指标"
        return True
    
    def to_dict(self):
        return {
            'timestamp': datetime.now().isoformat(),
            'protein_id': self.protein_id,
            'mutation': self.mutation,
            'site': self.site,
            'wt_aa': self.wt_aa,
            'mut_aa': self.mut_aa,
            'assay_temperature': self.assay_temperature,
            'assay_buffer': self.assay_buffer,
            'tm_value': self.tm_value,
            'tm_delta': self.tm_delta,
            'kcat_km': self.kcat_km,
            'activity_relative': self.activity_relative,
            'dsRNA_ratio': self.dsRNA_ratio,
            'fidelity_index': self.fidelity_index,
            'yield_mg_per_L': self.yield_mg_per_L,
            'half_life_min': self.half_life_min,
            'user_mechanism_label': self.user_mechanism_label,
            'experimenter': self.experimenter,
            'notes': self.notes,
            'data_quality_score': 0.0,
            'model_agreement': None,
            'status': 'pending'
        }

class FeedbackStorage:
    def __init__(self, storage_dir='data/user_feedback'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.pending_csv = os.path.join(storage_dir, 'feedback_pending.csv')
        if not os.path.exists(self.pending_csv):
            cols = ['timestamp', 'protein_id', 'mutation', 'site', 'wt_aa', 'mut_aa',
                    'assay_temperature', 'assay_buffer', 'tm_value', 'tm_delta',
                    'kcat_km', 'activity_relative', 'dsRNA_ratio', 'fidelity_index',
                    'yield_mg_per_L', 'half_life_min', 'user_mechanism_label',
                    'experimenter', 'notes', 'data_quality_score', 'model_agreement', 'status']
            pd.DataFrame(columns=cols).to_csv(self.pending_csv, index=False)
    
    def submit(self, datum: ExperimentalDatum):
        datum.validate()
        row = datum.to_dict()
        # 计算质量分
        filled = sum([row['tm_value'] is not None, row['tm_delta'] is not None,
                     row['kcat_km'] is not None, row['activity_relative'] is not None,
                     row['dsRNA_ratio'] is not None, row['fidelity_index'] is not None,
                     row['yield_mg_per_L'] is not None, row['half_life_min'] is not None])
        row['data_quality_score'] = round(filled / 8.0, 2)
        
        df = pd.DataFrame([row])
        df.to_csv(self.pending_csv, mode='a', header=False, index=False)
        return {
            'status': 'pending',
            'quality_score': row['data_quality_score'],
            'message': '数据已提交，等待审核后并入训练集'
        }
    
    def get_pending(self):
        return pd.read_csv(self.pending_csv) if os.path.exists(self.pending_csv) else pd.DataFrame()
    
    def merge_selected(self, protein_ids: List[str]):
        df = self.get_pending()
        to_merge = df[df['protein_id'].isin(protein_ids)]
        if to_merge.empty:
            return 0
        
        # 追加到训练集 annotations.csv
        train_path = 'data/processed/annotations.csv'
        if not os.path.exists(train_path):
            st.error("训练集文件不存在！")
            return 0
        
        train_df = pd.read_csv(train_path)
        new_rows = []
        for _, row in to_merge.iterrows():
            mech = row['user_mechanism_label'] if pd.notna(row['user_mechanism_label']) else '1.1.1'
            # 简化：只填充主导机制和基础列
            new_row = {
                'protein_id': row['protein_id'],
                'mutation': row['mutation'],
                'site': int(row['site']),
                'wt_aa': row['wt_aa'],
                'mut_aa': row['mut_aa'],
                'split': 'train',
                'dominant_mech': mech,
            }
            # 填充所有机制列（简化：只标主导机制）
            for m in ALL_MECHANISMS:
                new_row[m] = 1 if m == mech else 0
            # 效应方向/大小（用 activity_relative 映射，默认 0.5）
            effect = row['activity_relative'] if pd.notna(row['activity_relative']) else 0.5
            for m in ALL_MECHANISMS:
                new_row[f'effect_sign_{m}'] = 1 if m == mech else 0
                new_row[f'effect_size_{m}'] = effect if m == mech else 0.0
            new_rows.append(new_row)
        
        new_df = pd.DataFrame(new_rows)
        # 去重
        existing = set(train_df['mutation'].tolist())
        new_df = new_df[~new_df['mutation'].isin(existing)]
        
        combined = pd.concat([train_df, new_df], ignore_index=True)
        combined.to_csv(train_path, index=False)
        
        # 从 pending 移除已合并
        remaining = df[~df['protein_id'].isin(protein_ids)]
        remaining.to_csv(self.pending_csv, index=False)
        
        return len(new_df)

# ================== Streamlit 页面配置 ==================
st.set_page_config(
    page_title="T7 RNAP Mechanism Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.title("🔬 T7 RNAP Mech-Evo")
st.sidebar.markdown("""
**机制感知定向进化平台**  
显式解耦突变物理化学机制，设计组合突变。
""")

page = st.sidebar.radio("导航", ["🏠 首页", "🔍 位点扫描", "🧬 突变解析", "🔗 组合设计", "📊 实验反馈"])

# ================== 首页 ==================
if page == "🏠 首页":
    st.title("T7 RNA 聚合酶机制分类器")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### 核心能力
        
        | 功能 | 说明 |
        |------|------|
        | **🔍 位点扫描** | 输入位点，遍历19种氨基酸，推荐机制最优突变 |
        | **🧬 突变解析** | 预测突变的15种细粒度机制激活谱 |
        | **🔗 组合设计** | 评估多个突变的机制互补性与冲突 |
        | **📊 实验反馈** | 提交实验数据，驱动模型主动学习 |
        
        ### 机制本体
        
        - **1.1.x** 结构稳定性（空腔填充、盐桥、螺旋刚性化）
        - **1.2.x** 催化活性（DNA结合、NTP识别、金属配位）
        - **1.3.x** 启动子识别（PBD旋转、特异性环）
        - **1.4.x** 产物质量（RNA释放、保真度、自引物抑制）
        - **1.5.x** 变构通讯（NTD-CTD耦合、Foot调控）
        """)
    with col2:
        st.info(f"""
        **当前版本**: v0.1 (Proof of Concept)
        
        **模型状态**: ✅ 已加载  
        **权重路径**: `{MODEL_PATH or 'NOT FOUND'}`
        
        **ESM编码**: 轻量版 (One-hot投影)  
        **训练数据**: ~73条文献突变
        """)

# ================== 位点扫描 ==================
elif page == "🔍 位点扫描":
    st.title("🔍 位点扫描器")
    st.markdown("输入一个位点，自动遍历19种氨基酸替换，按机制激活概率排序推荐最优突变。")
    col_input, col_result = st.columns([1, 3])
    
    with col_input:
        scan_site_num = st.number_input("位点 (Site)", min_value=1, max_value=883, value=43, step=1)
        scan_target = st.selectbox(
            "目标机制大类",
            ["All", "stability", "activity", "quality", "allostery", "promoter"]
        )
        scan_topk = st.slider("显示Top-K", min_value=3, max_value=19, value=5)
        scan_btn = st.button("🚀 开始扫描", type="primary")
    
    with col_result:
        if scan_btn:
            with st.spinner("正在扫描19种氨基酸替换..."):
                result = scan_site(int(scan_site_num), scan_target if scan_target != "All" else None, scan_topk)
            st.subheader(f"Site {result['site']} (WT = {result['wt_aa']})")
            
            if not result['candidates']:
                st.warning("该位点未预测到显著机制激活，建议换一位点尝试。")
            else:
                df = pd.DataFrame([
                    {
                        "突变": c['mutation'],
                        "机制": c['mechanism_name'],
                        "大类": c['category'],
                        "概率": f"{c['probability']:.2f}",
                        "方向": c['direction'],
                        "效应大小": f"{c['magnitude']:.2f}"
                    }
                    for c in result['candidates']
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # 最佳突变机制覆盖图
                best = result['candidates'][0]
                cats = ['stability', 'activity', 'promoter', 'quality', 'allostery']
                scores = [best['category_scores'].get(c, 0) for c in cats]
                fig, ax = plt.subplots(figsize=(8, 3))
                colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
                bars = ax.barh(cats, scores, color=colors)
                ax.set_xlim(0, max(scores) * 1.5 + 0.1)
                ax.set_title(f"Best Candidate: {best['mutation']} ({best['mechanism_name']})")
                for bar, score in zip(bars, scores):
                    if score > 0.05:
                        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                               f"{score:.2f}", va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                
                # 概率排序条形图
                fig2, ax2 = plt.subplots(figsize=(8, 3))
                mut_names = [c['mutation'] for c in result['candidates']]
                probs = [c['probability'] for c in result['candidates']]
                colors2 = [colors[['stability','activity','promoter','quality','allostery'].index(c['category'])] 
                          for c in result['candidates']]
                ax2.barh(mut_names[::-1], probs[::-1], color=colors2[::-1])
                ax2.set_xlabel("Activation Probability")
                ax2.set_title("Mutation Ranking by Mechanism Probability")
                plt.tight_layout()
                st.pyplot(fig2)

# ================== 突变解析 ==================
elif page == "🧬 突变解析":
    st.title("🧬 突变机制解析")
    st.markdown("输入指定突变，查看其15种细粒度机制激活谱。")
    col_input, col_result = st.columns([1, 3])
    
    with col_input:
        pred_site = st.number_input("位点", min_value=1, max_value=883, value=639, step=1)
        pred_wt = st.text_input("WT氨基酸", value="Y", max_chars=1).upper()
        pred_mut = st.text_input("突变氨基酸", value="F", max_chars=1).upper()
        pred_btn = st.button("🔬 解析机制", type="primary")
    
    with col_result:
        if pred_btn:
            with st.spinner("正在解析机制..."):
                report = predictor.predict(int(pred_site), pred_wt, pred_mut, prob_threshold=0.2)
            if report is None:
                st.error(f"WT不匹配！位点 {pred_site} 的实际WT是 {T7_WT_SEQ[int(pred_site)-1]}，不是 {pred_wt}")
            else:
                st.subheader(f"{report['mutation']} 机制报告")
                col_info, col_radar = st.columns([1, 1])
                
                with col_info:
                    st.markdown(f"**主导大类**: `{report['dominant_category']}`  \n**主导机制**: `{report['dominant_mechanism']}`")
                    st.markdown("**激活机制详情**:")
                    for m in sorted(report['activated_mechanisms'], key=lambda x: -x['prob']):
                        st.markdown(f"- `{m['id']}` **{m['name']}** ({m['category']})  \n  概率: `{m['prob']:.2f}` | 方向: `{m['direction']}` | 大小: `{m['magnitude']:.2f}`")
                    if report['conflicts']:
                        st.error("⚠️ 检测到机制冲突！")
                        for c in report['conflicts']:
                            st.write(f"- {c[2]}")
                    else:
                        st.success("✅ 无机制冲突")
                
                with col_radar:
                    cats = ['stability', 'activity', 'promoter', 'quality', 'allostery']
                    scores = [report['category_scores'][c] for c in cats]
                    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
                    scores_plot = scores + scores[:1]
                    angles += angles[:1]
                    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
                    ax.fill(angles, scores_plot, color='#3498db', alpha=0.3)
                    ax.plot(angles, scores_plot, color='#3498db', linewidth=2, marker='o')
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(cats, fontsize=10)
                    ax.set_ylim(0, max(scores) * 1.3 if max(scores) > 0 else 1)
                    ax.set_title(f"Mechanism Profile\n{report['mutation']}", pad=20, fontsize=11)
                    plt.tight_layout()
                    st.pyplot(fig)
                
                fig2, ax2 = plt.subplots(figsize=(6, 3))
                colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
                bars = ax2.bar(cats, scores, color=colors)
                ax2.set_ylabel("Score")
                ax2.set_title("Category Scores")
                for bar, score in zip(bars, scores):
                    if score > 0.05:
                        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f"{score:.2f}", ha='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)

# ================== 组合设计 ==================
elif page == "🔗 组合设计":
    st.title("🔗 组合突变设计")
    st.markdown("输入多个突变（逗号分隔），评估其机制互补性、协同效应和潜在冲突。")
    
    combo_input = st.text_area(
        "突变列表（逗号分隔）",
        value="S43E, Q786M, Y639F",
        placeholder="例如: S43E, Q786M, Y639F, K631A"
    )
    combo_btn = st.button("🔗 评估组合", type="primary")
    
    if combo_btn:
        muts = [m.strip() for m in combo_input.split(",") if m.strip()]
        with st.spinner("正在评估组合..."):
            combo = evaluate_combination(muts)
        if combo is None:
            st.error("需要至少2个有效突变。请检查WT是否匹配、格式是否正确（如 S43E）。")
        else:
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Fitness Score", f"{combo['fitness']:.2f}")
                st.metric("Mechanism Diversity", f"{combo['diversity']:.2f}")
                st.metric("Synergy Score", f"{combo['synergy']:.2f}")
                if combo['conflicts'] > 0:
                    st.error(f"⚠️ 冲突数: {combo['conflicts']}")
                else:
                    st.success("✅ 无冲突")
                st.markdown("**位点分布**:")
                st.write(f"`{combo['sites']}`")
            
            with col2:
                st.markdown("**机制大类覆盖**:")
                cat_data = {cat: score for cat, score in sorted(combo['category_scores'].items(), key=lambda x: -x[1])}
                fig, ax = plt.subplots(figsize=(8, 4))
                cats_sorted = list(cat_data.keys())
                scores_sorted = list(cat_data.values())
                colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
                bars = ax.barh(cats_sorted, scores_sorted, color=colors[:len(cats_sorted)])
                ax.set_xlabel("Cumulative Activation Score")
                ax.set_title(f"Combination: {' + '.join(combo['mutations'])}")
                for bar, score in zip(bars, scores_sorted):
                    if score > 0.05:
                        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                               f"{score:.2f}", va='center', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                
                st.markdown("**组合激活的机制**:")
                mech_cols = st.columns(3)
                for i, mech in enumerate(combo['activated'][:9]):
                    with mech_cols[i % 3]:
                        name = MECHANISM_ONTOLOGY[mech][0]
                        cat = MECHANISM_ONTOLOGY[mech][1]
                        st.markdown(f"`{mech}` **{name}**  \n*{cat}*")

# ================== 实验反馈（主动学习） ==================
elif page == "📊 实验反馈":
    st.title("📊 主动学习：实验数据反馈")
    st.markdown("""
    **提交你的实验数据，帮助模型学习！**
    
    数据规范：
    - 突变位点：1-883（T7 RNAP 全长）
    - 至少提供一项定量指标（Tm、活性、dsRNA 等）
    - 实验条件需注明（温度、缓冲液）
    """)
    
    storage = FeedbackStorage()
    
    with st.form("feedback_form"):
        st.subheader("基础信息")
        col1, col2, col3 = st.columns(3)
        with col1:
            fb_protein_id = st.text_input("实验ID", value="Lab_001")
            fb_experimenter = st.text_input("实验者", value="anonymous")
        with col2:
            fb_site = st.number_input("位点", min_value=1, max_value=883, value=43)
            fb_wt = st.text_input("WT", value="S", max_chars=1).upper()
        with col3:
            fb_mut = st.text_input("Mut", value="E", max_chars=1).upper()
            fb_buffer = st.text_input("缓冲液", value="Tris-HCl pH 7.9")
        
        st.subheader("定量指标（至少填一项）")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            fb_tm = st.number_input("Tm (°C)", value=None, step=0.1, format="%.1f")
            fb_tm_delta = st.number_input("ΔTm (°C)", value=None, step=0.1, format="%.1f")
        with col_m2:
            fb_activity = st.number_input("相对活性", value=None, step=0.1, format="%.2f")
            fb_kcat = st.number_input("kcat/Km", value=None, step=1000.0, format="%.1f")
        with col_m3:
            fb_dsRNA = st.number_input("dsRNA (%)", value=None, step=0.1, format="%.2f")
            fb_fidelity = st.number_input("保真度指数", value=None, step=0.01, format="%.3f")
        with col_m4:
            fb_yield = st.number_input("产量 (mg/L)", value=None, step=1.0, format="%.1f")
            fb_halflife = st.number_input("半衰期 (min)", value=None, step=1.0, format="%.1f")
        
        st.subheader("机制标签（可选）")
        fb_mech = st.selectbox(
            "你认为这是什么机制？（不填则由模型自动推断）",
            ["自动推断", "1.1.1 cavity_filling", "1.1.2 salt_bridge", "1.1.3 helix_rigidification",
             "1.2.1 dna_binding_enhance", "1.2.2 ntp_recognition", "1.4.1 rna_release",
             "1.4.3 fidelity_enhance", "1.5.1 ntd_ctd_coupling", "1.5.2 foot_regulation"]
        )
        fb_notes = st.text_area("备注", value="")
        submitted = st.form_submit_button("📤 提交数据", type="primary")
    
    if submitted:
        try:
            datum = ExperimentalDatum(
                protein_id=fb_protein_id,
                mutation=f"{fb_wt}{int(fb_site)}{fb_mut}",
                site=int(fb_site),
                wt_aa=fb_wt,
                mut_aa=fb_mut,
                assay_temperature=37.0,
                assay_buffer=fb_buffer,
                tm_value=fb_tm,
                tm_delta=fb_tm_delta,
                activity_relative=fb_activity,
                kcat_km=fb_kcat,
                dsRNA_ratio=fb_dsRNA,
                fidelity_index=fb_fidelity,
                yield_mg_per_L=fb_yield,
                half_life_min=fb_halflife,
                user_mechanism_label=fb_mech if fb_mech != "自动推断" else None,
                experimenter=fb_experimenter,
                notes=fb_notes
            )
            result = storage.submit(datum)
            st.success(f"✅ 提交成功！数据质量分: {result['quality_score']:.2f}/1.0")
            if result['quality_score'] >= 0.75:
                st.balloons()
                st.success("高质量数据！管理员可立即将其并入训练集。")
        except Exception as e:
            st.error(f"❌ 提交失败: {str(e)}")
    
    # 管理员面板
    st.markdown("---")
    st.subheader("🔒 管理员：待审核数据")
    admin_pwd = st.text_input("管理员密码", type="password", value="")
    if admin_pwd == "t7admin":
        pending = storage.get_pending()
        if pending.empty:
            st.info("暂无待审核数据")
        else:
            st.write(f"待审核记录数: {len(pending)}")
            display_df = pending[['timestamp', 'protein_id', 'mutation', 'experimenter', 
                                 'data_quality_score', 'status']].copy()
            st.dataframe(display_df, use_container_width=True)
            
            to_merge = st.multiselect(
                "选择要并入训练集的记录",
                options=pending['protein_id'].tolist(),
                default=pending[pending['data_quality_score'] >= 0.75]['protein_id'].tolist() if 'data_quality_score' in pending.columns else []
            )
            
            if st.button("🔄 合并到训练集", type="primary"):
                with st.spinner("正在合并数据..."):
                    n_merged = storage.merge_selected(to_merge)
                    if n_merged > 0:
                        st.success(f"✅ 已合并 {n_merged} 条新数据到训练集！")
                        st.info("请在本地下载更新后的 `data/processed/annotations.csv`，并运行 `python train.py` 重新训练模型。")
                    else:
                        st.warning("没有新数据被合并（可能已存在或选择为空）。")

# ================== 页脚 ==================
st.sidebar.markdown("---")
st.sidebar.markdown("""
**v0.1 Proof of Concept**  
[GitHub Repo](https://github.com/ZhuiyuZhu/ESM-Based-Protein-Mechanism-Analysis-and-Combination-Mutation-Recommendation-Model)
""")
