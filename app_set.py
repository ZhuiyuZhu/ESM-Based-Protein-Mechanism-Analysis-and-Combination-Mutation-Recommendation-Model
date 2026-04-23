#!/usr/bin/env python3
"""
T7 RNAP 机制分类器 - Streamlit 版
运行: streamlit run app_set.py
"""

import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 确保在项目根目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# 导入项目模块
from predict import T7MechPredictor, T7_WT_SEQ
from mechanism_ontology import MECHANISM_ONTOLOGY, ALL_MECHANISMS, CATEGORY_NAMES
from combinatorial import CombinatorialOptimizer, MutantReport

# ================== 轻量 ESM 编码器（和 predict.py 一致）====================
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

PROJ_PATH = 'data/processed/esm_embeddings/projection_matrix.pt'
if os.path.exists(PROJ_PATH):
    PROJ_MATRIX = np.load(PROJ_PATH) if PROJ_PATH.endswith('.npy') else __import__('torch').load(PROJ_PATH)
else:
    import torch
    torch.manual_seed(42)
    PROJ_MATRIX = torch.nn.init.orthogonal_(torch.randn(20, 1280))
    os.makedirs('data/processed/esm_embeddings', exist_ok=True)
    torch.save(PROJ_MATRIX, PROJ_PATH)

# ================== 缓存加载模型（全局只加载一次）====================
@st.cache_resource
def load_model():
    predictor = T7MechPredictor()
    return predictor

predictor = load_model()

# ================== 辅助函数 ==================
def encode_sequence(seq):
    """One-hot -> 正交投影（和 predict.py 一致）"""
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

page = st.sidebar.radio("导航", ["🏠 首页", "🔍 位点扫描", "🧬 突变解析", "🔗 组合设计"])

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
        
        ### 机制本体
        
        - **1.1.x** 结构稳定性（空腔填充、盐桥、螺旋刚性化）
        - **1.2.x** 催化活性（DNA结合、NTP识别、金属配位）
        - **1.3.x** 启动子识别（PBD旋转、特异性环）
        - **1.4.x** 产物质量（RNA释放、保真度、自引物抑制）
        - **1.5.x** 变构通讯（NTD-CTD耦合、Foot调控）
        """)
    
    with col2:
        st.info("""
        **当前版本**: v0.1 (Proof of Concept)
        
        **模型状态**: ✅ 已加载
        
        **ESM编码**: 轻量版 (One-hot投影)
        
        **训练数据**: 73条文献突变
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
                # 表格
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
                
                # 最佳突变的机制覆盖图
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
                    st.markdown(f"""
                    **主导大类**: `{report['dominant_category']}`  
                    **主导机制**: `{report['dominant_mechanism']}`
                    """)
                    
                    st.markdown("**激活机制详情**:")
                    for m in sorted(report['activated_mechanisms'], key=lambda x: -x['prob']):
                        st.markdown(f"""
                        - `{m['id']}` **{m['name']}** ({m['category']})  
                          概率: `{m['prob']:.2f}` | 方向: `{m['direction']}` | 大小: `{m['magnitude']:.2f}`
                        """)
                    
                    if report['conflicts']:
                        st.error("⚠️ 检测到机制冲突！")
                        for c in report['conflicts']:
                            st.write(f"- {c[2]}")
                    else:
                        st.success("✅ 无机制冲突")
                
                with col_radar:
                    # 雷达图
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
                
                # 大类柱状图
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
                cat_data = {
                    cat: score 
                    for cat, score in sorted(combo['category_scores'].items(), key=lambda x: -x[1])
                }
                
                # 横向条形图
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
                
                # 激活机制列表
                st.markdown("**组合激活的机制**:")
                mech_cols = st.columns(3)
                for i, mech in enumerate(combo['activated'][:9]):
                    with mech_cols[i % 3]:
                        name = MECHANISM_ONTOLOGY[mech][0]
                        cat = MECHANISM_ONTOLOGY[mech][1]
                        st.markdown(f"`{mech}` **{name}**  \n*{cat}*")

# ================== 页脚 ==================
st.sidebar.markdown("---")
st.sidebar.markdown("""
**v0.1 Proof of Concept**  
[GitHub Repo](https://github.com/ZhuiyuZhu/ESM-Based-Protein-Mechanism-Analysis-and-Combination-Mutation-Recommendation-Model)
""")
