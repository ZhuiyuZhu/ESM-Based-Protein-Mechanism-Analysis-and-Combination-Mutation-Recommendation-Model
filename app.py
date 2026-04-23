#!/usr/bin/env python3
"""
T7 RNAP 机制分类器 - Gradio Web 界面
运行: python app.py
"""

import os
# 切到项目根目录（models/），因为权重文件在这里
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from predict import T7MechPredictor, T7_WT_SEQ
from combinatorial import CombinatorialOptimizer, MutantReport
from site_scanner import SiteScanner
from mechanism_ontology import MECHANISM_ONTOLOGY, CATEGORY_NAMES

# 初始化模型（全局只加载一次）
print("Loading model...")
PREDICTOR = T7MechPredictor()
SCANNER = SiteScanner(PREDICTOR)


def scan_single_site(site, target_category):
    """
    Gradio 回调：扫描单个位点
    """
    site = int(site)
    rec = SCANNER.recommend_mutations(site, target_category if target_category != "All" else None, top_k=10)

    # 构建表格数据
    data = []
    for c in rec['recommendations']:
        data.append([
            c['mutation'],
            c['mechanism_name'],
            c['category'],
            f"{c['probability']:.2f}",
            c['direction'],
            f"{c['magnitude']:.2f}"
        ])

    # 生成机制覆盖图
    fig, ax = plt.subplots(figsize=(8, 4))
    cats = ['stability', 'activity', 'promoter', 'quality', 'allostery']

    if rec['recommendations']:
        best = rec['recommendations'][0]
        scores = [best['category_scores'].get(c, 0) for c in cats]
        colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6']
        bars = ax.barh(cats, scores, color=colors)
        ax.set_xlim(0, max(scores) * 1.5 if max(scores) > 0 else 1)
        ax.set_title(f"Best Mutation: {best['mutation']} @ Site {site}")
        for bar, score in zip(bars, scores):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                    f"{score:.2f}", va='center')
    else:
        ax.text(0.5, 0.5, "No predictions", ha='center', va='center')

    plt.tight_layout()
    fig_path = "figures/scan_result.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return pd.DataFrame(data, columns=["Mutation", "Mechanism", "Category", "Prob", "Direction", "Magnitude"]), fig_path


def predict_mutation(site, wt, mut):
    """
    Gradio 回调：预测指定突变
    """
    site = int(site)
    report = PREDICTOR.predict(site, wt, mut, prob_threshold=0.2)

    if report is None:
        return "WT mismatch! Check sequence.", None

    # 文本报告
    lines = [
        f"Mutation: {report['mutation']}",
        f"Dominant Category: {report['dominant_category']}",
        f"Dominant Mechanism: {report['dominant_mechanism']}",
        "",
        "Activated Mechanisms:"
    ]
    for m in sorted(report['activated_mechanisms'], key=lambda x: -x['prob']):
        lines.append(
            f"  • {m['id']} {m['name']} ({m['category']}) | P={m['prob']:.2f} | {m['direction']} | Mag={m['magnitude']:.2f}")

    lines.append("")
    lines.append("Category Scores:")
    for cat, score in sorted(report['category_scores'].items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {score:.2f}")

    # 雷达图
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    cats = ['stability', 'activity', 'promoter', 'quality', 'allostery']
    scores = [report['category_scores'][c] for c in cats]
    angles = np.linspace(0, 2 * np.pi, len(cats), endpoint=False).tolist()
    scores += scores[:1]
    angles += angles[:1]

    ax.fill(angles, scores, color='#3498db', alpha=0.3)
    ax.plot(angles, scores, color='#3498db', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats)
    ax.set_title(f"Mechanism Profile: {report['mutation']}", pad=20)

    fig_path = "figures/mutation_profile.png"
    os.makedirs("figures", exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    return "\n".join(lines), fig_path


def design_combination(selected_mutations):
    """
    Gradio 回调：组合设计
    selected_mutations: 逗号分隔的突变字符串，如 "S43E,Y639F"
    """
    muts = [m.strip() for m in selected_mutations.split(",") if m.strip()]

    # 构建临时库
    reports = []
    for m in muts:
        if len(m) < 3:
            continue
        wt, site_str, mut_aa = m[0], m[1:-1], m[-1]
        site = int(site_str)
        r = PREDICTOR.predict(site, wt, mut_aa, prob_threshold=0.2)
        if r:
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
        return "Need at least 2 valid mutations for combination analysis.", None

    # 评估组合
    opt = CombinatorialOptimizer(reports, max_size=len(reports), min_site_dist=3)
    combo = opt._evaluate([r.mutation for r in reports])

    lines = [
        f"Combination: {' + '.join(combo['mutations'])}",
        f"Sites: {combo['sites']}",
        f"Fitness: {combo['fitness']:.2f}",
        f"Diversity: {combo['diversity']:.2f}",
        f"Synergy: {combo['synergy']:.2f}",
        f"Conflicts: {combo['conflicts']}",
        "",
        "Category Coverage:"
    ]
    for cat, score in sorted(combo['category_scores'].items(), key=lambda x: -x[1]):
        lines.append(f"  {cat}: {score:.2f}")

    if combo['conflicts'] > 0:
        lines.append("")
        lines.append("WARNING: Mechanism conflicts detected!")

    return "\n".join(lines), None


# ================== Gradio 界面构建 ==================
with gr.Blocks(title="T7 RNAP Mechanism Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔬 T7 RNAP 机制分类器 & 组突设计平台

    **基于机制理解（Mechanism-Aware）的蛋白质定向进化工具**

    输入位点或突变，模型解析其物理化学机制，并推荐机制互补的组合方案。
    """)

    with gr.Tab("🔍 位点扫描"):
        gr.Markdown("输入一个位点，自动扫描 19 种氨基酸替换，推荐最优突变。")
        with gr.Row():
            with gr.Column(scale=1):
                scan_site_input = gr.Number(label="位点 (Site)", value=43, precision=0)
                scan_target = gr.Dropdown(
                    choices=["All", "stability", "activity", "quality", "allostery", "promoter"],
                    value="All",
                    label="目标机制大类"
                )
                scan_btn = gr.Button("扫描", variant="primary")

            with gr.Column(scale=2):
                scan_table = gr.Dataframe(
                    headers=["Mutation", "Mechanism", "Category", "Prob", "Direction", "Magnitude"],
                    label="推荐突变排序"
                )
                scan_plot = gr.Image(label="机制覆盖图")

        scan_btn.click(
            fn=scan_single_site,
            inputs=[scan_site_input, scan_target],
            outputs=[scan_table, scan_plot]
        )

    with gr.Tab("🧬 突变解析"):
        gr.Markdown("预测指定突变的机制激活谱。")
        with gr.Row():
            with gr.Column(scale=1):
                pred_site = gr.Number(label="位点", value=639, precision=0)
                pred_wt = gr.Textbox(label="WT 氨基酸", value="Y", max_lines=1)
                pred_mut = gr.Textbox(label="突变氨基酸", value="F", max_lines=1)
                pred_btn = gr.Button("解析", variant="primary")

            with gr.Column(scale=2):
                pred_text = gr.Textbox(label="机制报告", lines=15)
                pred_plot = gr.Image(label="机制雷达图")

        pred_btn.click(
            fn=predict_mutation,
            inputs=[pred_site, pred_wt, pred_mut],
            outputs=[pred_text, pred_plot]
        )

    with gr.Tab("🔗 组合设计"):
        gr.Markdown("输入多个突变（逗号分隔），评估其机制互补性。")
        with gr.Row():
            with gr.Column(scale=1):
                combo_input = gr.Textbox(
                    label="突变列表",
                    placeholder="S43E, Q786M, Y639F",
                    value="S43E, Q786M, Y639F"
                )
                combo_btn = gr.Button("评估组合", variant="primary")

            with gr.Column(scale=2):
                combo_text = gr.Textbox(label="组合评估报告", lines=12)

        combo_btn.click(
            fn=design_combination,
            inputs=[combo_input],
            outputs=[combo_text, gr.State()]
        )

    gr.Markdown("""
    ---
    **使用说明**：
    1. **位点扫描**：不知道突变成什么？输入位点，模型遍历 19 种氨基酸，推荐机制最明确的突变。
    2. **突变解析**：已有候选突变？输入位点+WT+Mut，查看其机制激活谱。
    3. **组合设计**：把扫描出的好突变和已知突变组合，评估机制互补性和冲突。

    **注意**：当前为轻量版（One-hot 投影），换真实 ESM-2 后精度会大幅提升。
    """)

if __name__ == '__main__':
    demo.launch(share=False)  # share=True 可生成公网链接
