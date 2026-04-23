# ESM-Based-Protein-Mechanism-Analysis-and-Combination-Mutation-Recommendation-Model
Mechanism-aware directed evolution platform for T7 RNA polymerase.  Explicitly disentangles mutation effects into 15 biophysical mechanisms  (cavity filling, fidelity, allostery, etc.) and designs combinatorial  mutants via genetic algorithm.
# 🔬 T7 RNAP Mechanism-Aware Directed Evolution

基于**机制理解（Mechanism-Aware）**的 T7 RNA 聚合酶定向进化平台。

> **核心创新**：不同于传统黑箱模型（仅预测 ΔΔG 或活性分数），本系统显式解析突变背后的物理化学机制（空腔填充、保真度、变构通讯等），并基于**机制互补性**设计组合突变，实现可解释的蛋白质工程。

---

## 🎯 核心能力
| 功能 | 说明 | 示例 |
|------|------|------|
| **🔍 位点扫描** | 输入位点，遍历 19 种氨基酸替换，推荐机制最优突变 | Site 43 → `S43E` (cavity filling) |
| **🧬 突变解析** | 预测突变的 15 种细粒度机制激活谱 | `Y639F` → `1.4.3 fidelity_enhance` |
| **🔗 组合设计** | 遗传算法优化机制互补的突变组合 | `S43E + Y639F` → 跨大类互补 |
| **📊 可视化** | Gradio Web UI，零代码交互 | 雷达图 + 机制热图 |

---

## 🚀 快速开始

### 环境要求
- Python 3.10 ~ 3.13（当前 3.13 使用轻量版 ESM）
- 8GB+ 内存（轻量版无压力）
- Windows / Linux / macOS

### 安装依赖
```
pip install -r requirements.txt

启动 Web 界面
bash
python app.py
浏览器自动打开  http://127.0.0.1:7860

命令行：位点扫描
bash
python site_scanner.py

命令行：训练模型
bash
python train.py --config configs/default.yaml --output_dir outputs/exp_001
```
🏗️ 技术架构
```
输入序列/突变
    │
    ├──► 轻量 ESM 编码器 (One-hot + 正交投影)  ──► [883, 1280]
    │
    ├──► 结构特征提取器 (Mock/真实 PDB)        ──► [883, 20]
    │
    └──► 图神经网络 (GNN)                       ──► [883, 128]
              │
              └──► 突变差异编码器 (WT vs MUT)    ──► [256]
                        │
                        └──► 机制解耦头 (15 Experts)
                                  │
                                  ├──► 1.1.x 结构稳定性
                                  ├──► 1.2.x 催化活性
                                  ├──► 1.3.x 启动子识别
                                  ├──► 1.4.x 产物质量
                                  └──► 1.5.x 变构通讯
```
机制本体（Mechanism Ontology）
```
大类	    编号段	        目标性状	       典型实验指标	
结构稳定性	1.1.1  1.1.5	热稳定性 (Tm)   Tm 上升、半衰期延长	
催化活性	    1.2.1  1.2.5	转录效率	       kcat/Km、延伸速率	
启动子识别	1.3.1  1.3.3	特异性	       启动子结合亲和力	
产物质量	    1.4.1  1.4.4	RNA 质量	    dsRNA 减少、保真度	
变构通讯	    1.5.1  1.5.3	构象耦合	        域间协同性
```

⚠️ 当前局限性（诚实声明）
本项目处于概念验证（Proof-of-Concept）阶段，以下局限性将在后续版本中逐步解决：
1. ESM-2 为轻量版投影编码
 现状：使用 One-hot + 正交投影矩阵生成 1280 维特征，而非真实 ESM-2 650M 模型输出
 影响：序列语义捕获能力有限，mAP 约 0.69（真实 ESM-2 预期 0.85+）
 原因：Python 3.13 与  fair-esm  底层 C 扩展存在兼容性冲突，且 650M 模型需要 16GB+ 内存
 解决：迁移至 Python 3.10 环境，加载真实  esm2_t33_650M_UR50D  权重

2. 结构特征为 Mock 数据
 现状：使用随机生成的假坐标和假结构特征（空腔、SASA、B-factor）
 影响：模型无法利用真实几何/物理化学信息
 解决：接入  biopython  +  freesasa  解析 1CEZ/1MSW PDB 文件，提取真实空腔体积、氢键网络、盐桥等

3. 训练数据量较小
 现状：基于文献整理的 ~70 条突变数据（含数据增强）
 影响：部分稀有机制（如  1.3.x  启动子相关）样本不足，AP 偏低
 解决：系统性收集 M30、PROSS、定向进化轨迹等文献数据，扩充至 200+ 条

4. 无分子动力学（MD）动态特征
 现状：仅使用静态结构/序列特征
 影响：无法捕捉变构路径、域运动、柔性变化等动态机制
 解决：对关键突变进行短 MD 模拟，提取 RMSF、PCA 运动模式作为输入

5. 组突优化为离线批处理
 现状：需预先构建候选库，再运行遗传算法
 影响：无法实时响应用户输入的任意位点组合
 解决：开发 Streamlit 实时接口，支持动态添加/删除突变并即时评估

📈 性能基准（当前版本）
在 8 条测试集突变上：
指标	结果	
Mean Average Precision (mAP)	0.69	
1.1.1 cavity_filling AP	1.00	
1.4.3 fidelity_enhance AP	1.00	
主导大类准确率	87.5%	
机制相关性（越低越好）	0.57	

🗺️ 路线图（Roadmap）
阶段	目标	预期提升	
v0.1 ✅	机制分类器 + Gradio UI + 组突优化	mAP 0.69	
v0.2	真实 ESM-2 + 真实 PDB 结构特征	mAP 0.85+	
v0.3	扩充训练集至 200+ 条，引入对比学习	mAP 0.90+	
v0.4	MD 动态特征 + 机制因果推理	可解释性质变	
v0.5	Streamlit Cloud 部署 + 实验反馈闭环	在线工具	

📂 项目结构
```
├── app.py                    # Gradio Web 界面
├── train.py                  # 训练入口
├── evaluate.py               # 评估脚本
├── predict.py                # 单突变机制解析
├── site_scanner.py           # 位点扫描器
├── combinatorial.py          # 组突遗传算法
├── classifier.py             # T7-MechClassifier 模型
├── encoders.py               # ESM + GNN 编码器
├── mechanism_heads.py        # 15 个机制专家头
├── mechanism_ontology.py     # 机制本体定义
├── structure_utils.py        # 结构处理工具
├── dataset.py                # PyTorch Dataset
├── configs/
│   └── default.yaml          # 训练配置
├── data/
│   └── processed/
│       └── annotations.csv   # 机制标注数据集
├── scripts/
│   ├── generate_real_data.py # 数据生成
│   ├── precompute_esm_light.py # 轻量 ESM 预计算
│   └── fix_annotations.py    # 数据修正
└── outputs/
    └── exp_001/
        └── best_model.pt     # 训练好的权重
```
🤝 引用与致谢
 
ESM-2: facebookresearch/esm
 
T7 RNAP 结构数据: RCSB PDB 1CEZ
 
机制本体设计参考: M30 定向进化 (Jiang et al., 2026), PROSS 算法

📧 联系
如有问题或建议，欢迎提交 Issue 或 Pull Request。
E-mail：zhuzhuiyu@stu.xjtu.edu.cn
