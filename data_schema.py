#!/usr/bin/env python3
"""
主动学习数据规范
用户提交的每一条突变数据必须包含以下字段
"""

from dataclasses import dataclass
from typing import Optional, Dict
import json

@dataclass
class ExperimentalDatum:
    """
    单条实验数据记录
    """
    # 基础信息（必填）
    protein_id: str          # 用户自定义ID，如 "Lab_20250423_001"
    mutation: str            # 突变字符串，如 "S43E"
    site: int                # 1-indexed
    wt_aa: str               # 野生型氨基酸
    mut_aa: str              # 突变氨基酸
    
    # 实验条件（必填，用于归一化对比）
    assay_temperature: float   # 实验温度，如 37.0 (°C)
    assay_buffer: str         # 缓冲液，如 "Tris-HCl pH 7.9"
    
    # 定量指标（至少填一项，建议全填）
    # 1. 结构稳定性
    tm_value: Optional[float] = None           # 熔解温度 (°C)，如 52.3
    tm_delta: Optional[float] = None          # 相对WT的 ΔTm (°C)，如 +3.5
    
    # 2. 催化活性
    kcat_km: Optional[float] = None          # 催化效率 (M⁻¹s⁻¹)
    activity_relative: Optional[float] = None  # 相对WT活性倍数，如 1.2
    
    # 3. 产物质量
    dsRNA_ratio: Optional[float] = None        # dsRNA 占比 (%)，如 0.5
    fidelity_index: Optional[float] = None     # 保真度指数，如 0.95
    
    # 4. 其他
    yield_mg_per_L: Optional[float] = None     # 产量 mg/L
    half_life_min: Optional[float] = None        # 半衰期 (min)
    
    # 机制标签（用户可选填，如果不填由模型自动推断）
    user_mechanism_label: Optional[str] = None   # 如 "1.1.1"
    
    # 元数据
    experimenter: str = "anonymous"              # 实验者姓名/组名
    notes: str = ""                              # 备注
    
    def validate(self) -> bool:
        """校验数据完整性"""
        assert self.site >= 1 and self.site <= 883, "位点超出 1-883 范围"
        assert len(self.wt_aa) == 1 and len(self.mut_aa) == 1, "氨基酸必须为单字母"
        assert self.wt_aa != self.mut_aa, "WT 和 Mut 不能相同"
        
        # 至少有一个定量指标
        has_metric = any([
            self.tm_value, self.tm_delta, self.kcat_km,
            self.activity_relative, self.dsRNA_ratio,
            self.fidelity_index, self.yield_mg_per_L, self.half_life_min
        ])
        assert has_metric, "至少填写一项定量实验指标"
        return True
    
    def to_dict(self) -> Dict:
        return {
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
            'notes': self.notes
        }
    
    @classmethod
    def from_dict(cls, d: Dict):
        return cls(**{k: d.get(k) for k in cls.__dataclass_fields__.keys()})
