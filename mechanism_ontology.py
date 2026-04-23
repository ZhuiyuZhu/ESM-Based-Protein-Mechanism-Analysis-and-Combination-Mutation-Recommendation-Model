"""
机制本体定义 + 冲突规则
"""

MECHANISM_ONTOLOGY = {
    '1.1.1': ('cavity_filling', 'stability'),
    '1.1.2': ('salt_bridge', 'stability'),
    '1.1.3': ('helix_rigidification', 'stability'),
    '1.1.4': ('hydrophobic_core', 'stability'),
    '1.1.5': ('loop_ordering', 'stability'),
    '1.2.1': ('dna_binding_enhance', 'activity'),
    '1.2.2': ('ntp_recognition', 'activity'),
    '1.2.3': ('metal_coordination', 'activity'),
    '1.2.4': ('fingers_closing', 'activity'),
    '1.2.5': ('ec_stability', 'activity'),
    '1.3.1': ('pbd_rotation', 'promoter'),
    '1.3.2': ('specificity_loop', 'promoter'),
    '1.3.3': ('dna_melting', 'promoter'),
    '1.4.1': ('rna_release', 'quality'),
    '1.4.2': ('end_homogeneity', 'quality'),
    '1.4.3': ('fidelity_enhance', 'quality'),
    '1.4.4': ('self_priming_inhibit', 'quality'),
    '1.5.1': ('ntd_ctd_coupling', 'allostery'),
    '1.5.2': ('foot_regulation', 'allostery'),
    '1.5.3': ('thumb_displacement', 'allostery'),
}

ALL_MECHANISMS = list(MECHANISM_ONTOLOGY.keys())
NUM_MECHANISMS = len(ALL_MECHANISMS)

CATEGORY_MAP = {
    'stability': ['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5'],
    'activity': ['1.2.1', '1.2.2', '1.2.3', '1.2.4', '1.2.5'],
    'promoter': ['1.3.1', '1.3.2', '1.3.3'],
    'quality': ['1.4.1', '1.4.2', '1.4.3', '1.4.4'],
    'allostery': ['1.5.1', '1.5.2', '1.5.3'],
}

CATEGORY_NAMES = list(CATEGORY_MAP.keys())
NUM_CATEGORIES = len(CATEGORY_NAMES)

MECH_TO_CAT_IDX = {}
for cat_idx, (cat_name, mechs) in enumerate(CATEGORY_MAP.items()):
    for m in mechs:
        MECH_TO_CAT_IDX[m] = cat_idx

CONFLICT_PAIRS = [
    ('1.2.1', '1.4.1'),
    ('1.2.4', '1.4.3'),
    ('1.1.3', '1.5.1'),
]

def detect_conflicts(activated_mechanisms):
    activated_set = set(activated_mechanisms)
    conflicts = []
    for m1, m2 in CONFLICT_PAIRS:
        if m1 in activated_set and m2 in activated_set:
            conflicts.append((m1, m2, f"{MECHANISM_ONTOLOGY[m1][0]} vs {MECHANISM_ONTOLOGY[m2][0]}"))
    return conflicts
