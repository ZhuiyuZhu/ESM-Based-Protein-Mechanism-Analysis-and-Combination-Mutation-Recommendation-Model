#!/usr/bin/env python3
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict

import pandas as pd
import torch

from mechanism_ontology import (MECHANISM_ONTOLOGY, ALL_MECHANISMS,
                                CATEGORY_MAP, CATEGORY_NAMES, detect_conflicts)

# ================== 轻量 ESM 编码器（和 predict.py 完全一致）====================
AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}

PROJ_PATH = 'data/processed/esm_embeddings/projection_matrix.pt'
if os.path.exists(PROJ_PATH):
    PROJ_MATRIX = torch.load(PROJ_PATH)
else:
    torch.manual_seed(42)
    PROJ_MATRIX = torch.nn.init.orthogonal_(torch.randn(20, 1280))
    os.makedirs('data/processed/esm_embeddings', exist_ok=True)
    torch.save(PROJ_MATRIX, PROJ_PATH)


def encode_sequence(seq):
    indices = [AA_TO_IDX.get(aa, 0) for aa in seq]
    onehot = torch.zeros(len(seq), 20)
    for i, idx in enumerate(indices):
        onehot[i, idx] = 1.0
    return onehot @ PROJ_MATRIX


T7_WT_SEQ = (
    "MNTINIAKNDFSDIELAAIPFNTLADHYGERLAREQLALEHESYEMGEARFRKMFERQLK"
    "AGEVADNAAAKPLITTLLPKMIARINDWFEEVKAKRGKRPTAFQFLQEIKPEAVAYITIK"
    "TTLACLTSADNTTVQAVASAIGRAIEDEARFGRIRDLEAKHFKKNVEEQLNKRVGHVYKK"
    "AFMQVVEADMLSKGLLGGEAWSSWHKEDSIHVGVRCIEMLIESTGMVSLHRQNAGVVGQD"
    "SETIELAPEYAEAIATRAGALAGISPMFQPCVVPPKPWTGITGGGYWANGRRPLALVRTH"
    "SKKALMRYEDVYMPEVYKAINIAQNTAWKINKKVLAVANVITKWKHCPVEDIPAIEREEL"
    "PMKPEDIDMNPEALTAWKRAAAAVYRKDKARKSRRISLEFMLEQANKFANHKAIWFPYNM"
    "DWRGRVYAVSMFNPQGNDMTKGLLTLAKGKPIGKEGYYWLKIHGANCAGVDKVPFPERIK"
    "FIEENHENIMACAKSPLENTWWAEQDSPFCFLAFCFEYAGVQHHGLSYNCSLPLAFDGSC"
    "SGIQHFSAMLRDEVGGRAVNLLPSETVQDIYGIVAKKVNEILQADAINGTDNEVVTVTDE"
    "NTGEISEKVKLGTKALAGQWLAYGVTRSVTKRSVMTLAYGSKEFGFRQQVLEDTIQPAID"
    "SGKGLMFTQPNQAAGYMAKLIWESVSVTVVAAVEAMNWLKSAAKLLAAEVKDKKTGEILR"
    "KRCAVHWVTPDGFPVWQEYKKPIQTRLNLMFLGQFRLQPTINTNKDSEIDAHKQESGIAP"
    "NFVHSQDGSHLRKTVVWAHEKYGIESFALIHDSFGTIPADAANLFKAVRETMVDTYESCD"
    "VLADFYDQFADQLHESQLDKMPALPAKGNLNLRDILESDFAFA"
)


@dataclass
class MutantReport:
    mutation: str
    site: int
    mechanism_probs: Dict[str, float]
    dominant_mech: str
    effect_magnitude: float
    representation: np.ndarray
    category_scores: Dict[str, float]


class CombinatorialOptimizer:
    def __init__(self, library: List[MutantReport], max_size=4, min_site_dist=5):
        self.library = {r.mutation: r for r in library}
        self.max_size = max_size
        self.min_site_dist = min_site_dist

        self.cat_to_mutants = defaultdict(list)
        for r in library:
            for cat, score in r.category_scores.items():
                if score > 0.2:
                    self.cat_to_mutants[cat].append(r.mutation)

    def _spatial_score(self, sites: List[int]) -> float:
        if len(sites) <= 1:
            return 1.0
        min_dist = min(abs(sites[i] - sites[j])
                       for i in range(len(sites)) for j in range(i + 1, len(sites)))
        if min_dist < self.min_site_dist:
            return 0.0
        return min(min_dist / 50.0, 1.0)

    def _evaluate(self, mutations: List[str]):
        reports = [self.library[m] for m in mutations]
        sites = [r.site for r in reports]

        merged = {}
        for mech in ALL_MECHANISMS:
            probs = [r.mechanism_probs.get(mech, 0) for r in reports]
            merged[mech] = 1.0 - np.prod([1.0 - p for p in probs])

        activated = [m for m, p in merged.items() if p > 0.3]
        conflicts = detect_conflicts(activated)

        cat_scores = defaultdict(float)
        for m, p in merged.items():
            cat = MECHANISM_ONTOLOGY[m][1]
            cat_scores[cat] += p

        num_cats = len(set(MECHANISM_ONTOLOGY[m][1] for m in activated))
        diversity = num_cats / 5.0

        reps = np.stack([r.representation for r in reports])
        reps_norm = reps / (np.linalg.norm(reps, axis=1, keepdims=True) + 1e-8)
        sim = np.dot(reps_norm, reps_norm.T)
        mask = ~np.eye(len(sim), dtype=bool)
        synergy = 1.0 - sim[mask].mean()

        base = sum(r.effect_magnitude for r in reports)
        fitness = base * (1 + synergy) * (1 + diversity * 2) - len(conflicts) * 2.0

        spatial = self._spatial_score(sites)
        if spatial == 0:
            fitness = -999

        return {
            'mutations': mutations,
            'sites': sites,
            'fitness': fitness,
            'diversity': diversity,
            'synergy': synergy,
            'conflicts': len(conflicts),
            'category_scores': dict(cat_scores),
            'activated': activated,
            'reports': reports
        }

    def generate_initial_population(self, pop_size=100):
        population = []
        mutants = list(self.library.keys())

        for _ in range(pop_size // 2):
            size = random.randint(2, min(self.max_size, len(mutants)))
            combo = random.sample(mutants, size)
            candidate = self._evaluate(combo)
            if candidate['fitness'] > -900:
                population.append(candidate)

        categories = ['stability', 'activity', 'promoter', 'quality', 'allostery']
        for _ in range(pop_size // 2):
            combo = []
            for cat in random.sample(categories, random.randint(2, min(5, self.max_size))):
                if self.cat_to_mutants[cat]:
                    m = random.choice(self.cat_to_mutants[cat])
                    if m not in combo:
                        combo.append(m)
            if len(combo) >= 2:
                candidate = self._evaluate(combo)
                if candidate['fitness'] > -900:
                    population.append(candidate)

        return population

    def crossover(self, p1, p2):
        pool = list(set(p1['mutations'] + p2['mutations']))
        if len(pool) <= self.max_size:
            return self._evaluate(pool)

        selected = []
        covered = set()
        for m in sorted(pool, key=lambda x: sum(
                self.library[x].mechanism_probs.get(mech, 0)
                for mech in self.library[x].mechanism_probs
        ), reverse=True):
            if len(selected) >= self.max_size:
                break
            selected.append(m)
            for mech, prob in self.library[m].mechanism_probs.items():
                if prob > 0.3:
                    covered.add(mech)
        return self._evaluate(selected)

    def mutate(self, candidate, rate=0.3):
        if random.random() > rate:
            return candidate
        combo = candidate['mutations'].copy()
        idx = random.randint(0, len(combo) - 1)
        old = combo[idx]
        old_cat = MECHANISM_ONTOLOGY[self.library[old].dominant_mech][1]
        alts = [m for m in self.cat_to_mutants.get(old_cat, []) if m not in combo]
        if alts:
            combo[idx] = random.choice(alts)
        return self._evaluate(combo)

    def optimize(self, generations=50, pop_size=100, elite=10):
        pop = self.generate_initial_population(pop_size)
        pop.sort(key=lambda x: x['fitness'], reverse=True)
        best = pop[0]

        for gen in range(generations):
            new_pop = pop[:elite]
            while len(new_pop) < pop_size:
                t1 = random.sample(pop[:pop_size // 2], 3)
                p1 = max(t1, key=lambda x: x['fitness'])
                t2 = random.sample(pop[:pop_size // 2], 3)
                p2 = max(t2, key=lambda x: x['fitness'])
                child = self.crossover(p1, p2)
                if child['fitness'] > -900:
                    child = self.mutate(child)
                    new_pop.append(child)
            pop = sorted(new_pop, key=lambda x: x['fitness'], reverse=True)
            if pop[0]['fitness'] > best['fitness']:
                best = pop[0]
            if gen % 10 == 0:
                print(f"Gen {gen}: Best={best['fitness']:.2f}, Div={best['diversity']:.2f}, Conf={best['conflicts']}")

        return pop[:10]


def build_library_from_annotations():
    """从训练库构建突变报告"""
    from predict import T7MechPredictor

    df = pd.read_csv('data/processed/annotations.csv')
    predictor = T7MechPredictor()

    reports = []
    for _, row in df.iterrows():
        site = int(row['site'])
        wt = row['wt_aa']
        mut = row['mut_aa']
        mut_str = f"{wt}{site}{mut}"

        r = predictor.predict(site, wt, mut, prob_threshold=0.3)
        if r is None:
            continue  # ← 关键修复：跳过 WT 不匹配的突变

        reports.append(MutantReport(
            mutation=mut_str,
            site=site,
            mechanism_probs={a['id']: a['prob'] for a in r['activated_mechanisms']},
            dominant_mech=r['dominant_mechanism'],
            effect_magnitude=sum(a['magnitude'] for a in r['activated_mechanisms']),
            representation=r['representation'],
            category_scores=r['category_scores']
        ))

    return reports


def run_combinatorial():
    print("Building library from training annotations...")
    library = build_library_from_annotations()
    print(f"Library size: {len(library)} mutants")

    if len(library) < 10:
        print("ERROR: Too few mutants in library. Need at least 10.")
        return

    opt = CombinatorialOptimizer(library, max_size=4, min_site_dist=5)
    top = opt.optimize(generations=30, pop_size=50)

    print("\n" + "=" * 60)
    print("TOP 5 COMBINATION DESIGNS")
    print("=" * 60)
    for i, combo in enumerate(top[:5], 1):
        print(f"\n[Rank {i}] Fitness: {combo['fitness']:.2f}")
        print(f"  Mutations: {' + '.join(combo['mutations'])}")
        print(f"  Sites: {combo['sites']}")
        print(
            f"  Diversity: {combo['diversity']:.2f} | Synergy: {combo['synergy']:.2f} | Conflicts: {combo['conflicts']}")
        print(f"  Category Coverage:")
        for cat, score in sorted(combo['category_scores'].items(), key=lambda x: -x[1]):
            print(f"    {cat:<12} {score:.2f}")


if __name__ == '__main__':
    run_combinatorial()