import os
import torch
import pandas as pd
from torch.utils.data import Dataset

# Fix: flat imports
from mechanism_ontology import ALL_MECHANISMS, MECH_TO_CAT_IDX
from structure_utils import MockStructureProcessor


class T7MutationDataset(Dataset):
    def __init__(self, annotation_csv, esm_dir, use_mock=False, split='train'):
        self.df = pd.read_csv(annotation_csv)
        if 'split' in self.df.columns:
            self.df = self.df[self.df['split'] == split].reset_index(drop=True)

        self.esm_dir = esm_dir
        self.use_mock = use_mock
        self.processor = MockStructureProcessor(length=883)
        self.edge_index, self.edge_attr = self.processor.build_graph()
        self.struct_feat = self.processor.extract_node_features()
        self.mech_list = ALL_MECHANISMS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        site = int(row['site']) - 1

        if self.use_mock:
            esm_wt = torch.randn(883, 1280)
            esm_mut = torch.randn(883, 1280)
            struct_feat = self.struct_feat
            edge_index = self.edge_index
            mutation_mask = self.processor.get_mutation_context(site)
        else:
            pid = row['protein_id']
            esm_wt = torch.load(os.path.join(self.esm_dir, f"{pid}_wt.pt"))
            esm_mut = torch.load(os.path.join(self.esm_dir, f"{pid}_mut.pt"))
            struct_feat = self.struct_feat
            edge_index = self.edge_index
            mutation_mask = self.processor.get_mutation_context(site)

        multilabel = torch.tensor([float(row.get(m, 0)) for m in self.mech_list], dtype=torch.float)
        dominant = MECH_TO_CAT_IDX.get(row.get('dominant_mech', '1.1.1'), 0)

        effect_dir = torch.tensor([float(row.get(f'effect_sign_{m}', 0)) for m in self.mech_list])
        effect_mag = torch.tensor([float(row.get(f'effect_size_{m}', 0)) for m in self.mech_list])

        labels = {
            'multilabel': multilabel,
            'dominant_category': torch.tensor(dominant, dtype=torch.long),
            'effect_direction': effect_dir,
            'effect_magnitude': effect_mag
        }

        return {
            'esm_wt': esm_wt,
            'esm_mut': esm_mut,
            'struct_feat': struct_feat,
            'edge_index': edge_index,
            'mutation_mask': mutation_mask,
            'labels': labels,
            'mutation': row['mutation']
        }


def collate_fn(batch):
    esm_wt = torch.stack([b['esm_wt'] for b in batch])
    esm_mut = torch.stack([b['esm_mut'] for b in batch])
    struct_feat = batch[0]['struct_feat']
    edge_index = batch[0]['edge_index']
    mutation_mask = torch.stack([b['mutation_mask'] for b in batch])

    labels = {k: torch.stack([b['labels'][k] for b in batch]) for k in batch[0]['labels']}

    return {
        'esm_wt': esm_wt,
        'esm_mut': esm_mut,
        'struct_feat': struct_feat,
        'edge_index': edge_index,
        'mutation_mask': mutation_mask,
        'labels': labels
    }
